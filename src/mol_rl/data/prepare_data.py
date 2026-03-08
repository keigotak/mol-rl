"""
Data preparation pipeline for molecular generation.

Downloads and preprocesses molecular datasets (MOSES benchmark from ChEMBL),
converts SMILES to SELFIES, filters for drug-like molecules, and creates
train/val/test splits.

Usage:
    python -m mol_rl.data.prepare_data --output_dir data/processed
    python -m mol_rl.data.prepare_data --source chembl --max_molecules 500000
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMILES Filtering
# ---------------------------------------------------------------------------

def is_valid_smiles(smi: str) -> bool:
    """Check if a SMILES string is valid and can be parsed by RDKit."""
    if not smi or not isinstance(smi, str):
        return False
    mol = Chem.MolFromSmiles(smi)
    return mol is not None


def passes_drug_filter(smi: str,
                       max_mw: float = 500.0,
                       max_logp: float = 5.0,
                       max_atoms: int = 50,
                       min_atoms: int = 5) -> bool:
    """Apply basic drug-likeness filters (loose Lipinski-like)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    n_atoms = mol.GetNumHeavyAtoms()

    return (mw <= max_mw and
            logp <= max_logp and
            min_atoms <= n_atoms <= max_atoms)


def canonicalize_smiles(smi: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# ---------------------------------------------------------------------------
# SELFIES Conversion
# ---------------------------------------------------------------------------

def smiles_to_selfies(smi: str) -> Optional[str]:
    """Convert SMILES to SELFIES. Returns None if conversion fails."""
    try:
        return sf.encoder(smi)
    except Exception:
        return None


def selfies_to_smiles(sel: str) -> Optional[str]:
    """Convert SELFIES back to SMILES. Returns None if conversion fails."""
    try:
        return sf.decoder(sel)
    except Exception:
        return None


def get_selfies_tokens(sel: str) -> list[str]:
    """Split a SELFIES string into individual tokens."""
    return list(sf.split_selfies(sel))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_moses_dataset() -> pd.DataFrame:
    """
    Load the MOSES benchmark dataset.

    MOSES (Molecular Sets) is a curated subset of ~1.9M molecules from ZINC
    Clean, filtered for drug-likeness. Standard benchmark for molecular generation.

    Downloads from: https://github.com/molecularsets/moses
    """
    logger.info("Loading MOSES benchmark dataset...")

    # MOSES dataset URLs (train/test/test_scaffolds)
    urls = {
        "train": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv",
        "test": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv",
        "test_scaffolds": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv",
    }

    dfs = {}
    for split, url in urls.items():
        try:
            df = pd.read_csv(url)
            dfs[split] = df
            logger.info(f"  {split}: {len(df)} molecules")
        except Exception as e:
            logger.warning(f"  Failed to download {split} from URL: {e}")
            logger.info(f"  Trying alternative loading method...")
            dfs[split] = None

    # If download fails, try loading from local files
    if all(v is None for v in dfs.values()):
        raise RuntimeError(
            "Could not download MOSES dataset. Please download manually from "
            "https://github.com/molecularsets/moses and place CSV files in data/raw/"
        )

    return dfs


def load_smiles_file(filepath: str) -> list[str]:
    """Load SMILES from a text file (one per line) or CSV."""
    path = Path(filepath)
    if path.suffix == ".csv":
        df = pd.read_csv(filepath)
        # Try common column names
        for col in ["SMILES", "smiles", "canonical_smiles", "Smiles"]:
            if col in df.columns:
                return df[col].dropna().tolist()
        # Fall back to first column
        return df.iloc[:, 0].dropna().tolist()
    else:
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Processing Pipeline
# ---------------------------------------------------------------------------

def _process_single_smiles(smi: str) -> Optional[dict]:
    """Convert a single SMILES to a record dict. Used by multiprocessing Pool."""
    sel = smiles_to_selfies(smi)
    if sel is None:
        return None
    tokens = get_selfies_tokens(sel)
    mol = Chem.MolFromSmiles(smi)
    qed_score = QED.qed(mol) if mol else 0.0
    return {
        "smiles": smi,
        "selfies": sel,
        "selfies_tokens": " ".join(tokens),
        "n_tokens": len(tokens),
        "qed": round(qed_score, 4),
    }


def process_molecules(smiles_list: list[str],
                      apply_filter: bool = True,
                      max_molecules: Optional[int] = None) -> pd.DataFrame:
    """
    Full processing pipeline: validate, filter, canonicalize, convert to SELFIES.

    Args:
        smiles_list: Raw SMILES strings
        apply_filter: Whether to apply drug-likeness filters
        max_molecules: Maximum number of molecules to keep

    Returns:
        DataFrame with columns: [smiles, selfies, selfies_tokens, n_tokens, qed]
    """
    logger.info(f"Processing {len(smiles_list)} molecules...")

    # Step 1: Canonicalize and deduplicate
    canonical = []
    seen = set()
    for smi in smiles_list:
        csmi = canonicalize_smiles(smi)
        if csmi and csmi not in seen:
            seen.add(csmi)
            canonical.append(csmi)

    logger.info(f"  After canonicalization + dedup: {len(canonical)}")

    # Step 2: Drug-likeness filter
    if apply_filter:
        filtered = [s for s in canonical if passes_drug_filter(s)]
        logger.info(f"  After drug-likeness filter: {len(filtered)}")
    else:
        filtered = canonical

    # Step 2.5: Subsample early to avoid expensive SELFIES conversion on full dataset
    if max_molecules and len(filtered) > max_molecules:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(filtered), max_molecules, replace=False)
        filtered = [filtered[i] for i in sorted(indices)]
        logger.info(f"  Subsampled to {len(filtered)} before SELFIES conversion")

    # Step 3: Convert to SELFIES (parallelized)
    from multiprocessing import Pool, cpu_count

    n_workers = min(cpu_count(), 8)
    logger.info(f"  Converting to SELFIES with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(_process_single_smiles, filtered, chunksize=1000)

    records = [r for r in results if r is not None]
    failed_conversions = len(filtered) - len(records)

    logger.info(f"  Failed SELFIES conversions: {failed_conversions}")
    logger.info(f"  Final dataset size: {len(records)}")

    df = pd.DataFrame(records)
    return df


def build_vocabulary(df: pd.DataFrame) -> dict:
    """
    Build SELFIES token vocabulary from the dataset.

    Returns:
        Dictionary with token-to-id and id-to-token mappings.
    """
    all_tokens = set()
    for tokens_str in df["selfies_tokens"]:
        all_tokens.update(tokens_str.split())

    # Sort for reproducibility
    sorted_tokens = sorted(all_tokens)

    # Special tokens
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

    token2id = {t: i for i, t in enumerate(special_tokens)}
    for t in sorted_tokens:
        if t not in token2id:
            token2id[t] = len(token2id)

    id2token = {v: k for k, v in token2id.items()}

    logger.info(f"  Vocabulary size: {len(token2id)} "
                f"({len(special_tokens)} special + {len(sorted_tokens)} SELFIES tokens)")

    return {
        "token2id": token2id,
        "id2token": id2token,
        "special_tokens": special_tokens,
    }


def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the dataset."""
    stats = {
        "n_molecules": len(df),
        "n_tokens_mean": round(df["n_tokens"].mean(), 1),
        "n_tokens_median": int(df["n_tokens"].median()),
        "n_tokens_max": int(df["n_tokens"].max()),
        "n_tokens_p95": int(np.percentile(df["n_tokens"], 95)),
        "n_tokens_p99": int(np.percentile(df["n_tokens"], 99)),
        "qed_mean": round(df["qed"].mean(), 4),
        "qed_median": round(df["qed"].median(), 4),
        "qed_std": round(df["qed"].std(), 4),
    }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare molecular dataset for RL training")
    parser.add_argument("--source", choices=["moses", "file"], default="moses",
                        help="Data source: 'moses' for MOSES benchmark, 'file' for custom SMILES file")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to SMILES file (required if source=file)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--max_molecules", type=int, default=None,
                        help="Maximum number of molecules to keep")
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip drug-likeness filtering")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.source == "moses":
        dfs = load_moses_dataset()
        train_smiles = dfs["train"]["SMILES"].tolist() if dfs["train"] is not None else []
        test_smiles = dfs["test"]["SMILES"].tolist() if dfs["test"] is not None else []

        if not train_smiles:
            raise RuntimeError("Failed to load MOSES training data")

        # Process train and test separately
        logger.info("=" * 60)
        logger.info("Processing TRAINING set")
        logger.info("=" * 60)
        train_df = process_molecules(train_smiles,
                                     apply_filter=not args.no_filter,
                                     max_molecules=args.max_molecules)

        # Split train into train/val
        n_val = int(len(train_df) * args.val_ratio)
        indices = np.random.permutation(len(train_df))
        val_df = train_df.iloc[indices[:n_val]].reset_index(drop=True)
        train_df = train_df.iloc[indices[n_val:]].reset_index(drop=True)

        logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}")

        if test_smiles:
            logger.info("=" * 60)
            logger.info("Processing TEST set")
            logger.info("=" * 60)
            test_df = process_molecules(test_smiles, apply_filter=not args.no_filter)
        else:
            test_df = None

    elif args.source == "file":
        if not args.input_file:
            raise ValueError("--input_file required when source=file")
        smiles_list = load_smiles_file(args.input_file)
        full_df = process_molecules(smiles_list,
                                    apply_filter=not args.no_filter,
                                    max_molecules=args.max_molecules)

        # Train/val/test split
        n = len(full_df)
        indices = np.random.permutation(n)
        n_test = int(n * 0.1)
        n_val = int(n * args.val_ratio)

        test_df = full_df.iloc[indices[:n_test]].reset_index(drop=True)
        val_df = full_df.iloc[indices[n_test:n_test + n_val]].reset_index(drop=True)
        train_df = full_df.iloc[indices[n_test + n_val:]].reset_index(drop=True)

        logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build vocabulary from training set
    logger.info("=" * 60)
    logger.info("Building vocabulary")
    logger.info("=" * 60)
    vocab = build_vocabulary(train_df)

    # Compute stats
    stats = compute_dataset_stats(train_df)
    logger.info(f"Dataset stats: {stats}")

    # Save everything
    import json

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    if test_df is not None:
        test_df.to_csv(output_dir / "test.csv", index=False)

    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"All data saved to {output_dir}/")
    logger.info("Files: train.csv, val.csv, test.csv, vocab.json, stats.json")


if __name__ == "__main__":
    main()
