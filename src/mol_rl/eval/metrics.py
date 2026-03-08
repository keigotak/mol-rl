"""
Evaluation metrics for molecular generation models.

Computes standard benchmarks: validity, uniqueness, novelty, diversity,
and property distributions (QED, SA, LogP, MW).

Usage:
    from mol_rl.eval.metrics import evaluate_model

    metrics = evaluate_model(
        model, tokenizer,
        n_samples=10000,
        training_smiles=set(train_df["smiles"]),
    )
    print(metrics)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import selfies as sf
import torch
from rdkit import Chem

from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
from mol_rl.models.rewards import (
    RewardFunction,
    compute_qed,
    compute_sa_score,
    compute_logp,
    compute_molecular_weight,
    internal_diversity,
    novelty,
    uniqueness,
    validity_rate,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Container for all generation evaluation metrics."""
    n_generated: int = 0
    n_valid: int = 0
    n_unique: int = 0
    validity: float = 0.0
    uniqueness: float = 0.0
    novelty: float = 0.0
    diversity: float = 0.0

    # Property distributions (over valid molecules)
    qed_mean: float = 0.0
    qed_std: float = 0.0
    sa_mean: float = 0.0
    sa_std: float = 0.0
    logp_mean: float = 0.0
    logp_std: float = 0.0
    mw_mean: float = 0.0
    mw_std: float = 0.0

    # Reward
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_max: float = 0.0

    # Lists for detailed analysis
    valid_smiles: list[str] = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        """Return a dict of scalar metrics (excludes lists)."""
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, list)}


@torch.no_grad()
def generate_molecules(
    model,
    tokenizer: SelfiesTokenizer,
    n_samples: int,
    device: torch.device,
    max_length: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    batch_size: int = 64,
) -> list[str]:
    """
    Generate SELFIES strings from a model.

    Returns list of SELFIES strings (length = n_samples).
    """
    model.eval()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    all_selfies = []
    use_amp = device.type == "cuda"

    while len(all_selfies) < n_samples:
        current_bs = min(batch_size, n_samples - len(all_selfies))

        input_ids = torch.full(
            (current_bs, 1), bos_id, dtype=torch.long, device=device
        )

        for step in range(max_length - 1):
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                top_k_vals = torch.topk(logits, top_k)[0]
                logits[logits < top_k_vals[..., -1, None]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (next_token.squeeze(1) == eos_id).all():
                break

        for seq in input_ids:
            selfies_str = tokenizer.decode(seq, skip_special_tokens=True)
            all_selfies.append(selfies_str)

    return all_selfies[:n_samples]


def selfies_to_smiles_list(selfies_list: list[str]) -> list[Optional[str]]:
    """Convert SELFIES to SMILES, returning None for failures."""
    smiles_list = []
    for sel in selfies_list:
        try:
            smi = sf.decoder(sel)
            if smi and Chem.MolFromSmiles(smi) is not None:
                smiles_list.append(smi)
            else:
                smiles_list.append(None)
        except Exception:
            smiles_list.append(None)
    return smiles_list


def compute_property_stats(valid_smiles: list[str]) -> dict:
    """Compute property distributions over valid SMILES."""
    qeds, sas, logps, mws = [], [], [], []

    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        qeds.append(compute_qed(mol))
        sas.append(compute_sa_score(mol))
        logps.append(compute_logp(mol))
        mws.append(compute_molecular_weight(mol))

    if not qeds:
        return {}

    return {
        "qed_mean": float(np.mean(qeds)),
        "qed_std": float(np.std(qeds)),
        "sa_mean": float(np.mean(sas)),
        "sa_std": float(np.std(sas)),
        "logp_mean": float(np.mean(logps)),
        "logp_std": float(np.std(logps)),
        "mw_mean": float(np.mean(mws)),
        "mw_std": float(np.std(mws)),
    }


def evaluate_model(
    model,
    tokenizer: SelfiesTokenizer,
    n_samples: int = 10000,
    device: Optional[torch.device] = None,
    training_smiles: Optional[set[str]] = None,
    reward_fn: Optional[RewardFunction] = None,
    max_length: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    batch_size: int = 64,
) -> GenerationMetrics:
    """
    Comprehensive evaluation of a molecular generation model.

    Args:
        model: The generative model (GPT2LMHeadModel)
        tokenizer: SelfiesTokenizer
        n_samples: Number of molecules to generate
        device: torch device
        training_smiles: Set of training SMILES for novelty computation
        reward_fn: RewardFunction for reward computation
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling threshold
        batch_size: Generation batch size

    Returns:
        GenerationMetrics with all computed metrics
    """
    if device is None:
        device = next(model.parameters()).device

    logger.info(f"Generating {n_samples} molecules...")
    selfies_list = generate_molecules(
        model, tokenizer, n_samples, device,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        batch_size=batch_size,
    )

    logger.info("Converting to SMILES...")
    smiles_list = selfies_to_smiles_list(selfies_list)
    valid_smiles = [s for s in smiles_list if s is not None]
    unique_smiles = list(set(valid_smiles))

    metrics = GenerationMetrics(
        n_generated=len(smiles_list),
        n_valid=len(valid_smiles),
        n_unique=len(unique_smiles),
        validity=len(valid_smiles) / max(len(smiles_list), 1),
        uniqueness=len(unique_smiles) / max(len(valid_smiles), 1),
        valid_smiles=valid_smiles,
    )

    # Novelty
    if training_smiles is not None:
        metrics.novelty = novelty(valid_smiles, training_smiles)
        logger.info(f"Novelty: {metrics.novelty:.4f}")

    # Diversity
    if len(valid_smiles) >= 2:
        logger.info("Computing diversity...")
        metrics.diversity = internal_diversity(valid_smiles, sample_size=1000)

    # Property distributions
    logger.info("Computing property distributions...")
    props = compute_property_stats(valid_smiles)
    for k, v in props.items():
        setattr(metrics, k, v)

    # Rewards
    if reward_fn is not None:
        logger.info("Computing rewards...")
        rewards = reward_fn.get_rewards_tensor(selfies_list)
        metrics.reward_mean = rewards.mean().item()
        metrics.reward_std = rewards.std().item()
        metrics.reward_max = rewards.max().item()

    logger.info(f"Results: validity={metrics.validity:.4f}, "
                f"uniqueness={metrics.uniqueness:.4f}, "
                f"novelty={metrics.novelty:.4f}, "
                f"diversity={metrics.diversity:.4f}")

    return metrics
