"""
Reward functions for molecular optimization.

Provides QED, SA Score, and combined reward functions that evaluate
generated SELFIES/SMILES molecules. All rewards are normalized to [0, 1]
where higher is better.

Usage:
    from mol_rl.models.rewards import RewardFunction

    reward_fn = RewardFunction(weights={"qed": 0.4, "sa": 0.3, "validity": 0.3})
    scores = reward_fn.score_selfies_batch(["[C][=C][C][Ring1][Branch1]", ...])
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.DataStructs import TanimotoSimilarity

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual Reward Components
# ---------------------------------------------------------------------------

def compute_qed(mol) -> float:
    """Quantitative Estimate of Drug-likeness. Returns value in [0, 1]."""
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0


def compute_sa_score(mol) -> float:
    """
    Synthetic Accessibility score, normalized to [0, 1] where 1 = easy to synthesize.

    Raw SA score: 1 (easy) to 10 (hard). We invert and normalize.
    """
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        # Import SA scorer from RDKit contrib
        sa_module_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_module_path not in sys.path:
            sys.path.insert(0, sa_module_path)

        import sascorer
        raw_score = sascorer.calculateScore(mol)
        # Normalize: (10 - score) / 9 maps [1, 10] -> [1.0, 0.0]
        return (10.0 - raw_score) / 9.0
    except Exception:
        # Fallback: use a simple heuristic based on ring count and heteroatom count
        try:
            n_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
            n_atoms = mol.GetNumHeavyAtoms()
            # Simple heuristic: penalize very large or very ringy molecules
            score = max(0.0, 1.0 - (n_rings / 10.0) - (max(0, n_atoms - 30) / 50.0))
            return score
        except Exception:
            return 0.0


def compute_logp(mol) -> float:
    """LogP (lipophilicity). Not normalized - used for analysis."""
    try:
        return Descriptors.MolLogP(mol)
    except Exception:
        return 0.0


def compute_molecular_weight(mol) -> float:
    """Molecular weight. Not normalized - used for analysis."""
    try:
        return Descriptors.MolWt(mol)
    except Exception:
        return 0.0


def compute_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprint for similarity calculations."""
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

@dataclass
class MoleculeScore:
    """Container for all computed scores of a molecule."""
    smiles: Optional[str] = None
    selfies: Optional[str] = None
    is_valid: bool = False
    qed: float = 0.0
    sa: float = 0.0
    logp: float = 0.0
    mw: float = 0.0
    reward: float = 0.0


class RewardFunction:
    """
    Combined reward function for molecular generation RL.

    Computes a weighted combination of molecular property scores.

    Args:
        weights: Dictionary of reward component weights.
                 Available components: "qed", "sa", "validity"
        validity_penalty: Penalty for invalid molecules (default: -1.0)
        invalid_reward: Reward for invalid molecules (overrides penalty if set)
    """

    def __init__(self,
                 weights: Optional[dict[str, float]] = None,
                 validity_penalty: float = 0.0,
                 invalid_reward: float = 0.0):

        self.weights = weights or {"qed": 0.5, "sa": 0.5}
        self.validity_penalty = validity_penalty
        self.invalid_reward = invalid_reward

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"RewardFunction initialized with weights: {self.weights}")

    def score_smiles(self, smiles: str) -> MoleculeScore:
        """Score a single SMILES string."""
        result = MoleculeScore(smiles=smiles)

        if not smiles:
            result.reward = self.invalid_reward
            return result

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result.reward = self.invalid_reward
            return result

        result.is_valid = True
        result.qed = compute_qed(mol)
        result.sa = compute_sa_score(mol)
        result.logp = compute_logp(mol)
        result.mw = compute_molecular_weight(mol)

        # Weighted reward
        reward = 0.0
        if "qed" in self.weights:
            reward += self.weights["qed"] * result.qed
        if "sa" in self.weights:
            reward += self.weights["sa"] * result.sa
        if "validity" in self.weights:
            reward += self.weights["validity"] * (1.0 if result.is_valid else 0.0)

        result.reward = reward
        return result

    def score_selfies(self, selfies_str: str) -> MoleculeScore:
        """Score a single SELFIES string by converting to SMILES first."""
        try:
            smiles = sf.decoder(selfies_str)
        except Exception:
            result = MoleculeScore(selfies=selfies_str)
            result.reward = self.invalid_reward
            return result

        result = self.score_smiles(smiles)
        result.selfies = selfies_str
        return result

    def score_selfies_batch(self, selfies_list: list[str]) -> list[MoleculeScore]:
        """Score a batch of SELFIES strings."""
        return [self.score_selfies(s) for s in selfies_list]

    def score_smiles_batch(self, smiles_list: list[str]) -> list[MoleculeScore]:
        """Score a batch of SMILES strings."""
        return [self.score_smiles(s) for s in smiles_list]

    def get_rewards_tensor(self, selfies_list: list[str]) -> "torch.Tensor":
        """Get reward values as a PyTorch tensor. Used in RL training loop."""
        import torch
        scores = self.score_selfies_batch(selfies_list)
        rewards = [s.reward for s in scores]
        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

def internal_diversity(smiles_list: list[str], sample_size: int = 1000) -> float:
    """
    Compute internal diversity as average pairwise Tanimoto distance.

    IntDiv = 1 - avg(Tanimoto(fp_i, fp_j)) for all pairs i < j.
    Higher = more diverse.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if s]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return 0.0

    # Subsample if too many
    if len(mols) > sample_size:
        indices = np.random.choice(len(mols), sample_size, replace=False)
        mols = [mols[i] for i in indices]

    fps = [compute_fingerprint(m) for m in mols]
    fps = [f for f in fps if f is not None]

    if len(fps) < 2:
        return 0.0

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(TanimotoSimilarity(fps[i], fps[j]))

    return 1.0 - np.mean(similarities)


def novelty(generated_smiles: list[str], training_smiles: set[str]) -> float:
    """Fraction of generated molecules not in the training set."""
    if not generated_smiles:
        return 0.0
    novel = sum(1 for s in generated_smiles if s not in training_smiles)
    return novel / len(generated_smiles)


def uniqueness(smiles_list: list[str]) -> float:
    """Fraction of unique molecules among valid generations."""
    valid = [s for s in smiles_list if s and Chem.MolFromSmiles(s) is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def validity_rate(smiles_list: list[str]) -> float:
    """Fraction of valid SMILES in the list."""
    if not smiles_list:
        return 0.0
    valid = sum(1 for s in smiles_list if s and Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list)
