"""
Evaluate a trained molecular generation model.

Generates molecules and computes standard benchmarks: validity, uniqueness,
novelty, diversity, and property distributions.

Usage:
    # Evaluate SFT model
    python scripts/evaluate.py --checkpoint checkpoints/sft/best

    # Evaluate RL model
    python scripts/evaluate.py --checkpoint checkpoints/rl/best --n_samples 10000

    # Compare with training set (novelty)
    python scripts/evaluate.py --checkpoint checkpoints/rl/best --data_dir data/processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2LMHeadModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
from mol_rl.eval.metrics import evaluate_model
from mol_rl.models.rewards import RewardFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate molecular generation model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (for novelty computation)")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--save_molecules", type=str, default=None,
                        help="Save generated SMILES to CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    ckpt_dir = Path(args.checkpoint)
    vocab_path = ckpt_dir / "vocab.json"
    if not vocab_path.exists() and args.data_dir:
        vocab_path = Path(args.data_dir) / "vocab.json"

    tokenizer = SelfiesTokenizer.from_vocab_file(str(vocab_path), max_length=args.max_length)
    model = GPT2LMHeadModel.from_pretrained(str(ckpt_dir))
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {ckpt_dir}")
    logger.info(f"Tokenizer: {tokenizer}")

    # Training SMILES for novelty
    training_smiles = None
    if args.data_dir:
        train_csv = Path(args.data_dir) / "train.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            training_smiles = set(df["smiles"].tolist())
            logger.info(f"Loaded {len(training_smiles)} training SMILES for novelty")

    # Reward function
    reward_fn = RewardFunction(weights={"qed": 0.5, "sa": 0.5})

    # Evaluate
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        device=device,
        training_smiles=training_smiles,
        reward_fn=reward_fn,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    # Print results
    summary = metrics.summary()
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k:20s}: {v:.4f}")
        else:
            logger.info(f"  {k:20s}: {v}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    if args.save_molecules and metrics.valid_smiles:
        df_out = pd.DataFrame({"smiles": metrics.valid_smiles})
        df_out.to_csv(args.save_molecules, index=False)
        logger.info(f"Generated molecules saved to {args.save_molecules}")


if __name__ == "__main__":
    main()
