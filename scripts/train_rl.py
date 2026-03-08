"""
RL training of molecular generator using REINFORCE with RLOO baseline.

Fine-tunes an SFT-pretrained GPT-2 model to optimize molecular properties
(QED, SA score) via policy gradient with KL penalty against the SFT reference.

Usage:
    # Default settings (recommended for GTX 1080 8GB)
    python scripts/train_rl.py

    # Custom settings
    python scripts/train_rl.py \
        --sft_checkpoint checkpoints/sft/best \
        --steps 2000 \
        --batch_size 64 \
        --kl_coef 0.05 \
        --fp16

    # Quick test run
    python scripts/train_rl.py --debug
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, get_cosine_schedule_with_warmup

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
from mol_rl.models.rewards import (
    RewardFunction,
    internal_diversity,
    novelty,
    uniqueness,
    validity_rate,
)
from mol_rl.trainers.reinforce import ReinforceConfig, ReinforceTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_policy(trainer, n_samples=500):
    """Generate molecules and compute comprehensive metrics."""
    import selfies as sf
    from rdkit import Chem

    cfg = trainer.config
    all_selfies = []
    all_smiles = []
    all_rewards = []

    remaining = n_samples
    while remaining > 0:
        bs = min(remaining, cfg.batch_size)
        gen = trainer.generate(bs)
        selfies_list = trainer.decode_sequences(gen["sequences"])
        rewards = trainer.reward_fn.get_rewards_tensor(selfies_list)

        for sel, r in zip(selfies_list, rewards):
            all_selfies.append(sel)
            all_rewards.append(r.item())
            try:
                smi = sf.decoder(sel)
                all_smiles.append(smi)
            except Exception:
                all_smiles.append(None)

        remaining -= bs

    # Compute metrics
    valid_smiles = [s for s in all_smiles if s and Chem.MolFromSmiles(s) is not None]

    metrics = {
        "n_generated": len(all_smiles),
        "reward_mean": np.mean(all_rewards),
        "reward_std": np.std(all_rewards),
        "reward_max": max(all_rewards),
        "validity": len(valid_smiles) / max(len(all_smiles), 1),
        "uniqueness": uniqueness(valid_smiles) if valid_smiles else 0.0,
        "n_unique": len(set(valid_smiles)),
        "diversity": internal_diversity(valid_smiles, sample_size=500) if len(valid_smiles) >= 2 else 0.0,
    }

    # Log examples
    scores = trainer.reward_fn.score_selfies_batch(all_selfies[:10])
    logger.info("Sample generated molecules:")
    for i, (sel, smi, sc) in enumerate(zip(all_selfies[:5], all_smiles[:5], scores[:5])):
        logger.info(f"  [{i}] SMILES: {smi}")
        logger.info(f"       QED={sc.qed:.3f} SA={sc.sa:.3f} reward={sc.reward:.3f}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL on molecular generation")
    parser.add_argument("--sft_checkpoint", type=str, default="checkpoints/sft/best")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rl")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--rloo_k", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.2)
    parser.add_argument("--kl_target", type=float, default=6.0,
                        help="Adaptive KL target (set to 0 to disable)")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--reward_qed_weight", type=float, default=0.5)
    parser.add_argument("--reward_sa_weight", type=float, default=0.5)
    parser.add_argument("--invalid_reward", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--n_eval_samples", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Debug mode
    if args.debug:
        args.steps = 20
        args.batch_size = 8
        args.mini_batch_size = 4
        args.eval_every = 5
        args.n_eval_samples = 16
        args.log_every = 1
        args.save_every = 10
        logger.info("DEBUG MODE: reduced settings")

    # Load tokenizer
    data_dir = Path(args.data_dir)
    sft_dir = Path(args.sft_checkpoint)

    # Prefer vocab from SFT checkpoint, fall back to data dir
    vocab_path = sft_dir / "vocab.json"
    if not vocab_path.exists():
        vocab_path = data_dir / "vocab.json"
    tokenizer = SelfiesTokenizer.from_vocab_file(str(vocab_path), max_length=args.max_length)
    logger.info(f"Tokenizer: {tokenizer}")

    # Load policy model (from SFT checkpoint)
    logger.info(f"Loading SFT checkpoint from {sft_dir}...")
    policy = GPT2LMHeadModel.from_pretrained(str(sft_dir))
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.to(device)
    logger.info(f"Policy model: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params")

    # Load reference model (frozen copy of SFT)
    ref_model = GPT2LMHeadModel.from_pretrained(str(sft_dir))
    ref_model.to(device)
    ref_model.eval()
    logger.info("Reference model loaded (frozen)")

    # Reward function
    reward_fn = RewardFunction(
        weights={"qed": args.reward_qed_weight, "sa": args.reward_sa_weight},
        invalid_reward=args.invalid_reward,
    )

    # Trainer config
    rl_config = ReinforceConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        rloo_k=args.rloo_k,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target if args.kl_target > 0 else None,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
    )

    trainer = ReinforceTrainer(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=rl_config,
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    warmup_steps = int(args.steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.steps,
    )

    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    # W&B
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------------------------------------------------------------------
    # Initial evaluation
    # ---------------------------------------------------------------------------
    logger.info("\n--- Initial evaluation (SFT baseline) ---")
    init_metrics = evaluate_policy(trainer, n_samples=args.n_eval_samples)
    logger.info(f"Initial metrics: {init_metrics}")

    # ---------------------------------------------------------------------------
    # RL Training Loop
    # ---------------------------------------------------------------------------
    logger.info(f"\nStarting RL training for {args.steps} steps...")
    logger.info(f"  batch_size={args.batch_size}, lr={args.lr}, kl_coef={args.kl_coef}")

    best_reward = init_metrics["reward_mean"]
    history = []

    for step in range(1, args.steps + 1):
        metrics = trainer.step(optimizer, scaler=scaler)
        scheduler.step()
        metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(metrics)

        # Log
        if step % args.log_every == 0:
            logger.info(
                f"Step {step}/{args.steps} | "
                f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.3f} "
                f"(max={metrics['reward_max']:.3f}) | "
                f"validity={metrics['validity']:.3f} | "
                f"kl={metrics['kl']:.4f} (β={metrics['kl_coef']:.3f}) | "
                f"loss={metrics['loss']:.4f} | "
                f"lr={metrics['lr']:.2e}"
            )

        if args.wandb_project:
            import wandb
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

        # Evaluate
        if step % args.eval_every == 0:
            logger.info(f"\n--- Evaluation at step {step} ---")
            eval_metrics = evaluate_policy(trainer, n_samples=args.n_eval_samples)
            logger.info(f"Eval metrics: {eval_metrics}")

            if args.wandb_project:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

            # Save best
            if eval_metrics["reward_mean"] > best_reward:
                best_reward = eval_metrics["reward_mean"]
                policy.save_pretrained(output_dir / "best")
                tokenizer.save(str(output_dir / "best" / "vocab.json"))
                logger.info(f"New best model saved (reward={best_reward:.4f})")

        # Periodic save
        if step % args.save_every == 0:
            ckpt_dir = output_dir / f"step_{step}"
            policy.save_pretrained(ckpt_dir)
            tokenizer.save(str(ckpt_dir / "vocab.json"))
            logger.info(f"Checkpoint saved to {ckpt_dir}")

    # ---------------------------------------------------------------------------
    # Final save
    # ---------------------------------------------------------------------------
    policy.save_pretrained(output_dir / "final")
    tokenizer.save(str(output_dir / "final" / "vocab.json"))

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nRL training complete.")
    logger.info(f"  Best reward: {best_reward:.4f}")
    logger.info(f"  Models saved to {output_dir}/")

    # Final evaluation
    logger.info("\n--- Final evaluation ---")
    final_metrics = evaluate_policy(trainer, n_samples=args.n_eval_samples)
    logger.info(f"Final metrics: {final_metrics}")

    improvement = final_metrics["reward_mean"] - init_metrics["reward_mean"]
    logger.info(f"Reward improvement: {init_metrics['reward_mean']:.4f} → {final_metrics['reward_mean']:.4f} ({improvement:+.4f})")

    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
