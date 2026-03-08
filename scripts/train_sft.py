"""
Supervised Fine-Tuning (SFT) of GPT-2 on SELFIES molecular data.

Trains a GPT-2 model to generate valid SELFIES sequences representing
drug-like molecules. This SFT checkpoint serves as the starting point
for subsequent RL/DPO optimization.

Usage:
    # Default settings (recommended for GTX 1080 8GB)
    python scripts/train_sft.py

    # Custom settings
    python scripts/train_sft.py \
        --data_dir data/processed \
        --output_dir checkpoints/sft \
        --model_name gpt2 \
        --epochs 10 \
        --batch_size 32 \
        --max_length 128 \
        --lr 5e-4 \
        --fp16

    # Quick test run
    python scripts/train_sft.py --debug
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
from mol_rl.data.dataset import SelfiesDataset
from mol_rl.models.rewards import validity_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, use_fp16=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_fp16:
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | LR: {lr:.2e}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, use_fp16=False):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_fp16:
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += outputs.loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def generate_and_evaluate(model, tokenizer, device, n_samples=500,
                          max_length=128, temperature=1.0, top_k=0, top_p=0.95):
    """Generate molecules and compute validity metrics."""
    import selfies as sf

    model.eval()
    generated_selfies = []
    generated_smiles = []

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    batch_size = 64
    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        current_bs = min(batch_size, n_samples - len(generated_selfies))
        if current_bs <= 0:
            break

        # Start with BOS token
        input_ids = torch.full((current_bs, 1), bos_id, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]  # (batch, vocab)

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check if all sequences have generated EOS
            if (next_token == eos_id).all():
                break

        # Decode
        for seq in input_ids:
            selfies_str = tokenizer.decode(seq, skip_special_tokens=True)
            generated_selfies.append(selfies_str)
            try:
                smi = sf.decoder(selfies_str)
                generated_smiles.append(smi)
            except Exception:
                generated_smiles.append(None)

    # Compute metrics
    valid_smiles = [s for s in generated_smiles if s is not None]
    from rdkit import Chem
    valid_mols = [s for s in valid_smiles if Chem.MolFromSmiles(s) is not None]

    n_total = len(generated_smiles)
    n_valid = len(valid_mols)
    n_unique = len(set(valid_mols))

    metrics = {
        "n_generated": n_total,
        "validity": n_valid / max(n_total, 1),
        "uniqueness": n_unique / max(n_valid, 1),
        "n_unique": n_unique,
    }

    # Show some examples
    logger.info("Sample generated molecules:")
    for i, (sel, smi) in enumerate(zip(generated_selfies[:5], generated_smiles[:5])):
        logger.info(f"  [{i}] SELFIES: {sel[:80]}...")
        logger.info(f"       SMILES:  {smi}")

    return metrics


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def create_model(vocab_size: int,
                 model_name: str = "gpt2",
                 from_scratch: bool = True,
                 gradient_checkpointing: bool = False) -> GPT2LMHeadModel:
    """
    Create GPT-2 model for SELFIES generation.

    Args:
        vocab_size: Size of the SELFIES vocabulary
        model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
        from_scratch: If True, initialize randomly. If False, load pretrained.
        gradient_checkpointing: Enable gradient checkpointing to save memory
    """
    if from_scratch:
        # Use GPT-2 architecture but with custom vocab and random init
        config = GPT2Config.from_pretrained(model_name)
        config.vocab_size = vocab_size
        config.bos_token_id = 1  # [BOS]
        config.eos_token_id = 2  # [EOS]
        config.pad_token_id = 0  # [PAD]
        model = GPT2LMHeadModel(config)
        logger.info(f"Created GPT-2 from scratch: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # Resize embeddings for SELFIES vocab
        model.resize_token_embeddings(vocab_size)
        logger.info(f"Loaded pretrained {model_name}, resized vocab to {vocab_size}")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SFT model on SELFIES data")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--from_scratch", action="store_true", default=True,
                        help="Train from random init (default: True for molecular data)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (reduce to 16 for 8GB GPU)")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (recommended for GTX 1080)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save VRAM")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--generate_every", type=int, default=2,
                        help="Generate and evaluate molecules every N epochs")
    parser.add_argument("--n_generate", type=int, default=500,
                        help="Number of molecules to generate for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                        help="Quick debug run with small data")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (disabled if not set)")
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

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer = SelfiesTokenizer.from_vocab_file(str(data_dir / "vocab.json"),
                                                  max_length=args.max_length)
    logger.info(f"Tokenizer: {tokenizer}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SelfiesDataset(str(data_dir / "train.csv"), tokenizer,
                                    max_length=args.max_length)
    val_dataset = SelfiesDataset(str(data_dir / "val.csv"), tokenizer,
                                  max_length=args.max_length)

    if args.debug:
        # Use tiny subset for debugging
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(500, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))
        args.epochs = 2
        args.generate_every = 1
        args.n_generate = 50
        logger.info("DEBUG MODE: Using small subset")

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        model_name=args.model_name,
        from_scratch=args.from_scratch,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total training steps: {total_steps}, warmup: {warmup_steps}")

    # W&B
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, use_fp16=args.fp16,
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        if epoch % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, use_fp16=args.fp16)
            logger.info(f"Val loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(output_dir / "best")
                tokenizer.save(str(output_dir / "best" / "vocab.json"))
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            if args.wandb_project:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": scheduler.get_last_lr()[0],
                })

        # Generate molecules for evaluation
        if epoch % args.generate_every == 0:
            logger.info(f"\nGenerating {args.n_generate} molecules...")
            gen_metrics = generate_and_evaluate(
                model, tokenizer, device,
                n_samples=args.n_generate,
                max_length=args.max_length,
            )
            logger.info(f"Generation metrics: {gen_metrics}")

            if args.wandb_project:
                import wandb
                wandb.log({f"gen/{k}": v for k, v in gen_metrics.items()})

    # Save final model
    model.save_pretrained(output_dir / "final")
    tokenizer.save(str(output_dir / "final" / "vocab.json"))
    logger.info(f"\nTraining complete. Models saved to {output_dir}/")
    logger.info(f"Best val loss: {best_val_loss:.4f}")

    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
