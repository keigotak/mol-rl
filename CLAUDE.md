# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Install uv (Windows PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Create venv with Python 3.11 and install dependencies
uv python install 3.11
uv venv .venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e .

# Activate venv
.venv\Scripts\activate        # Windows cmd/PowerShell
source .venv/Scripts/activate # Git Bash
```

## Commands

```bash
# Install dependencies (after activating venv)
uv pip install -r requirements.txt
uv pip install -e .

# Prepare data (downloads MOSES benchmark ~1.9M molecules from GitHub)
python -m mol_rl.data.prepare_data --output_dir data/processed
python -m mol_rl.data.prepare_data --source file --input_file path/to/smiles.csv --output_dir data/processed

# SFT training (GTX 1080 / 8GB VRAM optimized)
python scripts/train_sft.py --fp16 --gradient_checkpointing
python scripts/train_sft.py --debug   # Quick smoke test with small dataset

# RL training (requires SFT checkpoint)
python scripts/train_rl.py --fp16 --gradient_checkpointing
python scripts/train_rl.py --debug   # Quick smoke test

# Evaluate a trained model
python scripts/evaluate.py --checkpoint checkpoints/sft/best --data_dir data/processed --n_samples 10000
python scripts/evaluate.py --checkpoint checkpoints/rl/best --data_dir data/processed

# Run all tests
pytest tests/ -v

# Run a single test class or method
pytest tests/test_core.py::TestSelfiesTokenizer -v
pytest tests/test_core.py::TestRewardFunctions::test_reward_function_batch -v
pytest tests/test_core.py::TestReinforceTrainer -v
```

## Architecture

### Molecular representation pipeline

All molecules flow as: **SMILES → canonical SMILES → SELFIES → token IDs**. SELFIES is used instead of SMILES because it guarantees syntactic validity — any token sequence decodes to a chemically valid molecule, which is critical for generative RL.

The vocabulary (`data/processed/vocab.json`) is built from the training corpus and contains only SELFIES tokens that appear in the data, plus four special tokens: `[PAD]=0`, `[BOS]=1`, `[EOS]=2`, `[UNK]=3`.

### Token flow through training

1. `prepare_data.py` produces CSVs with both `smiles` and `selfies` columns, plus `selfies_tokens` (space-separated) and a `vocab.json`.
2. `SelfiesTokenizer` (in `data/selfies_tokenizer.py`) loads `vocab.json` and wraps encoding/decoding. It is HuggingFace-compatible (`__call__`, `batch_encode`), but is a **custom class** — not a `PreTrainedTokenizer`. GPT-2's original tokenizer is **not used**; the model's embedding layer is resized to `vocab_size` from `vocab.json`.
3. `SelfiesDataset` (`data/dataset.py`) reads the CSVs, encodes each molecule, and sets labels to `input_ids` with padding positions replaced by `-100` (standard HuggingFace causal LM convention).
4. `train_sft.py` trains GPT-2 **from random initialization** (`from_scratch: true` in `configs/sft.yaml`) because English pretraining is irrelevant to SELFIES.

### Reward functions

`RewardFunction` in `models/rewards.py` is the central interface for RL. Key design choices:
- `score_selfies()` converts SELFIES → SMILES via `sf.decoder()` before calling RDKit — invalid SELFIES return `invalid_reward` (default 0.0).
- SA score uses RDKit's contrib `sascorer` with a fallback heuristic if the contrib path is unavailable.
- `get_rewards_tensor()` is the method intended for the RL training loop (returns a `torch.Tensor`).
- Weights are **auto-normalized** at construction time — no need to ensure they sum to 1.

### Configuration

Training is configured via argparse in scripts (not Hydra), with `configs/sft.yaml` serving as documentation of recommended hyperparameters. W&B logging is opt-in by setting `wandb_project` in the config or passing `--wandb_project`.

### RL training pipeline

`ReinforceTrainer` in `trainers/reinforce.py` implements REINFORCE with RLOO (Leave-One-Out) baseline:
- Generates molecules from the policy, computes rewards via `RewardFunction`, updates via policy gradient.
- KL penalty against a frozen SFT reference model prevents mode collapse.
- RLOO baseline: within groups of `k` samples, each sample's baseline is the mean reward of the other `k-1` samples.
- Supports gradient accumulation via `mini_batch_size` and mixed precision (`fp16`).
- `scripts/train_rl.py` is the entry point; `configs/rl.yaml` documents recommended hyperparameters.

### Evaluation

`eval/metrics.py` provides `evaluate_model()` which generates molecules and computes: validity, uniqueness, novelty, diversity, property distributions (QED, SA, LogP, MW), and rewards. `scripts/evaluate.py` is the CLI entry point.

### Evaluation

`eval/metrics.py` provides `evaluate_model()` which generates molecules and computes: validity, uniqueness, novelty, diversity, property distributions (QED, SA, LogP, MW), and rewards. `scripts/evaluate.py` is the CLI entry point.

### What is not yet implemented

DPO training is not yet implemented. The docking reward (`meeko`, `vina`) is commented out in `requirements.txt`.
