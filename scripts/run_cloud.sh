#!/bin/bash
# Cloud training script for mol-rl
# Usage: bash scripts/run_cloud.sh [sft|rl|eval|all]
#
# Prerequisites:
#   - NVIDIA GPU with CUDA
#   - Python 3.11+
#   - uv package manager (or pip)

set -euo pipefail

STAGE="${1:-all}"
DATA_DIR="data/processed"
SFT_DIR="checkpoints/sft"
RL_DIR="checkpoints/rl"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup() {
    echo "=== Setting up environment ==="
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if [ ! -d ".venv" ]; then
        uv venv .venv --python 3.11
    fi
    source .venv/bin/activate
    uv pip install -r requirements.txt
    uv pip install -e .
    echo "Setup complete."
}

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
prepare_data() {
    echo "=== Preparing data ==="
    if [ -f "$DATA_DIR/train.csv" ]; then
        TRAIN_SIZE=$(wc -l < "$DATA_DIR/train.csv")
        if [ "$TRAIN_SIZE" -gt 1000 ]; then
            echo "Data already prepared ($TRAIN_SIZE rows). Skipping."
            return
        fi
    fi
    python -m mol_rl.data.prepare_data --output_dir "$DATA_DIR" --max_molecules 200000
    echo "Data preparation complete."
}

# ---------------------------------------------------------------------------
# SFT Training
# ---------------------------------------------------------------------------
train_sft() {
    echo "=== SFT Training ==="
    python scripts/train_sft.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$SFT_DIR" \
        --epochs 10 \
        --batch_size 64 \
        --lr 5e-4 \
        --fp16 \
        --gradient_checkpointing \
        --generate_every 2 \
        --n_generate 500
    echo "SFT training complete."
}

# ---------------------------------------------------------------------------
# RL Training
# ---------------------------------------------------------------------------
train_rl() {
    echo "=== RL Training ==="
    python scripts/train_rl.py \
        --sft_checkpoint "$SFT_DIR/best" \
        --data_dir "$DATA_DIR" \
        --output_dir "$RL_DIR" \
        --steps 2000 \
        --batch_size 64 \
        --mini_batch_size 16 \
        --lr 1e-5 \
        --kl_coef 0.2 \
        --kl_target 6.0 \
        --fp16 \
        --gradient_checkpointing \
        --eval_every 100 \
        --save_every 500 \
        --log_every 10
    echo "RL training complete."
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
evaluate() {
    echo "=== Evaluation ==="
    echo "--- SFT Baseline ---"
    python scripts/evaluate.py \
        --checkpoint "$SFT_DIR/best" \
        --data_dir "$DATA_DIR" \
        --n_samples 10000 \
        --output "$SFT_DIR/eval_results.json" \
        --save_molecules "$SFT_DIR/generated_molecules.csv"

    if [ -d "$RL_DIR/best" ]; then
        echo "--- RL Model ---"
        python scripts/evaluate.py \
            --checkpoint "$RL_DIR/best" \
            --data_dir "$DATA_DIR" \
            --n_samples 10000 \
            --output "$RL_DIR/eval_results.json" \
            --save_molecules "$RL_DIR/generated_molecules.csv"
    fi
    echo "Evaluation complete."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
setup

case "$STAGE" in
    sft)
        prepare_data
        train_sft
        ;;
    rl)
        train_rl
        ;;
    eval)
        evaluate
        ;;
    all)
        prepare_data
        train_sft
        train_rl
        evaluate
        ;;
    *)
        echo "Usage: bash scripts/run_cloud.sh [sft|rl|eval|all]"
        exit 1
        ;;
esac

echo "=== Done ==="
