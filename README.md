# mol-rl-bench

**Systematic Comparison of PPO, RLOO, and DPO for Molecular Generation**

A benchmark comparing reinforcement learning and preference optimization methods for steering molecular language models toward generating drug-like molecules with desirable properties.

## Overview

This project fine-tunes a GPT-2 language model on [SELFIES](https://github.com/aspuru-guzik-group/selfies) molecular representations, then applies three post-training methods to optimize for drug-likeness (QED), synthetic accessibility (SA), and binding affinity (AutoDock Vina):

| Method | Type | Key Advantage |
|--------|------|---------------|
| **PPO** | Online RL (policy gradient + value function) | Well-established, flexible reward shaping |
| **RLOO** | Online RL (leave-one-out baseline) | No value head → lower memory, competitive performance |
| **DPO** | Preference optimization | No reward model at train time, stable training |

## Quick Start

### 1. Setup

```bash
git clone https://github.com/<your-username>/mol-rl-bench.git
cd mol-rl-bench
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python -m mol_rl.data.prepare_data --output_dir data/processed
```

Downloads the MOSES benchmark (~1.9M drug-like molecules from ZINC), converts to SELFIES, builds vocabulary.

### 3. Train SFT Model

```bash
# Default settings (GTX 1080 / 8GB VRAM friendly)
python scripts/train_sft.py --fp16 --gradient_checkpointing

# Quick test run
python scripts/train_sft.py --debug
```

### 4. RL / DPO Optimization (Week 2-3)

```bash
# Coming soon
python scripts/train_rl.py --method ppo --config configs/ppo.yaml
python scripts/train_rl.py --method rloo --config configs/rloo.yaml
python scripts/train_rl.py --method dpo --config configs/dpo.yaml
```

## Project Structure

```
mol-rl-bench/
├── configs/              # YAML configs for each experiment
├── data/                 # Data download and preprocessing
├── src/mol_rl/
│   ├── data/             # Dataset, tokenizer, SELFIES utilities
│   ├── models/           # Reward functions
│   ├── trainers/         # SFT, PPO, RLOO, DPO trainer wrappers
│   └── eval/             # Evaluation metrics, visualization
├── scripts/              # Entry-point scripts
├── notebooks/            # Analysis and figure generation
├── tests/                # Unit tests
└── paper/                # LaTeX source for arXiv preprint
```

## Reward Functions

- **QED** (Quantitative Estimate of Drug-likeness): Continuous [0,1] score from RDKit
- **SA Score** (Synthetic Accessibility): Inverted and normalized to [0,1]
- **Docking Score** (AutoDock Vina): Binding affinity to target protein (optional)

Combined reward: `R = w₁·QED + w₂·SA + w₃·Docking`

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validity | % of generated SELFIES decoding to valid molecules |
| Uniqueness | % unique among valid generations |
| Novelty | % not present in training set |
| Mean QED | Average drug-likeness score |
| IntDiv | Internal diversity (pairwise Tanimoto distance) |
| KL Divergence | Distribution shift from SFT reference |

## GPU Requirements

| Task | VRAM | Recommended GPU |
|------|------|-----------------|
| SFT | ~8 GB | GTX 1080+ |
| DPO | ~12-16 GB | RTX 3090+ |
| RLOO (K=4) | ~16-20 GB | RTX 4090 / A100 |
| PPO | ~20-24 GB | A100 40GB |

## Citation

```
@article{mol-rl-bench-2026,
  title={PPO, RLOO, or DPO? A Systematic Comparison of Post-Training Methods for Molecular Generation},
  author={Keigo},
  year={2026}
}
```

## License

MIT
