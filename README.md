# mol-rl

**Molecular Generation with Reinforcement Learning**

Fine-tunes a GPT-2 language model on [SELFIES](https://github.com/aspuru-guzik-group/selfies) molecular representations, then optimizes for drug-likeness (QED) and synthetic accessibility (SA) via REINFORCE with RLOO baseline.

## Quick Start

```bash
git clone https://github.com/keigotak/mol-rl.git
cd mol-rl
pip install -r requirements.txt
pip install -e .
```

Or run everything on cloud:

```bash
bash scripts/run_cloud.sh all
```

### Step by Step

```bash
# 1. Prepare data (downloads MOSES benchmark, ~200K molecules)
python -m mol_rl.data.prepare_data --output_dir data/processed --max_molecules 200000

# 2. SFT training (GTX 1080 / 8GB VRAM friendly)
python scripts/train_sft.py --fp16 --gradient_checkpointing

# 3. RL optimization (REINFORCE + RLOO baseline)
python scripts/train_rl.py --fp16 --gradient_checkpointing

# 4. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/rl/best --data_dir data/processed
```

## Project Structure

```
mol-rl/
├── configs/              # YAML configs (sft.yaml, rl.yaml)
├── data/                 # Data download and preprocessing
├── src/mol_rl/
│   ├── data/             # Dataset, tokenizer, SELFIES utilities
│   ├── models/           # Reward functions (QED, SA)
│   ├── trainers/         # REINFORCE/RLOO trainer
│   └── eval/             # Evaluation metrics
├── scripts/              # Entry-point scripts
└── tests/                # Unit tests
```

## Reward Functions

- **QED** (Quantitative Estimate of Drug-likeness): Continuous [0,1] score from RDKit
- **SA Score** (Synthetic Accessibility): Inverted and normalized to [0,1]

Combined reward: `R = w_qed * QED + w_sa * SA`

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validity | % of generated SELFIES decoding to valid molecules |
| Uniqueness | % unique among valid generations |
| Novelty | % not present in training set |
| Diversity | Internal diversity (pairwise Tanimoto distance) |
| Mean QED | Average drug-likeness score |

## Tests

```bash
pytest tests/ -v
```

## License

MIT
