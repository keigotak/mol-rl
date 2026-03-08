"""
Unit tests for mol-rl-bench core components.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# SELFIES Tokenizer Tests
# ---------------------------------------------------------------------------

class TestSelfiesTokenizer:
    """Test the SELFIES tokenizer."""

    @pytest.fixture
    def tokenizer(self, tmp_path):
        """Create a tokenizer with a small vocabulary."""
        import json
        from mol_rl.data.selfies_tokenizer import SelfiesTokenizer

        vocab = {
            "token2id": {
                "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                "[C]": 4, "[=C]": 5, "[N]": 6, "[=N]": 7,
                "[O]": 8, "[=O]": 9, "[Branch1]": 10, "[Ring1]": 11,
                "[#Branch1]": 12, "[S]": 13, "[F]": 14, "[Cl]": 15,
            },
            "id2token": {},
            "special_tokens": ["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        }

        vocab_path = tmp_path / "vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)

        return SelfiesTokenizer.from_vocab_file(str(vocab_path))

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 16
        assert len(tokenizer) == 16

    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_encode_decode_roundtrip(self, tokenizer):
        selfies = "[C][=C][C]"
        encoded = tokenizer.encode(selfies, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == selfies

    def test_encode_with_special_tokens(self, tokenizer):
        selfies = "[C][N]"
        encoded = tokenizer.encode(selfies, add_special_tokens=True)
        assert encoded[0] == tokenizer.bos_token_id
        assert encoded[-1] == tokenizer.eos_token_id

    def test_batch_encode(self, tokenizer):
        selfies_list = ["[C][=C]", "[C][N][O]"]
        result = tokenizer.batch_encode(selfies_list, return_tensors="pt")
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2

    def test_unknown_token(self, tokenizer):
        # Token not in vocab should map to UNK
        encoded = tokenizer.encode("[Br]", add_special_tokens=False)
        assert tokenizer.unk_token_id in encoded

    def test_return_tensor(self, tokenizer):
        result = tokenizer.encode("[C]", return_tensors="pt")
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Data Preparation Tests
# ---------------------------------------------------------------------------

class TestDataPreparation:
    """Test molecular data processing functions."""

    def test_smiles_validation(self):
        from mol_rl.data.prepare_data import is_valid_smiles
        assert is_valid_smiles("CCO") is True
        assert is_valid_smiles("c1ccccc1") is True
        assert is_valid_smiles("invalid!!!") is False
        assert is_valid_smiles("") is False
        assert is_valid_smiles(None) is False

    def test_canonicalize(self):
        from mol_rl.data.prepare_data import canonicalize_smiles
        # Different representations of ethanol
        assert canonicalize_smiles("OCC") == canonicalize_smiles("CCO")
        assert canonicalize_smiles("invalid") is None

    def test_smiles_to_selfies(self):
        from mol_rl.data.prepare_data import smiles_to_selfies, selfies_to_smiles
        smi = "CCO"
        sel = smiles_to_selfies(smi)
        assert sel is not None
        # Roundtrip
        smi_back = selfies_to_smiles(sel)
        assert smi_back is not None

    def test_drug_filter(self):
        from mol_rl.data.prepare_data import passes_drug_filter
        assert passes_drug_filter("CCCCC") is True  # Pentane - 5 heavy atoms, passes filter
        assert passes_drug_filter("C") is False      # Methane - too few atoms (1 < min_atoms=5)
        assert passes_drug_filter("CCO") is False    # Ethanol - too few atoms (3 < min_atoms=5)

    def test_selfies_tokenization(self):
        from mol_rl.data.prepare_data import get_selfies_tokens, smiles_to_selfies
        sel = smiles_to_selfies("c1ccccc1")  # Benzene
        tokens = get_selfies_tokens(sel)
        assert len(tokens) > 0
        assert all(t.startswith("[") and t.endswith("]") for t in tokens)


# ---------------------------------------------------------------------------
# Reward Function Tests
# ---------------------------------------------------------------------------

class TestRewardFunctions:
    """Test molecular reward functions."""

    def test_qed(self):
        from mol_rl.models.rewards import compute_qed
        from rdkit import Chem
        # Aspirin
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        qed = compute_qed(mol)
        assert 0.0 <= qed <= 1.0

    def test_reward_function_batch(self):
        from mol_rl.models.rewards import RewardFunction
        rf = RewardFunction(weights={"qed": 0.5, "sa": 0.5})

        selfies_list = ["[C][=C][C][=C][C][=C][Ring1][=Branch1]"]  # Benzene
        scores = rf.score_selfies_batch(selfies_list)
        assert len(scores) == 1
        assert scores[0].is_valid
        assert scores[0].reward > 0

    def test_invalid_molecule_reward(self):
        from mol_rl.models.rewards import RewardFunction
        rf = RewardFunction(invalid_reward=-1.0)
        score = rf.score_smiles("not_a_molecule")
        assert score.is_valid is False
        assert score.reward == -1.0

    def test_validity_rate(self):
        from mol_rl.models.rewards import validity_rate
        smiles = ["CCO", "c1ccccc1", "invalid", None]
        rate = validity_rate(smiles)
        assert rate == 0.5  # 2 out of 4

    def test_uniqueness(self):
        from mol_rl.models.rewards import uniqueness
        smiles = ["CCO", "CCO", "c1ccccc1"]
        u = uniqueness(smiles)
        assert abs(u - 2.0 / 3.0) < 0.01


# ---------------------------------------------------------------------------
# Dataset Tests
# ---------------------------------------------------------------------------

class TestDataset:
    """Test PyTorch dataset."""

    @pytest.fixture
    def setup_data(self, tmp_path):
        import json
        import pandas as pd
        from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
        from mol_rl.data.prepare_data import smiles_to_selfies

        # Create a small CSV
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        selfies_list = [smiles_to_selfies(s) for s in smiles_list]

        df = pd.DataFrame({"smiles": smiles_list, "selfies": selfies_list})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        # Create vocab from these molecules
        from mol_rl.data.prepare_data import get_selfies_tokens
        all_tokens = set()
        for sel in selfies_list:
            all_tokens.update(get_selfies_tokens(sel))

        special = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        token2id = {t: i for i, t in enumerate(special)}
        for t in sorted(all_tokens):
            if t not in token2id:
                token2id[t] = len(token2id)

        vocab = {"token2id": token2id, "id2token": {}, "special_tokens": special}
        vocab_path = tmp_path / "vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)

        tokenizer = SelfiesTokenizer.from_vocab_file(str(vocab_path), max_length=64)
        return csv_path, tokenizer

    def test_dataset_length(self, setup_data):
        from mol_rl.data.dataset import SelfiesDataset
        csv_path, tokenizer = setup_data
        ds = SelfiesDataset(str(csv_path), tokenizer, max_length=64)
        assert len(ds) == 3

    def test_dataset_item_shape(self, setup_data):
        from mol_rl.data.dataset import SelfiesDataset
        csv_path, tokenizer = setup_data
        ds = SelfiesDataset(str(csv_path), tokenizer, max_length=64)
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["labels"].shape == (64,)

    def test_dataset_labels_padding(self, setup_data):
        from mol_rl.data.dataset import SelfiesDataset
        csv_path, tokenizer = setup_data
        ds = SelfiesDataset(str(csv_path), tokenizer, max_length=64)
        item = ds[0]
        # Labels should have -100 where attention_mask is 0
        padding_mask = item["attention_mask"] == 0
        assert (item["labels"][padding_mask] == -100).all()


# ---------------------------------------------------------------------------
# REINFORCE Trainer Tests
# ---------------------------------------------------------------------------

class TestReinforceTrainer:
    """Test the REINFORCE/RLOO trainer."""

    @pytest.fixture
    def setup_trainer(self, tmp_path):
        """Create a small trainer for testing."""
        import json
        from transformers import GPT2Config, GPT2LMHeadModel
        from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
        from mol_rl.models.rewards import RewardFunction
        from mol_rl.trainers.reinforce import ReinforceConfig, ReinforceTrainer

        # Small vocab
        vocab = {
            "token2id": {
                "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                "[C]": 4, "[=C]": 5, "[N]": 6, "[O]": 7,
                "[Branch1]": 8, "[Ring1]": 9, "[=Branch1]": 10,
                "[=N]": 11, "[S]": 12, "[F]": 13,
            },
        }
        vocab_path = tmp_path / "vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        tokenizer = SelfiesTokenizer.from_vocab_file(str(vocab_path), max_length=32)

        # Tiny GPT-2
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=64,
            n_head=2,
            n_layer=2,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        policy = GPT2LMHeadModel(config)
        ref_model = GPT2LMHeadModel(config)
        ref_model.load_state_dict(policy.state_dict())

        device = torch.device("cpu")
        policy.to(device)
        ref_model.to(device)

        reward_fn = RewardFunction(weights={"qed": 0.5, "sa": 0.5})
        rl_config = ReinforceConfig(
            batch_size=8,
            mini_batch_size=4,
            rloo_k=4,
            kl_coef=0.05,
            max_length=32,
            fp16=False,
        )

        trainer = ReinforceTrainer(
            policy=policy,
            ref_model=ref_model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=rl_config,
            device=device,
        )

        return trainer

    def test_generate(self, setup_trainer):
        trainer = setup_trainer
        gen = trainer.generate(batch_size=4)
        assert gen["sequences"].shape[0] == 4
        assert gen["log_probs"].shape[0] == 4
        assert gen["attention_mask"].shape[0] == 4
        # Sequences should start with BOS
        assert (gen["sequences"][:, 0] == trainer.tokenizer.bos_token_id).all()

    def test_rloo_advantages(self, setup_trainer):
        trainer = setup_trainer
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        advantages = trainer.compute_rloo_advantages(rewards, k=4)
        assert advantages.shape == (8,)
        # Within each group of 4, advantages should sum to ~0
        group1_sum = advantages[:4].sum().item()
        group2_sum = advantages[4:].sum().item()
        assert abs(group1_sum) < 1e-5
        assert abs(group2_sum) < 1e-5

    def test_step(self, setup_trainer):
        trainer = setup_trainer
        optimizer = torch.optim.AdamW(trainer.policy.parameters(), lr=1e-4)
        metrics = trainer.step(optimizer)
        assert "reward_mean" in metrics
        assert "loss" in metrics
        assert "kl" in metrics
        assert "validity" in metrics
        assert 0.0 <= metrics["validity"] <= 1.0

    def test_ref_model_frozen(self, setup_trainer):
        trainer = setup_trainer
        for p in trainer.ref_model.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Evaluation Module Tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    """Test the evaluation module."""

    @pytest.fixture
    def setup_model(self, tmp_path):
        """Create a tiny model and tokenizer for evaluation tests."""
        import json
        from transformers import GPT2Config, GPT2LMHeadModel
        from mol_rl.data.selfies_tokenizer import SelfiesTokenizer

        vocab = {
            "token2id": {
                "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                "[C]": 4, "[=C]": 5, "[N]": 6, "[O]": 7,
                "[Branch1]": 8, "[Ring1]": 9, "[=Branch1]": 10,
                "[=N]": 11, "[S]": 12, "[F]": 13,
            },
        }
        vocab_path = tmp_path / "vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)
        tokenizer = SelfiesTokenizer.from_vocab_file(str(vocab_path), max_length=32)

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=64, n_head=2, n_layer=2, n_positions=64,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        model = GPT2LMHeadModel(config)
        device = torch.device("cpu")
        model.to(device)

        return model, tokenizer, device

    def test_generate_molecules(self, setup_model):
        from mol_rl.eval.metrics import generate_molecules
        model, tokenizer, device = setup_model
        selfies_list = generate_molecules(
            model, tokenizer, n_samples=8, device=device,
            max_length=32, batch_size=4,
        )
        assert len(selfies_list) == 8
        assert all(isinstance(s, str) for s in selfies_list)

    def test_evaluate_model(self, setup_model):
        from mol_rl.eval.metrics import evaluate_model, GenerationMetrics
        from mol_rl.models.rewards import RewardFunction
        model, tokenizer, device = setup_model

        reward_fn = RewardFunction(weights={"qed": 0.5, "sa": 0.5})
        metrics = evaluate_model(
            model, tokenizer, n_samples=8, device=device,
            reward_fn=reward_fn, max_length=32, batch_size=4,
        )
        assert isinstance(metrics, GenerationMetrics)
        assert metrics.n_generated == 8
        assert 0.0 <= metrics.validity <= 1.0
        assert 0.0 <= metrics.uniqueness <= 1.0

    def test_metrics_summary(self, setup_model):
        from mol_rl.eval.metrics import evaluate_model
        model, tokenizer, device = setup_model
        metrics = evaluate_model(
            model, tokenizer, n_samples=4, device=device,
            max_length=32, batch_size=4,
        )
        summary = metrics.summary()
        assert isinstance(summary, dict)
        assert "validity" in summary
        assert "valid_smiles" not in summary  # lists excluded
