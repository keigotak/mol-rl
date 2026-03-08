"""
PyTorch Dataset for SELFIES molecular sequences.

Provides datasets compatible with HuggingFace's SFTTrainer and standard
PyTorch DataLoaders for training autoregressive molecular language models.

Usage:
    from mol_rl.data.dataset import SelfiesDataset

    dataset = SelfiesDataset("data/processed/train.csv", tokenizer, max_length=128)
    sample = dataset[0]  # {"input_ids": tensor, "attention_mask": tensor, "labels": tensor}
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from mol_rl.data.selfies_tokenizer import SelfiesTokenizer


class SelfiesDataset(Dataset):
    """
    Dataset of tokenized SELFIES sequences for autoregressive LM training.

    Each sample returns:
        - input_ids: [BOS, tok1, tok2, ..., EOS, PAD, PAD, ...]
        - attention_mask: [1, 1, 1, ..., 1, 0, 0, ...]
        - labels: same as input_ids (shifted internally by the model)
                  with PAD positions set to -100 (ignored in loss)
    """

    def __init__(self,
                 csv_path: str,
                 tokenizer: SelfiesTokenizer,
                 max_length: int = 128,
                 selfies_column: str = "selfies"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path)
        self.selfies_list = df[selfies_column].tolist()

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        selfies_str = self.selfies_list[idx]

        # Encode with BOS and EOS
        ids = self.tokenizer.encode(
            selfies_str,
            add_special_tokens=True,
            max_length=self.max_length,
        )

        # Pad
        pad_len = self.max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.tokenizer.pad_token_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Labels: same as input_ids but with -100 for padding (ignored in CE loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SelfiesTextDataset(Dataset):
    """
    Simple text dataset for TRL's SFTTrainer.

    SFTTrainer expects a dataset with a 'text' column containing raw strings.
    The trainer handles tokenization internally. We prepend BOS and append EOS
    to each SELFIES string.
    """

    def __init__(self, csv_path: str, selfies_column: str = "selfies"):
        df = pd.read_csv(csv_path)
        self.texts = df[selfies_column].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}
