"""
SELFIES Tokenizer for molecular generation.

A character-level tokenizer based on SELFIES tokens, compatible with
HuggingFace's Transformers library. Each SELFIES token (e.g., [C], [=N],
[Branch1]) becomes a single token in the vocabulary.

Usage:
    from mol_rl.data.selfies_tokenizer import SelfiesTokenizer

    tokenizer = SelfiesTokenizer.from_vocab_file("data/processed/vocab.json")
    encoded = tokenizer.encode("[C][=C][C][Ring1][Branch1]")
    decoded = tokenizer.decode(encoded)
"""

import json
from pathlib import Path
from typing import Optional, Union

import selfies as sf
import torch


class SelfiesTokenizer:
    """
    A tokenizer for SELFIES molecular strings.

    Splits SELFIES into tokens using selfies.split_selfies(), then maps to
    integer IDs. Compatible with HuggingFace training pipelines via
    __call__, encode, decode, and batch_encode methods.
    """

    def __init__(self, token2id: dict[str, int], max_length: int = 128):
        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}
        self.max_length = max_length

        # Special token IDs
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"

        self.pad_token_id = token2id[self.pad_token]
        self.bos_token_id = token2id[self.bos_token]
        self.eos_token_id = token2id[self.eos_token]
        self.unk_token_id = token2id[self.unk_token]

        self.vocab_size = len(token2id)

    @classmethod
    def from_vocab_file(cls, vocab_path: str, max_length: int = 128) -> "SelfiesTokenizer":
        """Load tokenizer from a vocabulary JSON file."""
        with open(vocab_path) as f:
            vocab = json.load(f)
        return cls(token2id=vocab["token2id"], max_length=max_length)

    def save(self, path: str):
        """Save tokenizer vocabulary to JSON."""
        data = {
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
            "special_tokens": [self.pad_token, self.bos_token, self.eos_token, self.unk_token],
            "max_length": self.max_length,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def tokenize(self, selfies_str: str) -> list[str]:
        """Split a SELFIES string into tokens."""
        try:
            return list(sf.split_selfies(selfies_str))
        except Exception:
            return []

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert token strings to integer IDs."""
        return [self.token2id.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert integer IDs back to token strings."""
        return [self.id2token.get(i, self.unk_token) for i in ids]

    def encode(self,
               selfies_str: str,
               add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               return_tensors: Optional[str] = None) -> Union[list[int], torch.Tensor]:
        """
        Encode a SELFIES string into token IDs.

        Args:
            selfies_str: Input SELFIES string
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Override default max_length
            return_tensors: "pt" for PyTorch tensor, None for list

        Returns:
            List of token IDs or PyTorch tensor
        """
        max_len = max_length or self.max_length
        tokens = self.tokenize(selfies_str)
        ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        # Truncate
        ids = ids[:max_len]

        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long)

        return ids

    def decode(self,
               ids: Union[list[int], torch.Tensor],
               skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to a SELFIES string.

        Args:
            ids: Token IDs (list or tensor)
            skip_special_tokens: Whether to remove special tokens

        Returns:
            SELFIES string
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []

        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            # Stop at EOS
            if i == self.eos_token_id:
                break
            token = self.id2token.get(i, self.unk_token)
            if token != self.unk_token:
                tokens.append(token)

        return "".join(tokens)

    def decode_to_smiles(self,
                         ids: Union[list[int], torch.Tensor],
                         skip_special_tokens: bool = True) -> Optional[str]:
        """Decode token IDs to SMILES via SELFIES intermediate."""
        selfies_str = self.decode(ids, skip_special_tokens=skip_special_tokens)
        if not selfies_str:
            return None
        try:
            return sf.decoder(selfies_str)
        except Exception:
            return None

    def batch_encode(self,
                     selfies_list: list[str],
                     max_length: Optional[int] = None,
                     padding: bool = True,
                     return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        """
        Batch encode multiple SELFIES strings with padding.

        Args:
            selfies_list: List of SELFIES strings
            max_length: Override default max_length
            padding: Whether to pad to max_length
            return_tensors: "pt" for PyTorch tensors

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        max_len = max_length or self.max_length

        all_ids = []
        for sel in selfies_list:
            ids = self.encode(sel, add_special_tokens=True, max_length=max_len)
            all_ids.append(ids)

        if padding:
            # Pad to the max length in this batch (or max_len)
            batch_max = min(max(len(ids) for ids in all_ids), max_len)
            attention_masks = []

            for i in range(len(all_ids)):
                pad_len = batch_max - len(all_ids[i])
                mask = [1] * len(all_ids[i]) + [0] * pad_len
                all_ids[i] = all_ids[i] + [self.pad_token_id] * pad_len
                attention_masks.append(mask)

            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(all_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                }

        return {"input_ids": all_ids}

    def __call__(self, text, **kwargs):
        """Make tokenizer callable for HuggingFace compatibility."""
        if isinstance(text, str):
            return self.encode(text, **kwargs)
        elif isinstance(text, list):
            return self.batch_encode(text, **kwargs)
        raise ValueError(f"Expected str or list[str], got {type(text)}")

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (f"SelfiesTokenizer(vocab_size={self.vocab_size}, "
                f"max_length={self.max_length})")
