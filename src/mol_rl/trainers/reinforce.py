"""
REINFORCE trainer with RLOO baseline for molecular generation.

Implements policy gradient optimization with:
- Leave-One-Out (RLOO) baseline for variance reduction
- KL divergence penalty against a frozen reference model
- Reward shaping via RewardFunction

Usage:
    from mol_rl.trainers.reinforce import ReinforceTrainer

    trainer = ReinforceTrainer(policy, ref_model, tokenizer, reward_fn, ...)
    trainer.train(n_steps=2000)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from mol_rl.data.selfies_tokenizer import SelfiesTokenizer
from mol_rl.models.rewards import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class ReinforceConfig:
    """Configuration for REINFORCE trainer."""
    batch_size: int = 64
    mini_batch_size: int = 16
    rloo_k: int = 4
    kl_coef: float = 0.2
    kl_target: Optional[float] = 6.0  # Target KL for adaptive control; None = fixed
    max_length: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.95
    max_grad_norm: float = 1.0
    fp16: bool = True


class ReinforceTrainer:
    """
    REINFORCE trainer with RLOO baseline for molecular RL.

    The training loop:
    1. Generate batch_size molecules from the current policy
    2. Compute rewards using RewardFunction (SELFIES → SMILES → RDKit)
    3. Compute per-token log-probs under policy and reference model
    4. Use RLOO baseline: advantage_i = reward_i - mean(rewards_{j != i})
    5. Policy gradient: loss = -mean(advantage * sum_log_prob) + kl_coef * KL
    6. Update policy parameters
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer: SelfiesTokenizer,
        reward_fn: RewardFunction,
        config: ReinforceConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.device = device

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def generate(self, batch_size: int) -> dict:
        """
        Generate molecules from the current policy.

        Returns dict with:
            - sequences: (batch_size, seq_len) token IDs
            - log_probs: (batch_size, seq_len) per-token log-probs
            - attention_mask: (batch_size, seq_len) mask for valid tokens
        """
        self.policy.eval()
        cfg = self.config

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Start with BOS
        input_ids = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=self.device
        )
        all_log_probs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        use_amp = cfg.fp16 and self.device.type == "cuda"

        for step in range(cfg.max_length - 1):
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = self.policy(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]  # (batch, vocab)

            # Temperature
            if cfg.temperature != 1.0:
                logits = logits / cfg.temperature

            # Top-k
            if cfg.top_k > 0:
                top_k_vals = torch.topk(logits, cfg.top_k)[0]
                logits[logits < top_k_vals[..., -1, None]] = float("-inf")

            # Top-p
            if cfg.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > cfg.top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Log-prob of sampled token
            log_prob = F.log_softmax(logits, dim=-1)
            token_log_prob = log_prob.gather(1, next_token)  # (batch, 1)

            # Mask finished sequences
            token_log_prob = token_log_prob.masked_fill(finished.unsqueeze(1), 0.0)
            next_token = next_token.masked_fill(
                finished.unsqueeze(1), self.tokenizer.pad_token_id
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)
            all_log_probs.append(token_log_prob)

            # Update finished
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        # Stack log_probs: (batch, generated_len)
        log_probs = torch.cat(all_log_probs, dim=1)

        # Build attention mask (1 for real tokens including BOS, 0 for pad after EOS)
        sequences = input_ids
        attention_mask = torch.ones_like(sequences)
        for i in range(batch_size):
            eos_positions = (sequences[i] == eos_id).nonzero(as_tuple=False)
            if len(eos_positions) > 0:
                eos_pos = eos_positions[0].item()
                attention_mask[i, eos_pos + 1:] = 0

        self.policy.train()
        return {
            "sequences": sequences,
            "log_probs": log_probs,
            "attention_mask": attention_mask,
        }

    def compute_ref_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute per-token log-probs under the reference model."""
        cfg = self.config
        use_amp = cfg.fp16 and self.device.type == "cuda"

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = self.ref_model(input_ids=sequences)
                logits = outputs.logits  # (batch, seq_len, vocab)

        # Shift: logits at position t predict token at t+1
        # So log_prob for token at position t+1 is from logits at position t
        shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
        shift_tokens = sequences[:, 1:]    # (batch, seq_len-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_tokens.unsqueeze(2)).squeeze(2)

        return token_log_probs

    def compute_policy_log_probs_from_sequences(
        self, sequences: torch.Tensor
    ) -> torch.Tensor:
        """Re-compute per-token log-probs under the current policy (with gradients).

        Uses eval mode to disable dropout — consistent with generation and
        reference model. Gradients still flow (eval mode only affects dropout/batchnorm).
        """
        cfg = self.config
        use_amp = cfg.fp16 and self.device.type == "cuda"

        self.policy.eval()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = self.policy(input_ids=sequences)
            logits = outputs.logits
        self.policy.train()

        shift_logits = logits[:, :-1, :]
        shift_tokens = sequences[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_tokens.unsqueeze(2)).squeeze(2)

        return token_log_probs

    def decode_sequences(self, sequences: torch.Tensor) -> list[str]:
        """Decode token sequences to SELFIES strings."""
        selfies_list = []
        for seq in sequences:
            selfies_str = self.tokenizer.decode(seq, skip_special_tokens=True)
            selfies_list.append(selfies_str)
        return selfies_list

    def compute_rloo_advantages(self, rewards: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute RLOO (Leave-One-Out) advantages.

        rewards: (batch_size,) tensor of rewards
        k: group size for RLOO

        The batch is split into groups of k. For each sample i in a group,
        the baseline is the mean reward of the other k-1 samples.
        advantage_i = reward_i - mean(rewards_{j != i})
        """
        batch_size = rewards.shape[0]
        n_groups = batch_size // k

        if n_groups == 0:
            # Fall back to simple baseline (mean of all)
            return rewards - rewards.mean()

        # Trim to exact multiple of k
        trimmed = rewards[:n_groups * k]
        grouped = trimmed.view(n_groups, k)

        # RLOO baseline: for sample i, baseline = (sum - reward_i) / (k - 1)
        group_sums = grouped.sum(dim=1, keepdim=True)  # (n_groups, 1)
        baselines = (group_sums - grouped) / max(k - 1, 1)  # (n_groups, k)
        advantages = grouped - baselines  # (n_groups, k)
        advantages = advantages.view(-1)

        # Handle leftover samples (simple baseline)
        if batch_size > n_groups * k:
            leftover_rewards = rewards[n_groups * k:]
            leftover_adv = leftover_rewards - leftover_rewards.mean()
            advantages = torch.cat([advantages, leftover_adv])

        return advantages

    def step(self, optimizer: torch.optim.Optimizer, scaler=None) -> dict:
        """
        Perform one RL training step.

        Returns dict with training metrics.
        """
        cfg = self.config

        # 1. Generate sequences
        gen = self.generate(cfg.batch_size)
        sequences = gen["sequences"]
        gen_log_probs = gen["log_probs"]  # (batch, gen_len) - from sampling
        attn_mask = gen["attention_mask"]

        # Mask for generated tokens only (exclude BOS at position 0)
        gen_mask = attn_mask[:, 1:sequences.shape[1]]
        # Trim gen_log_probs to match gen_mask length
        gen_len = gen_mask.shape[1]
        gen_log_probs = gen_log_probs[:, :gen_len]

        # 2. Decode and compute rewards
        selfies_list = self.decode_sequences(sequences)
        rewards_tensor = self.reward_fn.get_rewards_tensor(selfies_list).to(self.device)

        # 3. Compute reference log-probs
        ref_log_probs = self.compute_ref_log_probs(sequences)[:, :gen_len]

        # 4. Compute RLOO advantages
        advantages = self.compute_rloo_advantages(rewards_tensor, cfg.rloo_k)
        # Trim advantages to match batch size (in case of trimming in RLOO)
        advantages = advantages[:sequences.shape[0]]

        # 5. Policy gradient with mini-batches (gradient accumulation)
        self.policy.train()
        optimizer.zero_grad()

        batch_size = sequences.shape[0]
        n_mini = (batch_size + cfg.mini_batch_size - 1) // cfg.mini_batch_size
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl = 0.0

        for mb_start in range(0, batch_size, cfg.mini_batch_size):
            mb_end = min(mb_start + cfg.mini_batch_size, batch_size)
            mb_seq = sequences[mb_start:mb_end]
            mb_mask = gen_mask[mb_start:mb_end]
            mb_adv = advantages[mb_start:mb_end]
            mb_ref_lp = ref_log_probs[mb_start:mb_end]

            # Re-compute policy log-probs with gradients
            mb_policy_lp = self.compute_policy_log_probs_from_sequences(mb_seq)
            mb_policy_lp = mb_policy_lp[:, :gen_len]

            # Per-sequence log-prob (sum of per-token log-probs)
            seq_log_prob = (mb_policy_lp * mb_mask).sum(dim=1)

            # Policy gradient loss: -advantage * log_prob
            pg_loss = -(mb_adv * seq_log_prob).mean()

            # KL divergence: sum_t [policy_lp - ref_lp] per sequence
            kl_per_token = mb_policy_lp - mb_ref_lp
            kl_per_seq = (kl_per_token * mb_mask).sum(dim=1)
            kl_loss = kl_per_seq.mean()

            loss = (pg_loss + cfg.kl_coef * kl_loss) / n_mini

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * n_mini
            total_pg_loss += pg_loss.item()
            total_kl += kl_loss.item()

        # Gradient clipping and optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # Adaptive KL coefficient
        mean_kl = total_kl / n_mini
        if cfg.kl_target is not None and mean_kl > 0:
            # Increase coef if KL > 1.5 * target, decrease if KL < target / 1.5
            if mean_kl > 1.5 * cfg.kl_target:
                cfg.kl_coef = min(cfg.kl_coef * 1.5, 10.0)
            elif mean_kl < cfg.kl_target / 1.5:
                cfg.kl_coef = max(cfg.kl_coef / 1.5, 0.01)

        # Metrics
        scores = self.reward_fn.score_selfies_batch(selfies_list)
        valid_count = sum(1 for s in scores if s.is_valid)

        metrics = {
            "loss": total_loss / n_mini,
            "pg_loss": total_pg_loss / n_mini,
            "kl": mean_kl,
            "kl_coef": cfg.kl_coef,
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "reward_max": rewards_tensor.max().item(),
            "validity": valid_count / len(scores),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
        }

        return metrics
