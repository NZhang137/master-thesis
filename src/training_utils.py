"""Minimal REINFORCE training utilities for a LoRA language model."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

try:
    from .evaluation_utils import generate_samples
except ImportError:
    from evaluation_utils import generate_samples


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


@dataclass
class SimpleRLTrainer:
    """Train LoRA parameters from rewards assigned to sampled completions."""

    model: Any
    tokenizer: Any
    reward_model: Any
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    reward_history: list[float] = field(default_factory=list, init=False)
    loss_history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("The model has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
        )

    def compute_policy_loss(
        self,
        samples: Sequence[dict[str, Any]],
        baseline_reward: float | None = None,
    ) -> torch.Tensor:
        """Compute a REINFORCE loss over generated completion tokens."""
        if not samples:
            raise ValueError("At least one generated sample is required.")
        if baseline_reward is None:
            baseline_reward = float(np.mean([sample["reward"] for sample in samples]))

        device = _model_device(self.model)
        losses = []

        for sample in samples:
            tokens = sample["tokens"].unsqueeze(0).to(device)
            outputs = self.model(tokens)
            logits = outputs.logits[:, :-1, :]
            targets = tokens[:, 1:]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                2,
                targets.unsqueeze(-1),
            ).squeeze(-1)

            completion_start = max(int(sample["prompt_len"]) - 1, 0)
            completion_log_probs = token_log_probs[:, completion_start:]
            if completion_log_probs.numel() == 0:
                completion_log_probs = token_log_probs

            advantage = float(sample["reward"]) - baseline_reward
            losses.append(-completion_log_probs.mean() * advantage)

        return torch.stack(losses).mean()

    def train_step(
        self,
        prompts: Sequence[str],
        *,
        num_samples: int = 2,
    ) -> tuple[float, float]:
        """Run one optimizer step for each supplied prompt."""
        if not prompts:
            raise ValueError("At least one training prompt is required.")

        rewards = []
        losses = []
        self.model.train()

        for prompt in prompts:
            samples = generate_samples(
                self.model,
                self.tokenizer,
                prompt,
                reward_model=self.reward_model,
                num_samples=num_samples,
            )
            loss = self.compute_policy_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()

            rewards.append(float(np.mean([sample["reward"] for sample in samples])))
            losses.append(float(loss.item()))

        mean_reward = float(np.mean(rewards))
        mean_loss = float(np.mean(losses))
        self.reward_history.append(mean_reward)
        self.loss_history.append(mean_loss)
        return mean_reward, mean_loss

    def fit(
        self,
        prompts: Sequence[str],
        *,
        epochs: int = 1,
        batch_size: int = 1,
        num_samples: int = 2,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        """Train for a small number of epochs and return epoch summaries."""
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")

        rng = random.Random(seed)
        history = []

        for epoch in range(epochs):
            shuffled_prompts = list(prompts)
            rng.shuffle(shuffled_prompts)
            epoch_rewards = []
            epoch_losses = []

            for start in range(0, len(shuffled_prompts), batch_size):
                reward, loss = self.train_step(
                    shuffled_prompts[start : start + batch_size],
                    num_samples=num_samples,
                )
                epoch_rewards.append(reward)
                epoch_losses.append(loss)

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_reward": float(np.mean(epoch_rewards)),
                    "mean_loss": float(np.mean(epoch_losses)),
                }
            )

        return history
