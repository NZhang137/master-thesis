"""Generation and lightweight reward evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sentiment_reward(result: dict[str, Any]) -> float:
    label = str(result["label"]).lower()
    score = float(result["score"])

    if "positive" in label or label == "label_2":
        return score
    if "neutral" in label or label == "label_1":
        return 0.5
    return 1.0 - score


def _toxicity_reward(result: dict[str, Any]) -> float:
    label = str(result["label"]).lower().replace("_", " ")
    score = float(result["score"])

    if "non toxic" in label or "not toxic" in label:
        return score
    if "toxic" in label or label == "label_1":
        return 1.0 - score
    return 1.0 - score


def length_reward(text: str, min_words: int = 10, max_words: int = 50) -> float:
    """Score responses highest when their length lies in a target interval."""
    words = len(text.split())
    if min_words <= words <= max_words:
        return 1.0
    if words < min_words:
        return words / max(min_words, 1)
    return max(0.1, 1.0 - (words - max_words) / 100)


@dataclass
class CompositeRewardModel:
    """Combine sentiment, non-toxicity, and response-length signals."""

    sentiment_weight: float = 0.4
    toxicity_weight: float = 0.4
    length_weight: float = 0.2
    classifier_device: int = -1

    def __post_init__(self) -> None:
        from transformers import pipeline

        total_weight = (
            self.sentiment_weight + self.toxicity_weight + self.length_weight
        )
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Reward weights must sum to 1.")

        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=self.classifier_device,
        )
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=self.classifier_device,
        )

    def get_reward(self, text: str) -> float:
        """Calculate a scalar reward in the interval [0, 1]."""
        clipped_text = text[:1000]
        sentiment = self.sentiment_classifier(
            clipped_text,
            truncation=True,
            max_length=512,
        )[0]
        toxicity = self.toxicity_classifier(
            clipped_text,
            truncation=True,
            max_length=512,
        )[0]

        reward = (
            self.sentiment_weight * _sentiment_reward(sentiment)
            + self.toxicity_weight * _toxicity_reward(toxicity)
            + self.length_weight * length_reward(clipped_text)
        )
        return float(np.clip(reward, 0.0, 1.0))


def generate_samples(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    reward_model: Any | None = None,
    num_samples: int = 2,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> list[dict[str, Any]]:
    """Generate completions and optionally attach a reward to each one."""
    device = _model_device(model)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
            )
    finally:
        model.train(was_training)

    samples = []
    for output in outputs:
        completion = tokenizer.decode(
            output[prompt_length:],
            skip_special_tokens=True,
        ).strip()
        sample = {
            "text": tokenizer.decode(output, skip_special_tokens=True),
            "completion": completion,
            "tokens": output.detach().cpu(),
            "prompt_len": prompt_length,
        }
        if reward_model is not None:
            sample["reward"] = reward_model.get_reward(completion or sample["text"])
        samples.append(sample)

    return samples


def evaluate_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    reward_model: Any,
    *,
    num_samples: int = 2,
) -> dict[str, Any]:
    """Evaluate the best sampled completion for each prompt."""
    results = []
    for prompt in prompts:
        samples = generate_samples(
            model,
            tokenizer,
            prompt,
            reward_model=reward_model,
            num_samples=num_samples,
        )
        best = max(samples, key=lambda sample: sample["reward"])
        results.append({"prompt": prompt, **best})

    rewards = [result["reward"] for result in results]
    return {
        "results": results,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
    }
