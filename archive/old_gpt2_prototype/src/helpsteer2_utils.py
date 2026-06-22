"""Data utilities for the lightweight NVIDIA HelpSteer2 prototype.

This module uses HelpSteer2 attribute ratings to construct objective-specific
supervised training texts. It is not full RLHF or PPO, does not train reward
models, and does not turn the ratings into a final preference-training method.
"""

from __future__ import annotations

from collections.abc import Mapping

from datasets import load_dataset


HELPSTEER2_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
HIGH_ATTRIBUTE_SCORE = 3


def load_helpsteer2_dataset(split: str = "train[:100]"):
    """Load a split or split slice from ``nvidia/HelpSteer2``."""
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string.")
    return load_dataset("nvidia/HelpSteer2", split=split)


def format_helpsteer2_text(example: Mapping[str, object]) -> str:
    """Format one HelpSteer2 prompt and response for causal LM training."""
    if "prompt" not in example or "response" not in example:
        raise ValueError("Each HelpSteer2 example needs prompt and response fields.")
    if example["prompt"] is None or example["response"] is None:
        raise ValueError("HelpSteer2 prompt and response fields must be non-empty.")

    prompt = str(example["prompt"]).strip()
    response = str(example["response"]).strip()
    if not prompt or not response:
        raise ValueError("HelpSteer2 prompt and response fields must be non-empty.")

    return f"Human: {prompt}\n\nAssistant: {response}"


def make_attribute_training_texts(
    attribute: str,
    split: str = "train[:100]",
) -> list[str]:
    """Build supervised texts from examples with high attribute ratings.

    The candidate split is sorted by the requested HelpSteer2 score, and
    examples rated at least 3 out of 4 are selected. If a very small split has
    no score of 3 or 4, examples with the highest observed score are used so
    smoke tests remain runnable.

    This is a lightweight objective-specific data-selection prototype. It is
    not full RLHF or PPO, does not train a reward model, and uses ratings only
    to select supervised prompt/response texts.
    """
    normalized_attribute = attribute.strip().lower()
    if normalized_attribute not in HELPSTEER2_ATTRIBUTES:
        supported = ", ".join(HELPSTEER2_ATTRIBUTES)
        raise ValueError(
            f"Unsupported HelpSteer2 attribute {attribute!r}. "
            f"Choose one of: {supported}."
        )

    dataset = load_helpsteer2_dataset(split=split)
    scored_texts = []

    for index, example in enumerate(dataset):
        if normalized_attribute not in example:
            raise ValueError(
                f"HelpSteer2 example is missing {normalized_attribute!r}."
            )
        try:
            numeric_score = float(example[normalized_attribute])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"HelpSteer2 {normalized_attribute!r} ratings must be integers."
            ) from error
        if not numeric_score.is_integer():
            raise ValueError(
                f"HelpSteer2 {normalized_attribute!r} ratings must be integers."
            )
        score = int(numeric_score)
        if score < 0 or score > 4:
            raise ValueError(
                f"HelpSteer2 {normalized_attribute!r} rating must be in [0, 4]."
            )

        scored_texts.append(
            (score, index, format_helpsteer2_text(example))
        )

    if not scored_texts:
        raise ValueError(f"HelpSteer2 split contains no examples: {split!r}.")

    selected = [
        item for item in scored_texts if item[0] >= HIGH_ATTRIBUTE_SCORE
    ]
    if not selected:
        highest_score = max(item[0] for item in scored_texts)
        selected = [item for item in scored_texts if item[0] == highest_score]

    selected.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, _, text in selected]
