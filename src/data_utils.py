"""Data loading and prompt extraction helpers for Anthropic HH-RLHF."""

from __future__ import annotations

from typing import Any


HH_RLHF_DATASET = "Anthropic/hh-rlhf"
ASSISTANT_MARKER = "\n\nAssistant:"


def extract_prompt_from_hh(text: str, max_chars: int | None = 800) -> str:
    """Extract the final Human-to-Assistant prompt from an HH-RLHF dialogue."""
    parts = text.split("\n\nHuman:")
    last_turn = f"Human:{parts[-1]}" if len(parts) > 1 else text

    assistant_index = last_turn.rfind(ASSISTANT_MARKER)
    if assistant_index != -1:
        last_turn = last_turn[: assistant_index + len(ASSISTANT_MARKER)]

    prompt = last_turn.strip()
    if max_chars is not None and len(prompt) > max_chars:
        prompt = prompt[-max_chars:]

    return prompt


def extract_chosen_response(text: str) -> str:
    """Return the final assistant response from an HH-RLHF chosen dialogue."""
    assistant_index = text.rfind(ASSISTANT_MARKER)
    if assistant_index == -1:
        return ""
    return text[assistant_index + len(ASSISTANT_MARKER) :].strip()


def load_hh_rlhf(
    split: str = "train",
    limit: int | None = 20,
    dataset_name: str = HH_RLHF_DATASET,
) -> Any:
    """Load an HH-RLHF split, optionally restricted to the first examples."""
    from datasets import load_dataset

    selected_split = f"{split}[:{limit}]" if limit is not None else split
    return load_dataset(dataset_name, split=selected_split)


def load_hh_prompts(
    split: str = "train",
    limit: int | None = 20,
    source_column: str = "chosen",
    max_chars: int | None = 800,
) -> list[str]:
    """Load HH-RLHF and return prompts extracted from one dialogue column."""
    dataset = load_hh_rlhf(split=split, limit=limit)
    return [
        extract_prompt_from_hh(example[source_column], max_chars=max_chars)
        for example in dataset
    ]


def build_sft_examples(
    split: str = "train",
    limit: int | None = 20,
    max_chars: int | None = 800,
) -> list[dict[str, str]]:
    """Build prompt-response examples from the preferred HH-RLHF answers."""
    dataset = load_hh_rlhf(split=split, limit=limit)
    examples = []

    for row in dataset:
        chosen = row["chosen"]
        examples.append(
            {
                "prompt": extract_prompt_from_hh(chosen, max_chars=max_chars),
                "response": extract_chosen_response(chosen),
            }
        )

    return examples
