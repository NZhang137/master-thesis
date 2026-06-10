"""Prompt-source utilities for Anthropic HH-RLHF.

This is only a prompt-source prototype. It uses ``example["chosen"]`` to
recover conversational prompts, while the original chosen/rejected preference
pairs are not used for training yet.
"""

from datasets import load_dataset


def extract_prompt_from_hh(text: str, max_chars: int = 800) -> str:
    """Extract the final Human-to-Assistant prompt from an HH-RLHF dialogue."""
    parts = text.split("\n\nHuman:")
    last_turn = f"Human:{parts[-1]}" if len(parts) > 1 else text

    assistant_marker = "\n\nAssistant:"
    marker_index = last_turn.rfind(assistant_marker)
    if marker_index != -1:
        last_turn = last_turn[: marker_index + len(assistant_marker)]

    prompt = last_turn.strip()
    if len(prompt) > max_chars:
        prompt = prompt[-max_chars:]

    return prompt


def load_hh_rlhf_prompts(split: str = "train[:20]") -> list[str]:
    """Load HH-RLHF prompts for the prompt-source prototype.

    Only the ``chosen`` dialogue is read, and only its prompt is returned.
    The chosen/rejected preference pairs are intentionally not used yet.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    return [extract_prompt_from_hh(example["chosen"]) for example in dataset]
