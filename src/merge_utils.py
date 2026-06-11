"""Utilities for prototype Rewarded-Soups-style LoRA adapter merging.

The weighted merge remains inside a fixed interpolation family. Relationship
matrix and coefficient utilities live in focused companion modules.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer


ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _validate_adapter_path(adapter_path: str | Path, label: str) -> Path:
    """Validate that a local path contains a saved PEFT adapter."""
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} adapter directory not found: {path}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{label} adapter is missing adapter_config.json: {path}"
        )
    if not any((path / filename).is_file() for filename in ADAPTER_WEIGHT_FILENAMES):
        expected = " or ".join(ADAPTER_WEIGHT_FILENAMES)
        raise FileNotFoundError(f"{label} adapter is missing {expected}: {path}")
    return path


def _normalize_two_weights(weights: Sequence[float]) -> list[float]:
    """Validate two non-negative weights and normalize them to the simplex."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (2,):
        raise ValueError("weights must contain helpful and harmless values.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must contain finite, non-negative values.")

    total = values.sum()
    if total <= 0:
        raise ValueError("At least one adapter weight must be positive.")
    return (values / total).tolist()


def compute_relationship_matrix(
    adapter_paths: Sequence[str | Path],
    adapter_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Compute R from flattened LoRA parameters using cosine similarity."""
    try:
        from .relationship_utils import compute_relationship_matrix as compute
    except ImportError:
        from relationship_utils import compute_relationship_matrix as compute

    return compute(adapter_paths, adapter_names=adapter_names)


def relationship_softmax_mapping(
    p: Sequence[float],
    R: np.ndarray,
    tau: float = 1.0,
) -> np.ndarray:
    """Compatibility wrapper for the M1 relationship-softmax mapping."""
    try:
        from .coefficient_utils import relationship_softmax_mapping as compute
    except ImportError:
        from coefficient_utils import relationship_softmax_mapping as compute

    return compute(p, R, tau=tau)


def load_model_with_weighted_lora_adapters(
    model_name: str,
    helpful_adapter_path: str | Path,
    harmless_adapter_path: str | Path,
    weights: Sequence[float],
    device: torch.device | str | None = None,
):
    """Load GPT-2 and activate a weighted helpful/harmless LoRA adapter.

    This function performs only a fixed weighted adapter interpolation for the
    technical merging prototype. It does not compute a relationship matrix or
    implement the final preference-aware coefficient correction method.
    """
    helpful_path = _validate_adapter_path(
        helpful_adapter_path,
        label="Helpful",
    )
    harmless_path = _validate_adapter_path(
        harmless_adapter_path,
        label="Harmless",
    )
    normalized_weights = _normalize_two_weights(weights)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.config.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(
        base_model,
        helpful_path,
        adapter_name="helpful",
        is_trainable=False,
    )
    model.load_adapter(
        harmless_path,
        adapter_name="harmless",
        is_trainable=False,
    )
    # PEFT performs the prototype merge; the thesis notation represents each
    # LoRA adapter through its effective parameter update delta_i.
    model.add_weighted_adapter(
        adapters=["helpful", "harmless"],
        weights=normalized_weights,
        adapter_name="weighted_merge",
        combination_type="linear",
    )
    model.set_adapter("weighted_merge")
    model.to(device)
    model.eval()
    return model, tokenizer


def merge_lora_adapters(
    adapter_paths: Sequence[str | Path],
    lambdas: Sequence[float],
    model_name: str = "gpt2",
    device: torch.device | str | None = None,
):
    """Load two local adapters and return their active weighted merge."""
    if len(adapter_paths) != 2:
        raise ValueError("Exactly two adapter paths are required.")
    return load_model_with_weighted_lora_adapters(
        model_name=model_name,
        helpful_adapter_path=adapter_paths[0],
        harmless_adapter_path=adapter_paths[1],
        weights=lambdas,
        device=device,
    )
