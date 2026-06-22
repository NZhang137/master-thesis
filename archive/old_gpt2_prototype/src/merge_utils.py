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


def _normalize_weights(
    weights: Sequence[float],
    expected_length: int,
) -> list[float]:
    """Validate non-negative adapter weights and normalize them to sum to one."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (expected_length,):
        raise ValueError(
            f"weights must contain exactly {expected_length} values."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must contain finite, non-negative values.")

    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one adapter weight must be positive.")
    return (values / total).tolist()


def _normalize_two_weights(weights: Sequence[float]) -> list[float]:
    """Compatibility helper for the two-objective prototype."""
    return _normalize_weights(weights, expected_length=2)


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
    """Compatibility wrapper for the legacy relationship-softmax mapping."""
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
    return load_model_with_weighted_lora_adapters_multi(
        model_name=model_name,
        adapter_paths=[helpful_adapter_path, harmless_adapter_path],
        adapter_names=["helpful", "harmless"],
        weights=_normalize_two_weights(weights),
        device=device,
    )


def load_model_with_weighted_lora_adapters_multi(
    model_name: str,
    adapter_paths: Sequence[str | Path],
    adapter_names: Sequence[str],
    weights: Sequence[float],
    device: torch.device | str | None = None,
):
    """Load GPT-2 and activate a weighted merge of multiple LoRA adapters.

    All adapters must be compatible with the same fresh base model. PEFT
    performs a linear weighted merge, while the thesis notation represents the
    adapters through their effective parameter updates.
    """
    if not model_name.strip():
        raise ValueError("model_name must be non-empty.")

    paths = list(adapter_paths)
    names = [str(name).strip() for name in adapter_names]
    if len(paths) < 2:
        raise ValueError("At least two adapter paths are required.")
    if len(names) != len(paths) or len(weights) != len(paths):
        raise ValueError(
            "adapter_paths, adapter_names, and weights must have the same length."
        )
    if any(not name for name in names):
        raise ValueError("adapter_names must contain non-empty names.")
    if len(set(names)) != len(names):
        raise ValueError("adapter_names must be unique.")
    if "weighted_merge" in names:
        raise ValueError("adapter name 'weighted_merge' is reserved.")

    normalized_weights = _normalize_weights(
        weights,
        expected_length=len(paths),
    )
    validated_paths = [
        _validate_adapter_path(path, label=name)
        for name, path in zip(names, paths)
    ]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.config.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(
        base_model,
        validated_paths[0],
        adapter_name=names[0],
        is_trainable=False,
    )
    for adapter_path, adapter_name in zip(
        validated_paths[1:],
        names[1:],
    ):
        model.load_adapter(
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=False,
        )

    model.add_weighted_adapter(
        adapters=names,
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
