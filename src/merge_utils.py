"""LoRA adapter geometry and weighted merging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def normalize_coefficients(
    coefficients: Sequence[float],
    *,
    allow_negative: bool = False,
) -> np.ndarray:
    """Validate coefficients and normalize them to sum to one."""
    weights = np.asarray(coefficients, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Coefficients must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Coefficients must be finite.")
    if not allow_negative and np.any(weights < 0):
        raise ValueError("Coefficients must be non-negative.")

    total = float(weights.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Coefficients must not sum to zero.")
    return weights / total


def flatten_adapter_state(
    state_dict: Mapping[str, torch.Tensor],
    *,
    trainable_only: bool = True,
) -> torch.Tensor:
    """Flatten a LoRA state dictionary into one CPU float vector."""
    tensors = []
    for name in sorted(state_dict):
        if trainable_only and "lora_" not in name:
            continue
        tensor = state_dict[name]
        if torch.is_floating_point(tensor):
            tensors.append(tensor.detach().float().cpu().reshape(-1))

    if not tensors:
        raise ValueError("No floating-point LoRA tensors were found.")
    return torch.cat(tensors)


def cosine_relationship_matrix(
    adapter_vectors: Sequence[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute a cosine-similarity matrix between flattened adapters."""
    if not adapter_vectors:
        raise ValueError("At least one adapter vector is required.")
    if len({vector.numel() for vector in adapter_vectors}) != 1:
        raise ValueError("All adapter vectors must have the same length.")

    vectors = torch.stack(
        [vector.detach().float().cpu().reshape(-1) for vector in adapter_vectors]
    )
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(eps)
    normalized = vectors / norms
    return (normalized @ normalized.T).numpy()


def correct_coefficients(
    preference: Sequence[float],
    relationship_matrix: np.ndarray,
    *,
    correction_strength: float = 0.0,
) -> np.ndarray:
    """Apply a simple geometry-aware correction and project to the simplex.

    A strength of zero reproduces the direct Rewarded-Soups baseline
    lambda = p. Positive values reduce weight in directions that are already
    strongly represented by correlated adapters.
    """
    preference_vector = normalize_coefficients(preference)
    relationships = np.asarray(relationship_matrix, dtype=np.float64)
    expected_shape = (preference_vector.size, preference_vector.size)
    if relationships.shape != expected_shape:
        raise ValueError(f"relationship_matrix must have shape {expected_shape}.")

    redundancy = relationships @ preference_vector
    corrected = preference_vector - correction_strength * redundancy
    corrected = np.clip(corrected, 0.0, None)

    if np.isclose(corrected.sum(), 0.0):
        return preference_vector
    return normalize_coefficients(corrected)


def load_and_merge_adapters(
    base_model: Any,
    adapter_paths: Sequence[str | Path],
    coefficients: Sequence[float],
    *,
    adapter_names: Sequence[str] | None = None,
    merged_name: str = "merged",
    combination_type: str = "linear",
) -> Any:
    """Load compatible LoRA adapters and activate their weighted merge."""
    from peft import PeftModel

    if len(adapter_paths) < 2:
        raise ValueError("At least two adapters are required for merging.")
    weights = normalize_coefficients(coefficients)
    if len(adapter_paths) != len(weights):
        raise ValueError("Each adapter requires exactly one coefficient.")

    names = list(adapter_names or [f"adapter_{i}" for i in range(len(adapter_paths))])
    if len(names) != len(adapter_paths) or len(set(names)) != len(names):
        raise ValueError("Adapter names must be unique and match adapter_paths.")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name=names[0],
        is_trainable=False,
    )
    for path, name in zip(adapter_paths[1:], names[1:]):
        model.load_adapter(path, adapter_name=name, is_trainable=False)

    model.add_weighted_adapter(
        adapters=names,
        weights=weights.tolist(),
        adapter_name=merged_name,
        combination_type=combination_type,
    )
    model.set_adapter(merged_name)
    return model


def save_merged_adapter(
    model: Any,
    output_dir: str | Path,
    *,
    adapter_name: str = "merged",
) -> Path:
    """Save one merged adapter without serializing the full base model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, selected_adapters=[adapter_name])
    return output_path
