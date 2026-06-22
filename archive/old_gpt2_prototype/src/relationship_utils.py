"""Compute static relationship matrices from LoRA adapter geometry.

The current prototype represents each specialist by its flattened saved LoRA
parameters and uses cosine similarity as a geometry proxy. It does not compute
coefficient corrections or the final lambda = f(p, R) mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file


LORA_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


def load_adapter_state_dict(
    adapter_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Load only LoRA tensors from a local PEFT adapter folder.

    Safetensors is preferred. A PyTorch ``adapter_model.bin`` file is used as
    a fallback when no safetensors file is present.
    """
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {path}")

    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}")

    safetensors_path = path / LORA_WEIGHT_FILENAMES[0]
    bin_path = path / LORA_WEIGHT_FILENAMES[1]

    if safetensors_path.is_file():
        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.is_file():
        try:
            state_dict = torch.load(
                bin_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(bin_path, map_location="cpu")
    else:
        expected = " or ".join(LORA_WEIGHT_FILENAMES)
        raise FileNotFoundError(f"Missing {expected} in {path}")

    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Adapter weights must contain a state dictionary: {path}")

    lora_state = {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if "lora_" in name.lower() and torch.is_tensor(tensor)
    }
    if not lora_state:
        raise ValueError(f"No LoRA tensors were found in adapter weights: {path}")

    return lora_state


def flatten_adapter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten sorted floating-point LoRA tensors into one CPU vector."""
    tensors = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if "lora_" not in name.lower():
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(f"State entry is not a tensor: {name}")
        if not torch.is_floating_point(tensor):
            continue
        flat_tensor = (
            tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        )
        if not torch.isfinite(flat_tensor).all():
            raise ValueError(f"LoRA tensor contains non-finite values: {name}")
        tensors.append(flat_tensor)

    if not tensors:
        raise ValueError("No floating-point LoRA tensors were available to flatten.")
    return torch.cat(tensors)


def cosine_similarity(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute cosine similarity between two equally sized flat vectors."""
    if eps <= 0:
        raise ValueError("eps must be positive.")

    a = vec_a.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    b = vec_b.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(
            f"Adapter vectors must have equal length, got {a.numel()} and {b.numel()}."
        )
    if a.numel() == 0:
        raise ValueError("Adapter vectors must not be empty.")

    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denominator.item() < eps:
        raise ValueError("Cosine similarity is undefined for a near-zero vector.")

    similarity = torch.dot(a, b) / denominator.clamp_min(eps)
    return float(similarity.clamp(-1.0, 1.0).item())


def _validate_compatible_states(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    adapter_names: Sequence[str],
) -> None:
    """Ensure adapters expose matching LoRA keys and tensor shapes."""
    reference = state_dicts[0]
    reference_keys = sorted(reference)

    for name, state_dict in zip(adapter_names[1:], state_dicts[1:]):
        keys = sorted(state_dict)
        if keys != reference_keys:
            raise ValueError(
                f"Adapter {name!r} has different LoRA parameter keys. "
                "All adapters must use the same base model and LoRA configuration."
            )

        for key in reference_keys:
            if state_dict[key].shape != reference[key].shape:
                raise ValueError(
                    f"Adapter {name!r} has an incompatible shape for {key}: "
                    f"{tuple(state_dict[key].shape)} != {tuple(reference[key].shape)}"
                )


def compute_relationship_matrix(
    adapter_paths: Sequence[str | Path],
    adapter_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Compute the cosine relationship matrix for compatible LoRA adapters."""
    if not adapter_paths:
        raise ValueError("At least one adapter path is required.")

    names = list(
        adapter_names
        if adapter_names is not None
        else [f"adapter_{index}" for index in range(len(adapter_paths))]
    )
    if len(names) != len(adapter_paths):
        raise ValueError("adapter_names must match the number of adapter paths.")
    if len(set(names)) != len(names):
        raise ValueError("adapter_names must be unique.")

    state_dicts = [load_adapter_state_dict(path) for path in adapter_paths]
    _validate_compatible_states(state_dicts, names)
    vectors = [flatten_adapter_state_dict(state) for state in state_dicts]

    size = len(vectors)
    matrix = torch.empty((size, size), dtype=torch.float32)
    for row in range(size):
        for column in range(row, size):
            similarity = cosine_similarity(vectors[row], vectors[column])
            matrix[row, column] = similarity
            matrix[column, row] = similarity

    return matrix
