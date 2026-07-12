"""Exact effective-LoRA merging for coefficient-space experiments.

The thesis interpolation family is

    theta(lambda) = theta_0 + sum_i lambda_i * delta_i

where each LoRA task vector is the effective update

    delta_i,l = (lora_alpha_i,l / r_i,l) * B_i,l @ A_i,l.

Do not use PEFT `add_weighted_adapter(..., combination_type="linear")` for this
experiment. In PEFT 0.10.0 that path combines factors as

    A_merged = sum_i sqrt(lambda_i * s_i) A_i
    B_merged = sum_i sqrt(lambda_i * s_i) B_i

with the merged adapter scaling set to 1. The resulting update is

    B_merged @ A_merged
      = sum_i lambda_i * s_i * B_i @ A_i
        + sum_{i != j} sqrt(lambda_i lambda_j s_i s_j) * B_i @ A_j,

so every interior lambda contains cross terms that are not part of
theta_0 + sum_i lambda_i delta_i. PEFT `cat` can represent the exact sum, but
this module applies the effective updates directly so the geometry and reward
evaluation use the same task vectors.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _adapter_names(adapter_paths: Mapping[str, Path]) -> list[str]:
    """Return adapter names in insertion order and validate non-empty input."""
    names = list(adapter_paths.keys())
    if not names:
        raise ValueError("At least one adapter path is required.")
    return names


def _validate_geometries(
    adapter_paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    """Load and validate compatible effective-LoRA geometries."""
    from src.effective_lora_geometry import (
        load_effective_lora_geometry,
        validate_compatible_geometries,
    )

    names = _adapter_names(adapter_paths)
    geometries = {
        name: load_effective_lora_geometry(Path(adapter_paths[name]))
        for name in names
    }
    validate_compatible_geometries([geometries[name] for name in names], names)

    ranks = {
        int(layer.lora_a.shape[0])
        for geometry in geometries.values()
        for layer in geometry.values()
    }
    if len(ranks) != 1:
        raise ValueError(f"Adapters must use one shared LoRA rank; found ranks {sorted(ranks)}.")
    return geometries


def effective_deltas(adapter_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Return per-adapter, per-module effective updates in float32 on CPU."""
    import torch

    geometries = _validate_geometries(adapter_paths)
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    for adapter_name, geometry in geometries.items():
        adapter_deltas: dict[str, torch.Tensor] = {}
        for module_name, layer in geometry.items():
            lora_a = layer.lora_a.detach().to(device="cpu", dtype=torch.float32)
            lora_b = layer.lora_b.detach().to(device="cpu", dtype=torch.float32)
            adapter_deltas[module_name] = torch.mul(lora_b @ lora_a, float(layer.scaling))
        deltas[adapter_name] = adapter_deltas
    return deltas


def combine_effective_deltas(
    lmbda: np.ndarray | list[float] | tuple[float, ...],
    deltas_by_adapter: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Linearly combine already-materialized effective LoRA deltas."""
    names = list(deltas_by_adapter.keys())
    weights = np.asarray(lmbda, dtype=np.float64)
    if weights.shape != (len(names),):
        raise ValueError(f"lambda must have shape {(len(names),)}, got {weights.shape}.")

    reference_modules = sorted(deltas_by_adapter[names[0]])
    for name in names[1:]:
        if sorted(deltas_by_adapter[name]) != reference_modules:
            raise ValueError(f"Adapter {name!r} has a different effective-delta module set.")

    merged: dict[str, Any] = {}
    for module_name in reference_modules:
        first = deltas_by_adapter[names[0]][module_name]
        if hasattr(first, "detach"):
            import torch

            total = torch.zeros_like(first, dtype=torch.float32, device="cpu")
            for adapter_name, weight in zip(names, weights):
                total.add_(deltas_by_adapter[adapter_name][module_name].to(dtype=torch.float32), alpha=float(weight))
        else:
            total = np.zeros_like(np.asarray(first, dtype=np.float32), dtype=np.float32)
            for adapter_name, weight in zip(names, weights):
                total += float(weight) * np.asarray(deltas_by_adapter[adapter_name][module_name], dtype=np.float32)
        merged[module_name] = total
    return merged


def _candidate_module_names(module_name: str) -> list[str]:
    """Return possible base-model module names for one PEFT LoRA module key."""
    candidates = [module_name]
    prefixes = ("base_model.model.", "model.")
    for prefix in prefixes:
        if module_name.startswith(prefix):
            candidates.append(module_name[len(prefix):])
    if module_name.startswith("base_model.model."):
        candidates.append(module_name[len("base_model.model."):])
    if module_name.startswith("base_model.model.model."):
        candidates.append(module_name[len("base_model.model."):])
    return list(dict.fromkeys(candidates))


def resolve_base_module(model: Any, module_name: str) -> Any:
    """Find the base linear module corresponding to one adapter module name."""
    modules = dict(model.named_modules())
    for candidate in _candidate_module_names(module_name):
        if candidate in modules:
            module = modules[candidate]
            if not hasattr(module, "weight"):
                raise ValueError(f"Resolved module has no weight: {candidate}")
            return module

    suffix_matches = [
        module
        for name, module in modules.items()
        if any(name.endswith(f".{candidate}") for candidate in _candidate_module_names(module_name))
        and hasattr(module, "weight")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise KeyError(f"Could not resolve LoRA target module {module_name!r} in base model.")


def apply_effective_deltas_to_model(
    model: Any,
    merged_deltas: Mapping[str, Any],
) -> None:
    """Add effective deltas to the corresponding base weights in place."""
    import torch

    for module_name, delta_cpu in merged_deltas.items():
        module = resolve_base_module(model, module_name)
        weight = module.weight
        if tuple(weight.shape) != tuple(delta_cpu.shape):
            raise ValueError(
                f"Shape mismatch for {module_name}: base={tuple(weight.shape)} delta={tuple(delta_cpu.shape)}"
            )
        device = weight.device
        target_dtype = weight.dtype
        update = weight.detach().to(dtype=torch.float32) + delta_cpu.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            weight.copy_(update.to(dtype=target_dtype))


def merge_theta(
    lmbda: np.ndarray | list[float] | tuple[float, ...],
    adapter_paths: dict[str, Path],
    base_path: Path,
    dtype: Any = None,
) -> Any:
    """Load theta_SFT and add sum_i lambda_i delta_i directly to target weights."""
    import torch
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(base_path), **load_kwargs)
    deltas = effective_deltas(adapter_paths)
    merged = combine_effective_deltas(lmbda, deltas)
    apply_effective_deltas_to_model(model, merged)
    model.eval()
    return model


def merge_delta_norm_check(
    lmbda: np.ndarray | list[float] | tuple[float, ...],
    adapter_paths: dict[str, Path],
) -> float:
    """Return relative error of the exact effective-delta combination."""
    import torch

    deltas = effective_deltas(adapter_paths)
    merged = combine_effective_deltas(lmbda, deltas)
    names = list(deltas.keys())
    weights = np.asarray(lmbda, dtype=np.float64)
    total_sq = 0.0
    err_sq = 0.0
    for module_name in sorted(merged):
        exact = torch.zeros_like(merged[module_name], dtype=torch.float32)
        for adapter_name, weight in zip(names, weights):
            exact.add_(deltas[adapter_name][module_name].to(dtype=torch.float32), alpha=float(weight))
        diff = merged[module_name] - exact
        err_sq += float(torch.sum(diff.float() * diff.float()).item())
        total_sq += float(torch.sum(exact.float() * exact.float()).item())
    if total_sq <= 0.0:
        return 0.0 if err_sq == 0.0 else float("inf")
    return float((err_sq ** 0.5) / (total_sq ** 0.5))
