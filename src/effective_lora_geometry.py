"""Layer-wise geometry for effective LoRA updates.

The effective update of one LoRA layer is
``delta_W = scaling * (B @ A)``. This module computes Frobenius inner products
directly from the low-rank factors, without materializing full update matrices.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


LORA_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")
_LORA_FACTOR_PATTERN = re.compile(
    r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$"
)


@dataclass(frozen=True)
class EffectiveLoraLayer:
    """Low-rank factors and PEFT scaling for one adapted module."""

    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scaling: float

    @property
    def effective_shape(self) -> tuple[int, int]:
        """Return the output-by-input shape of the effective update."""
        return (int(self.lora_b.shape[0]), int(self.lora_a.shape[1]))


def _parse_lora_factor_key(name: str) -> tuple[str, str] | None:
    """Return the canonical module name and factor label for a PEFT key."""
    match = _LORA_FACTOR_PATTERN.fullmatch(name)
    if match is None:
        return None
    return match.group("module"), match.group("factor")


def load_adapter_config(adapter_path: str | Path) -> dict[str, Any]:
    """Load and validate the PEFT adapter configuration."""
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {path}")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Adapter config must contain a JSON object: {config_path}")
    return config


def load_lora_factor_state_dict(
    adapter_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Load only saved LoRA A/B factor tensors from a PEFT adapter."""
    path = Path(adapter_path)
    load_adapter_config(path)
    safetensors_path = path / LORA_WEIGHT_FILENAMES[0]
    bin_path = path / LORA_WEIGHT_FILENAMES[1]

    if safetensors_path.is_file():
        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.is_file():
        try:
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(bin_path, map_location="cpu")
    else:
        expected = " or ".join(LORA_WEIGHT_FILENAMES)
        raise FileNotFoundError(f"Missing {expected} in {path}")

    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Adapter weights must contain a state dictionary: {path}")
    factors = {
        key: tensor.detach().cpu()
        for key, tensor in state_dict.items()
        if _parse_lora_factor_key(key) is not None and torch.is_tensor(tensor)
    }
    if not factors:
        raise ValueError(f"No LoRA A/B tensors were found in adapter weights: {path}")
    return factors


def _resolve_pattern_value(
    pattern: Mapping[str, Any] | None,
    module_name: str,
    default: Any,
) -> Any:
    """Resolve a PEFT per-module pattern using the longest suffix match."""
    if not pattern:
        return default
    matches = [
        (key, value)
        for key, value in pattern.items()
        if module_name == key or module_name.endswith(f".{key}")
    ]
    if not matches:
        return default
    return max(matches, key=lambda item: len(item[0]))[1]


def _group_lora_factors(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Group saved LoRA A/B tensors by adapted module."""
    factors: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        parsed = _parse_lora_factor_key(key)
        if parsed is None:
            continue
        module_name, factor_name = parsed
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError(f"LoRA factor must be a two-dimensional tensor: {key}")
        if not torch.is_floating_point(tensor):
            raise ValueError(f"LoRA factor must be floating point: {key}")
        value = tensor.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError(f"LoRA factor contains non-finite values: {key}")
        module_factors = factors.setdefault(module_name, {})
        if factor_name in module_factors:
            raise ValueError(f"Duplicate LoRA {factor_name} factor for {module_name}")
        module_factors[factor_name] = value

    if not factors:
        raise ValueError("No LoRA A/B factor pairs were found.")
    for module_name, module_factors in factors.items():
        missing = {"A", "B"}.difference(module_factors)
        if missing:
            raise ValueError(
                f"Incomplete LoRA factors for {module_name}: missing {sorted(missing)}"
            )
    return factors


def prepare_effective_lora_geometry(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, EffectiveLoraLayer]:
    """Prepare layer factors and scaling for effective LoRA updates."""
    if config.get("use_dora", False):
        raise ValueError("DoRA requires a different effective-update formula.")
    if config.get("modules_to_save"):
        raise ValueError(
            "Adapters with modules_to_save are not pure LoRA updates and are unsupported."
        )
    try:
        default_rank = int(config["r"])
        default_alpha = float(config["lora_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Adapter config must define numeric r and lora_alpha.") from error
    if default_rank <= 0 or default_alpha < 0:
        raise ValueError("LoRA rank must be positive and lora_alpha non-negative.")

    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    use_rslora = bool(config.get("use_rslora", False))
    grouped_factors = _group_lora_factors(state_dict)
    geometry: dict[str, EffectiveLoraLayer] = {}

    for module_name, factors in grouped_factors.items():
        lora_a = factors["A"]
        lora_b = factors["B"]
        actual_rank = int(lora_a.shape[0])
        if lora_b.shape[1] != actual_rank:
            raise ValueError(
                f"Incompatible factor shapes for {module_name}: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )
        configured_rank = int(
            _resolve_pattern_value(rank_pattern, module_name, default_rank)
        )
        if configured_rank != actual_rank:
            raise ValueError(
                f"Configured rank for {module_name} is {configured_rank}, "
                f"but its saved factors use rank {actual_rank}."
            )
        alpha = float(
            _resolve_pattern_value(alpha_pattern, module_name, default_alpha)
        )
        if alpha < 0 or not math.isfinite(alpha):
            raise ValueError(f"Invalid lora_alpha for {module_name}: {alpha}")
        denominator = math.sqrt(actual_rank) if use_rslora else actual_rank
        geometry[module_name] = EffectiveLoraLayer(
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=alpha / denominator,
        )
    return geometry


def load_effective_lora_geometry(
    adapter_path: str | Path,
) -> dict[str, EffectiveLoraLayer]:
    """Load one adapter as layer-wise effective LoRA update factors."""
    return prepare_effective_lora_geometry(
        load_lora_factor_state_dict(adapter_path),
        load_adapter_config(adapter_path),
    )


def validate_compatible_geometries(
    geometries: Sequence[Mapping[str, EffectiveLoraLayer]],
    adapter_names: Sequence[str],
) -> None:
    """Ensure adapters update the same modules with compatible dimensions."""
    if not geometries:
        raise ValueError("At least one effective LoRA geometry is required.")
    reference = geometries[0]
    reference_modules = sorted(reference)
    for adapter_name, geometry in zip(adapter_names[1:], geometries[1:]):
        if sorted(geometry) != reference_modules:
            raise ValueError(
                f"Adapter {adapter_name!r} updates different LoRA modules."
            )
        for module_name in reference_modules:
            if geometry[module_name].effective_shape != reference[module_name].effective_shape:
                raise ValueError(
                    f"Adapter {adapter_name!r} has an incompatible update shape "
                    f"for {module_name}: {geometry[module_name].effective_shape} != "
                    f"{reference[module_name].effective_shape}"
                )


def effective_lora_inner_product(
    geometry_a: Mapping[str, EffectiveLoraLayer],
    geometry_b: Mapping[str, EffectiveLoraLayer],
) -> float:
    """Compute the global Frobenius inner product of two effective updates.

    Per layer, ``<B_a A_a, B_b A_b>_F`` equals
    ``sum((B_a.T B_b) * (A_a A_b.T))``. The PEFT scaling of both adapters is
    applied and no full ``B @ A`` tensor is materialized.
    """
    validate_compatible_geometries([geometry_a, geometry_b], ["a", "b"])
    total = torch.zeros((), dtype=torch.float64)
    for module_name in sorted(geometry_a):
        layer_a = geometry_a[module_name]
        layer_b = geometry_b[module_name]
        b_gram = layer_a.lora_b.transpose(0, 1) @ layer_b.lora_b
        a_gram = layer_a.lora_a @ layer_b.lora_a.transpose(0, 1)
        total += (
            layer_a.scaling
            * layer_b.scaling
            * torch.sum(b_gram * a_gram)
        )
    return float(total.item())


def effective_lora_update_norm(
    geometry: Mapping[str, EffectiveLoraLayer],
    eps: float = 1e-12,
) -> float:
    """Return the Frobenius norm of the complete effective LoRA update."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    squared_norm = effective_lora_inner_product(geometry, geometry)
    if squared_norm < -eps:
        raise ValueError(f"Effective update has invalid squared norm: {squared_norm}")
    norm = math.sqrt(max(squared_norm, 0.0))
    if norm < eps:
        raise ValueError("Effective LoRA update is near zero; cosine is undefined.")
    return norm


def effective_lora_update_numel(
    geometry: Mapping[str, EffectiveLoraLayer],
) -> int:
    """Return the conceptual number of entries in all effective updates."""
    return sum(math.prod(layer.effective_shape) for layer in geometry.values())
