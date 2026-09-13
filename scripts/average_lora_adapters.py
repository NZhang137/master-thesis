"""Average compatible PEFT LoRA adapters without averaging their factors.

For adapters with rank ``r`` and scaling ``alpha / r``, the arithmetic mean
of ``n`` effective updates is represented exactly by concatenating their A/B
factors, using rank ``n * r`` and retaining the original ``alpha``::

    (alpha / (n r)) [B_1 ... B_n] [A_1; ...; A_n]
      = (1/n) sum_k (alpha / r) B_k A_k.

This avoids the invalid operation of averaging A and B separately. The tool
is intentionally strict and supports the uniform pure-LoRA configuration used
by NB11/NB11.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from src.effective_lora_geometry import (
    effective_lora_inner_product,
    load_adapter_config,
    load_effective_lora_geometry,
    load_lora_factor_state_dict,
    validate_compatible_geometries,
)


_FACTOR_PATTERN = re.compile(
    r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$"
)
_TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

# PEFT stores target_modules internally as a set but serializes it as a JSON
# list.  Its order can therefore differ across otherwise identical training
# runs (and even across the five original NB11 adapters).  Only this field is
# order-insensitive here; all other metadata remains part of the strict
# compatibility check.
_ORDER_INSENSITIVE_CONFIG_FIELDS = ("target_modules",)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _weight_path(adapter_path: Path) -> Path:
    path = adapter_path / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {adapter_path}")
    return path


def _validate_uniform_config(config: dict[str, Any]) -> tuple[int, float]:
    if config.get("use_dora", False):
        raise ValueError("DoRA adapters are unsupported.")
    if config.get("use_rslora", False):
        raise ValueError("Rank-stabilized LoRA adapters are unsupported.")
    if config.get("modules_to_save"):
        raise ValueError("modules_to_save adapters are unsupported.")
    if config.get("rank_pattern"):
        raise ValueError("Per-module rank_pattern is unsupported.")
    if config.get("alpha_pattern"):
        raise ValueError("Per-module alpha_pattern is unsupported.")
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank <= 0 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"Invalid LoRA rank/alpha: r={rank}, alpha={alpha}")
    return rank, alpha


def _canonical_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the strict comparison form of a PEFT LoRA configuration."""
    ignored = {"r", "lora_alpha", "inference_mode"}
    canonical = {key: value for key, value in config.items() if key not in ignored}
    for field in _ORDER_INSENSITIVE_CONFIG_FIELDS:
        value = canonical.get(field)
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field} must contain only strings.")
            canonical[field] = sorted(set(value))
    return canonical


def _different_config_fields(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    keys = set(reference) | set(candidate)
    return sorted(key for key in keys if reference.get(key) != candidate.get(key))


def _group_keys(state: dict[str, torch.Tensor]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for key in state:
        match = _FACTOR_PATTERN.fullmatch(key)
        if match is None:
            continue
        module = match.group("module")
        factor = match.group("factor")
        if factor in grouped.setdefault(module, {}):
            raise ValueError(f"Duplicate LoRA factor {factor} for {module}")
        grouped[module][factor] = key
    if not grouped or any(set(keys) != {"A", "B"} for keys in grouped.values()):
        raise ValueError("Adapter does not contain complete LoRA A/B factor pairs.")
    return grouped


def _mean_relative_error(
    source_geometries: Sequence[dict[str, Any]],
    mean_geometry: dict[str, Any],
) -> float:
    """Return ||mean_adapter - arithmetic_mean|| / ||arithmetic_mean||."""
    n = len(source_geometries)
    mean_norm_sq = 0.0
    for left in source_geometries:
        for right in source_geometries:
            mean_norm_sq += effective_lora_inner_product(left, right) / (n * n)
    candidate_norm_sq = effective_lora_inner_product(mean_geometry, mean_geometry)
    cross = sum(
        effective_lora_inner_product(mean_geometry, source) / n
        for source in source_geometries
    )
    error_sq = candidate_norm_sq + mean_norm_sq - 2.0 * cross
    scale = max(candidate_norm_sq, mean_norm_sq, 1.0)
    if error_sq < 0 and abs(error_sq) <= 1e-12 * scale:
        error_sq = 0.0
    if error_sq < 0:
        raise RuntimeError(f"Invalid negative verification norm: {error_sq}")
    return math.sqrt(error_sq) / max(math.sqrt(mean_norm_sq), 1e-30)


def average_lora_adapters(
    adapter_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    label: str,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Create or verify an exact concatenated-factor mean adapter."""
    inputs = [Path(path).resolve() for path in adapter_paths]
    destination = Path(output_dir).resolve()
    if len(inputs) < 2:
        raise ValueError("At least two source adapters are required.")
    if seeds is not None and len(seeds) != len(inputs):
        raise ValueError("The number of seeds must equal the number of adapters.")

    configs = [load_adapter_config(path) for path in inputs]
    ranks_alphas = [_validate_uniform_config(config) for config in configs]
    if len(set(ranks_alphas)) != 1:
        raise ValueError(f"Source adapters differ in rank/alpha: {ranks_alphas}")
    reference_config = configs[0]
    canonical_configs = [_canonical_adapter_config(config) for config in configs]
    incompatible = [
        (index, _different_config_fields(canonical_configs[0], config))
        for index, config in enumerate(canonical_configs[1:], start=1)
        if config != canonical_configs[0]
    ]
    if incompatible:
        raise ValueError(
            "Source adapter configurations are not compatible; "
            f"differing fields by input index: {incompatible}"
        )

    source_states = [load_lora_factor_state_dict(path) for path in inputs]
    reference_keys = set(source_states[0])
    if any(set(state) != reference_keys for state in source_states[1:]):
        raise ValueError("Source adapters contain different LoRA factor keys.")
    groups = _group_keys(source_states[0])
    for state in source_states[1:]:
        if _group_keys(state) != groups:
            raise ValueError("Source adapters use incompatible LoRA key layouts.")

    rank, alpha = ranks_alphas[0]
    n = len(inputs)
    output_state: dict[str, torch.Tensor] = {}
    for module, keys in groups.items():
        a_tensors = [state[keys["A"]] for state in source_states]
        b_tensors = [state[keys["B"]] for state in source_states]
        if len({tuple(tensor.shape) for tensor in a_tensors}) != 1:
            raise ValueError(f"Incompatible A shapes for {module}")
        if len({tuple(tensor.shape) for tensor in b_tensors}) != 1:
            raise ValueError(f"Incompatible B shapes for {module}")
        if any(tensor.dtype != a_tensors[0].dtype for tensor in a_tensors):
            raise ValueError(f"Incompatible A dtypes for {module}")
        if any(tensor.dtype != b_tensors[0].dtype for tensor in b_tensors):
            raise ValueError(f"Incompatible B dtypes for {module}")
        output_state[keys["A"]] = torch.cat(a_tensors, dim=0).contiguous()
        output_state[keys["B"]] = torch.cat(b_tensors, dim=1).contiguous()

    input_records = [
        {
            "path": str(path),
            "seed": int(seeds[index]) if seeds is not None else None,
            "adapter_model_sha256": sha256_file(_weight_path(path)),
            "adapter_config_sha256": sha256_file(path / "adapter_config.json"),
        }
        for index, path in enumerate(inputs)
    ]
    binding = {
        "schema_version": 1,
        "operation": "exact_arithmetic_mean_of_effective_lora_updates",
        "label": label,
        "source_count": n,
        "source_rank": rank,
        "source_lora_alpha": alpha,
        "output_rank": n * rank,
        "output_lora_alpha": alpha,
        "source_adapters": input_records,
    }
    binding_sha256 = canonical_hash(binding)
    manifest_path = destination / "averaging_manifest.json"
    if destination.exists():
        if not manifest_path.is_file() or not _weight_path(destination).is_file():
            raise RuntimeError(
                f"Existing output is incomplete: {destination}. Use a new output path."
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("binding_sha256") != binding_sha256:
            raise RuntimeError(
                f"Existing mean adapter has a different binding: {destination}"
            )
        if existing.get("adapter_model_sha256") != sha256_file(_weight_path(destination)):
            raise RuntimeError(f"Existing mean adapter hash mismatch: {destination}")
        print(f"[skip] verified existing mean adapter: {destination}")
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        output_config = dict(reference_config)
        output_config["r"] = n * rank
        output_config["lora_alpha"] = alpha
        output_config["inference_mode"] = True
        output_config["rank_pattern"] = {}
        output_config["alpha_pattern"] = {}
        if isinstance(output_config.get("target_modules"), list):
            output_config["target_modules"] = sorted(set(output_config["target_modules"]))
        (temporary / "adapter_config.json").write_text(
            json.dumps(output_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_file(
            output_state,
            str(temporary / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        for filename in _TOKENIZER_FILES:
            source = inputs[0] / filename
            if source.is_file():
                shutil.copy2(source, temporary / filename)

        source_geometries = [load_effective_lora_geometry(path) for path in inputs]
        validate_compatible_geometries(source_geometries, [str(path) for path in inputs])
        mean_geometry = load_effective_lora_geometry(temporary)
        validate_compatible_geometries(
            [source_geometries[0], mean_geometry], [str(inputs[0]), "mean"]
        )
        relative_error = _mean_relative_error(source_geometries, mean_geometry)
        if relative_error > 1e-6:
            raise RuntimeError(
                f"Mean adapter verification failed: relative error={relative_error}"
            )
        manifest = {
            **binding,
            "binding_sha256": binding_sha256,
            "verification_relative_frobenius_error": relative_error,
            "adapter_config_sha256": sha256_file(temporary / "adapter_config.json"),
            "adapter_model_sha256": sha256_file(
                temporary / "adapter_model.safetensors"
            ),
        }
        (temporary / "averaging_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"[OK] exact mean adapter {label}: rank {rank} x {n} = {n * rank}; "
        f"relative error={relative_error:.3e}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", action="append", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    average_lora_adapters(
        args.adapter,
        args.output_dir,
        label=args.label,
        seeds=args.seed,
    )


if __name__ == "__main__":
    main()
