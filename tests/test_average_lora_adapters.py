from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.effective_lora_geometry import load_effective_lora_geometry


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "average_lora_adapters.py"
SPEC = importlib.util.spec_from_file_location("average_lora_adapters", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_adapter(path: Path, a: torch.Tensor, b: torch.Tensor) -> None:
    path.mkdir(parents=True)
    config = {
        "r": int(a.shape[0]),
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj"],
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "inference_mode": True,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.model").write_bytes(b"test-tokenizer")
    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": a,
            "base_model.model.q_proj.lora_B.weight": b,
        },
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


def test_exact_effective_update_mean_is_saved_as_concatenated_rank(tmp_path: Path) -> None:
    a1 = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    b1 = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    a2 = torch.tensor([[2.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    b2 = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "mean"
    _write_adapter(first, a1, b1)
    _write_adapter(second, a2, b2)

    manifest = MODULE.average_lora_adapters(
        [first, second], output, label="helpfulness", seeds=[137, 138]
    )
    geometry = load_effective_lora_geometry(output)
    layer = geometry["base_model.model.q_proj"]
    actual = layer.scaling * (layer.lora_b @ layer.lora_a)
    expected = ((4 / 2) * (b1 @ a1) + (4 / 2) * (b2 @ a2)) / 2

    assert torch.allclose(actual, expected.to(torch.float64), atol=1e-12)
    assert json.loads((output / "adapter_config.json").read_text())["r"] == 4
    assert (output / "tokenizer.model").read_bytes() == b"test-tokenizer"
    assert manifest["source_count"] == 2
    assert manifest["verification_relative_frobenius_error"] <= 1e-6


def test_existing_output_is_reused_only_for_the_same_binding(tmp_path: Path) -> None:
    a = torch.eye(2)
    b = torch.eye(2)
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "mean"
    _write_adapter(first, a, b)
    _write_adapter(second, 2 * a, b)
    initial = MODULE.average_lora_adapters(
        [first, second], output, label="coherence", seeds=[137, 138]
    )
    repeated = MODULE.average_lora_adapters(
        [first, second], output, label="coherence", seeds=[137, 138]
    )
    assert repeated["binding_sha256"] == initial["binding_sha256"]


def test_target_module_order_does_not_make_configs_incompatible() -> None:
    base = {
        "r": 8,
        "lora_alpha": 16,
        "inference_mode": True,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    }
    reordered = {**base, "target_modules": ["v_proj", "q_proj", "k_proj"]}
    assert MODULE._canonical_adapter_config(base) == MODULE._canonical_adapter_config(reordered)


def test_true_config_difference_is_still_detected() -> None:
    reference = MODULE._canonical_adapter_config({
        "r": 8, "lora_alpha": 16, "inference_mode": True,
        "target_modules": ["q_proj"], "lora_dropout": 0.05,
    })
    changed = MODULE._canonical_adapter_config({
        "r": 8, "lora_alpha": 16, "inference_mode": True,
        "target_modules": ["q_proj"], "lora_dropout": 0.0,
    })
    assert MODULE._different_config_fields(reference, changed) == ["lora_dropout"]
