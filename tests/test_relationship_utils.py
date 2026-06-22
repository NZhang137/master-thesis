"""Tests for effective LoRA update geometry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.effective_lora_geometry import (
    effective_lora_inner_product,
    prepare_effective_lora_geometry,
)
from src.relationship_utils import compute_relationship_matrix


def make_state(prefix: str, lora_a: torch.Tensor, lora_b: torch.Tensor):
    """Build a minimal PEFT-style LoRA state dictionary."""
    return {
        f"{prefix}.lora_A.weight": lora_a,
        f"{prefix}.lora_B.weight": lora_b,
    }


class EffectiveLoraGeometryTests(unittest.TestCase):
    """Verify low-rank inner products and factor gauge invariance."""

    def setUp(self) -> None:
        self.config = {
            "r": 2,
            "lora_alpha": 4,
            "use_rslora": False,
            "use_dora": False,
            "modules_to_save": None,
        }

    def test_inner_product_matches_materialized_effective_updates(self) -> None:
        prefix = "base_model.model.layers.0.self_attn.q_proj"
        a_1 = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])
        b_1 = torch.tensor([[1.0, 0.5], [-1.0, 2.0], [0.0, 1.0]])
        a_2 = torch.tensor([[2.0, -1.0, 0.5], [1.5, 0.0, -2.0]])
        b_2 = torch.tensor([[0.5, 1.0], [2.0, -1.0], [1.0, 0.25]])

        geometry_1 = prepare_effective_lora_geometry(
            make_state(prefix, a_1, b_1), self.config
        )
        geometry_2 = prepare_effective_lora_geometry(
            make_state(prefix, a_2, b_2), self.config
        )
        scale = self.config["lora_alpha"] / self.config["r"]
        expected = torch.sum((scale * (b_1 @ a_1)) * (scale * (b_2 @ a_2))).item()

        actual = effective_lora_inner_product(geometry_1, geometry_2)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_relationship_is_invariant_to_factor_reparameterization(self) -> None:
        prefix = "base_model.model.layers.0.mlp.up_proj"
        lora_a = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])
        lora_b = torch.tensor([[1.0, 0.5], [-1.0, 2.0], [0.0, 1.0]])
        gauge = torch.tensor([[2.0, 0.5], [0.0, 1.5]])
        transformed_a = torch.linalg.inv(gauge) @ lora_a
        transformed_b = lora_b @ gauge

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter_paths = []
            for name, state in (
                ("original", make_state(prefix, lora_a, lora_b)),
                ("transformed", make_state(prefix, transformed_a, transformed_b)),
            ):
                adapter_path = root / name
                adapter_path.mkdir()
                with (adapter_path / "adapter_config.json").open(
                    "w", encoding="utf-8"
                ) as config_file:
                    json.dump(self.config, config_file)
                save_file(state, str(adapter_path / "adapter_model.safetensors"))
                adapter_paths.append(adapter_path)

            matrix = compute_relationship_matrix(
                adapter_paths, adapter_names=["original", "transformed"]
            )

        self.assertTrue(torch.allclose(matrix, torch.ones_like(matrix), atol=1e-10))


if __name__ == "__main__":
    unittest.main()
