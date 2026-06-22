"""Compute a cosine relationship matrix for TinyLlama HelpSteer2 adapters.

The matrix uses effective PEFT-scaled LoRA updates rather than raw A/B
factors. It remains a prototype proxy for task relationships.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.relationship_utils import (
    compute_relationship_matrix,
)
from src.effective_lora_geometry import (
    effective_lora_update_norm,
    effective_lora_update_numel,
    load_effective_lora_geometry,
)


ADAPTER_NAMES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_matrix_csv(
    output_path: Path,
    adapter_names: tuple[str, ...],
    matrix_values: list[list[float]],
) -> None:
    """Write a labeled relationship matrix to a small CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["adapter", *adapter_names])
        for name, row in zip(adapter_names, matrix_values):
            writer.writerow([name, *[f"{value:.8f}" for value in row]])


def parse_args() -> argparse.Namespace:
    """Parse adapter and output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine relationships between five TinyLlama HelpSteer2 "
            "LoRA adapters."
        )
    )
    for adapter_name in ADAPTER_NAMES:
        parser.add_argument(
            f"--{adapter_name}_adapter_path",
            f"--{adapter_name}-adapter-path",
            dest=f"{adapter_name}_adapter_path",
            default=f"adapters/tinyllama-helpsteer2-{adapter_name}-adapter",
        )
    parser.add_argument(
        "--output_csv",
        "--output-csv",
        dest="output_csv",
        default="results/tinyllama_helpsteer2_relationship_matrix.csv",
    )
    parser.add_argument(
        "--output_metadata",
        "--output-metadata",
        dest="output_metadata",
        default="results/tinyllama_helpsteer2_relationship_matrix_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Compute R, save the labeled matrix, and write compact metadata."""
    args = parse_args()
    configured_paths = [
        getattr(args, f"{adapter_name}_adapter_path")
        for adapter_name in ADAPTER_NAMES
    ]
    adapter_paths = [
        resolve_project_path(path_value) for path_value in configured_paths
    ]
    output_csv = resolve_project_path(args.output_csv)
    output_metadata = resolve_project_path(args.output_metadata)

    print(
        "Computing TinyLlama HelpSteer2 cosine relationships from "
        "effective LoRA updates..."
    )
    for adapter_name, adapter_path in zip(ADAPTER_NAMES, adapter_paths):
        print(f"  {adapter_name}: {adapter_path}")

    geometries = [load_effective_lora_geometry(path) for path in adapter_paths]
    layer_counts = [len(geometry) for geometry in geometries]
    effective_update_dimensions = [
        effective_lora_update_numel(geometry) for geometry in geometries
    ]
    effective_update_norms = [
        effective_lora_update_norm(geometry) for geometry in geometries
    ]

    matrix = compute_relationship_matrix(
        adapter_paths=adapter_paths,
        adapter_names=ADAPTER_NAMES,
    )
    matrix_values = matrix.tolist()
    write_matrix_csv(output_csv, ADAPTER_NAMES, matrix_values)

    metadata = {
        "adapter_names": list(ADAPTER_NAMES),
        "adapter_paths": configured_paths,
        "similarity_type": "cosine_similarity_effective_lora_update",
        "number_of_adapters": len(ADAPTER_NAMES),
        "output_csv_path": args.output_csv,
        "representation": "effective LoRA updates: delta_W = scaling * (B @ A)",
        "inner_product_computation": (
            "layer-wise low-rank Frobenius identity without materializing delta_W"
        ),
        "layer_counts": layer_counts,
        "effective_update_dimensions": effective_update_dimensions,
        "effective_update_frobenius_norms": effective_update_norms,
        "note": (
            "R uses the effective updates combined by weighted LoRA merging "
            "and is invariant to factor reparameterizations preserving B @ A."
        ),
        "caveat": (
            "Effective-update cosine remains a prototype proxy for functional "
            "objective relationships and requires empirical validation."
        ),
    }
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with output_metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    print("\nTinyLlama HelpSteer2 relationship matrix:")
    for adapter_name, row in zip(ADAPTER_NAMES, matrix_values):
        formatted_row = ", ".join(f"{value:.4f}" for value in row)
        print(f"  {adapter_name}: [{formatted_row}]")
    print(f"\nSaved matrix to {output_csv}")
    print(f"Saved metadata to {output_metadata}")


if __name__ == "__main__":
    main()
