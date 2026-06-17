"""Compute a cosine relationship matrix from local LoRA adapter weights."""

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
    flatten_adapter_state_dict,
    load_adapter_state_dict,
)


ADAPTER_NAMES = ["helpful", "harmless"]


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_matrix_csv(
    output_path: Path,
    adapter_names: list[str],
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
        description="Compute cosine relationships between two LoRA adapters."
    )
    parser.add_argument(
        "--helpful_adapter_path",
        "--helpful-adapter-path",
        dest="helpful_adapter_path",
        default="adapters/gpt2-helpful-adapter",
    )
    parser.add_argument(
        "--harmless_adapter_path",
        "--harmless-adapter-path",
        dest="harmless_adapter_path",
        default="adapters/gpt2-harmless-adapter",
    )
    parser.add_argument(
        "--output_csv",
        "--output-csv",
        dest="output_csv",
        default="results/relationship_matrix.csv",
    )
    parser.add_argument(
        "--output_metadata",
        "--output-metadata",
        dest="output_metadata",
        default="results/relationship_matrix_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Compute R, save the labeled matrix, and record compact metadata."""
    args = parse_args()
    adapter_paths = [
        resolve_project_path(args.helpful_adapter_path),
        resolve_project_path(args.harmless_adapter_path),
    ]
    output_csv = resolve_project_path(args.output_csv)
    output_metadata = resolve_project_path(args.output_metadata)

    print("Computing cosine relationships from LoRA adapter parameters...")
    for name, path in zip(ADAPTER_NAMES, adapter_paths):
        print(f"  {name}: {path}")

    matrix = compute_relationship_matrix(
        adapter_paths=adapter_paths,
        adapter_names=ADAPTER_NAMES,
    )
    matrix_values = matrix.tolist()
    write_matrix_csv(output_csv, ADAPTER_NAMES, matrix_values)

    vector_lengths = [
        flatten_adapter_state_dict(load_adapter_state_dict(path)).numel()
        for path in adapter_paths
    ]
    metadata = {
        "adapter_names": ADAPTER_NAMES,
        "adapter_paths": [
            args.helpful_adapter_path,
            args.harmless_adapter_path,
        ],
        "similarity_type": "cosine",
        "representation": "flattened LoRA adapter parameters",
        "vector_lengths": vector_lengths,
        "note": (
            "This static matrix is computed from flattened saved LoRA adapter "
            "parameters and is a proxy for objective or specialist relationships."
        ),
        "caveat": (
            "The geometry proxy must be empirically validated and does not yet "
            "implement the final lambda = f(p, R) method."
        ),
    }
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with output_metadata.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    print("\nRelationship matrix:")
    for name, row in zip(ADAPTER_NAMES, matrix_values):
        formatted = ", ".join(f"{value:.4f}" for value in row)
        print(f"  {name}: [{formatted}]")
    print(f"\nSaved matrix to {output_csv}")
    print(f"Saved metadata to {output_metadata}")


if __name__ == "__main__":
    main()
