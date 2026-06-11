"""Compute first-prototype M1 coefficients from a saved relationship matrix."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_utils import (
    l1_distance,
    l2_distance,
    load_labeled_relationship_matrix,
    normalize_simplex,
    relationship_softmax_mapping,
)


OBJECTIVE_NAMES = ["helpful", "harmless"]
EXAMPLE_PREFERENCES = (
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([0.8, 0.2], dtype=np.float64),
    np.array([0.2, 0.8], dtype=np.float64),
)
TAU_VALUES = (0.0, 0.5, 1.0, 2.0)


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_rows(R: np.ndarray) -> list[dict[str, float | str]]:
    """Compute M1 coefficients, relationship scores, and correction distances."""
    rows = []

    for preference in EXAMPLE_PREFERENCES:
        p = normalize_simplex(preference)
        scores = R @ p

        for tau in TAU_VALUES:
            lambdas = relationship_softmax_mapping(p, R, tau=tau)
            rows.append(
                {
                    "method": "M1_relationship_softmax",
                    "tau": tau,
                    "p_helpful": p[0],
                    "p_harmless": p[1],
                    "lambda_helpful": lambdas[0],
                    "lambda_harmless": lambdas[1],
                    "score_helpful": scores[0],
                    "score_harmless": scores[1],
                    "l1_distance_to_p": l1_distance(lambdas, p),
                    "l2_distance_to_p": l2_distance(lambdas, p),
                }
            )

    return rows


def write_rows(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write the compact M1 coefficient table."""
    fieldnames = [
        "method",
        "tau",
        "p_helpful",
        "p_harmless",
        "lambda_helpful",
        "lambda_harmless",
        "score_helpful",
        "score_harmless",
        "l1_distance_to_p",
        "l2_distance_to_p",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value if isinstance(value, str) else f"{value:.8f}"
                    for key, value in row.items()
                }
            )


def print_summary(rows: list[dict[str, float | str]]) -> None:
    """Print a readable overview of preference corrections across tau."""
    print("\nM1 relationship-softmax coefficients:")
    for row in rows:
        print(
            "  "
            f"p=[{float(row['p_helpful']):.2f}, "
            f"{float(row['p_harmless']):.2f}], "
            f"tau={float(row['tau']):.1f} -> "
            f"lambda=[{float(row['lambda_helpful']):.4f}, "
            f"{float(row['lambda_harmless']):.4f}], "
            f"L1={float(row['l1_distance_to_p']):.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse relationship-matrix input and coefficient output paths."""
    parser = argparse.ArgumentParser(
        description="Compute M1 relationship-softmax merge coefficients."
    )
    parser.add_argument(
        "--relationship_matrix_path",
        "--relationship-matrix-path",
        dest="relationship_matrix_path",
        default="results/relationship_matrix.csv",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        dest="output_path",
        default="results/m1_coefficients.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Load R, compute all example M1 coefficients, and save the table."""
    args = parse_args()
    matrix_path = resolve_project_path(args.relationship_matrix_path)
    output_path = resolve_project_path(args.output_path)

    try:
        R = load_labeled_relationship_matrix(matrix_path, OBJECTIVE_NAMES)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}. Run scripts/compute_relationship_matrix.py first."
        ) from error
    rows = compute_rows(R)
    write_rows(output_path, rows)

    print(f"Loaded relationship matrix from {matrix_path}")
    print_summary(rows)
    print(f"\nSaved {len(rows)} M1 coefficient rows to {output_path}")


if __name__ == "__main__":
    main()
