"""Compute HelpSteer2 direct-preference, M1, and C1 coefficients."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_methods import (
    c1_trust_region_cagrad_mapping,
    direct_preference_mapping,
    l1_distance,
    l2_distance,
    m1_mgda_inspired_mapping,
    validate_preference_vector,
    validate_relationship_matrix,
)


OBJECTIVE_NAMES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

PREFERENCES = {
    "balanced": (0.2, 0.2, 0.2, 0.2, 0.2),
    "quality_focused": (0.15, 0.35, 0.25, 0.10, 0.15),
    "detailed_answer": (0.15, 0.15, 0.20, 0.25, 0.25),
    "helpfulness_focused": (0.50, 0.15, 0.15, 0.10, 0.10),
}

M1_RHO_VALUES = (1.0,)
C1_C_VALUES = (0.5,)

CSV_COLUMNS = [
    "preference_name",
    "method",
    "hyperparameter_name",
    "hyperparameter_value",
    *(f"p_{name}" for name in OBJECTIVE_NAMES),
    *(f"lambda_{name}" for name in OBJECTIVE_NAMES),
    *(f"score_{name}" for name in OBJECTIVE_NAMES),
    "min_relationship_score",
    "l1_distance_to_p",
    "l2_distance_to_p",
]


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_relationship_matrix(matrix_path: Path) -> np.ndarray:
    """Load the labeled HelpSteer2 relationship matrix in objective order."""
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Relationship matrix not found: {matrix_path}. "
            "Run scripts/compute_helpsteer2_relationship_matrix.py first."
        )

    with matrix_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        expected_columns = ["adapter", *OBJECTIVE_NAMES]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                "Relationship matrix columns must be exactly: "
                + ", ".join(expected_columns)
            )
        rows = list(reader)

    row_names = [row["adapter"] for row in rows]
    if row_names != list(OBJECTIVE_NAMES):
        raise ValueError(
            "Relationship matrix rows must be ordered as: "
            + ", ".join(OBJECTIVE_NAMES)
        )

    try:
        matrix = np.array(
            [
                [float(row[column]) for column in OBJECTIVE_NAMES]
                for row in rows
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Relationship matrix entries must contain numeric values."
        ) from error

    return validate_relationship_matrix(
        matrix,
        num_objectives=len(OBJECTIVE_NAMES),
    )


def build_result_row(
    preference_name: str,
    method: str,
    hyperparameter_name: str,
    hyperparameter_value: float | str,
    preference: np.ndarray,
    lambdas: np.ndarray,
    relationships: np.ndarray,
) -> dict[str, float | str]:
    """Build one coefficient result row with scores and distances."""
    scores = relationships @ lambdas
    row: dict[str, float | str] = {
        "preference_name": preference_name,
        "method": method,
        "hyperparameter_name": hyperparameter_name,
        "hyperparameter_value": hyperparameter_value,
        "min_relationship_score": float(np.min(scores)),
        "l1_distance_to_p": l1_distance(lambdas, preference),
        "l2_distance_to_p": l2_distance(lambdas, preference),
    }
    row.update(
        {
            f"p_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, preference)
        }
    )
    row.update(
        {
            f"lambda_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, lambdas)
        }
    )
    row.update(
        {
            f"score_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, scores)
        }
    )
    return row


def compute_rows(relationships: np.ndarray) -> list[dict[str, float | str]]:
    """Compute direct-preference, M1, and C1 rows for all preferences."""
    rows: list[dict[str, float | str]] = []

    for preference_name, preference_values in PREFERENCES.items():
        preference = validate_preference_vector(preference_values)

        direct = direct_preference_mapping(preference)
        rows.append(
            build_result_row(
                preference_name,
                "direct_preference",
                "",
                "",
                preference,
                direct,
                relationships,
            )
        )

        for rho in M1_RHO_VALUES:
            lambdas = m1_mgda_inspired_mapping(
                preference,
                relationships,
                rho=rho,
            )
            rows.append(
                build_result_row(
                    preference_name,
                    "M1",
                    "rho",
                    rho,
                    preference,
                    lambdas,
                    relationships,
                )
            )

        for c in C1_C_VALUES:
            lambdas = c1_trust_region_cagrad_mapping(
                preference,
                relationships,
                c=c,
            )
            rows.append(
                build_result_row(
                    preference_name,
                    "C1",
                    "c",
                    c,
                    preference,
                    lambdas,
                    relationships,
                )
            )

    return rows


def write_results(
    output_path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    """Write the coefficient table to a small UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        value
                        if isinstance(value, str)
                        else f"{float(value):.10f}"
                    )
                    for column, value in row.items()
                }
            )


def write_metadata(
    output_path: Path,
    matrix_path_argument: str,
    output_csv_argument: str,
    relationships: np.ndarray,
    row_count: int,
) -> None:
    """Write compact method and experiment metadata as JSON."""
    metadata = {
        "objective_order": list(OBJECTIVE_NAMES),
        "relationship_matrix_path": matrix_path_argument,
        "output_csv_path": output_csv_argument,
        "number_of_objectives": len(OBJECTIVE_NAMES),
        "number_of_rows": row_count,
        "preferences": {
            name: list(values) for name, values in PREFERENCES.items()
        },
        "methods": {
            "direct_preference": {
                "description": "Normalized baseline lambda = p."
            },
            "M1": {
                "description": (
                    "MGDA-inspired one-shot mapping: minimize "
                    "lambda^T R lambda + rho ||lambda - p||_2^2."
                ),
                "hyperparameter": "rho",
                "values": list(M1_RHO_VALUES),
            },
            "C1": {
                "description": (
                    "Trust-region CAGrad-inspired mapping: maximize "
                    "min_i (R lambda)_i under the simplex and R-trust-region "
                    "constraint."
                ),
                "hyperparameter": "c",
                "values": list(C1_C_VALUES),
                "optimizer": "scipy.optimize.minimize with SLSQP",
                "initialization": "normalized preference vector p",
                "fallback": (
                    "Choose the best feasible candidate among p, uniform, "
                    "and simplex vertices."
                ),
            },
        },
        "relationship_matrix_symmetrized": True,
        "relationship_matrix_eigenvalues": np.linalg.eigvalsh(
            relationships
        ).tolist(),
        "note": (
            "These methods select coefficients inside the fixed "
            "Rewarded-Soups-style interpolation family."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def print_summary(rows: list[dict[str, float | str]]) -> None:
    """Print a concise coefficient and worst-score summary."""
    print("\nHelpSteer2 coefficient mappings:")
    for row in rows:
        hyperparameter = ""
        if row["hyperparameter_name"]:
            hyperparameter = (
                f", {row['hyperparameter_name']}="
                f"{float(row['hyperparameter_value']):g}"
            )
        lambdas = [
            float(row[f"lambda_{name}"]) for name in OBJECTIVE_NAMES
        ]
        lambda_text = ", ".join(f"{value:.4f}" for value in lambdas)
        print(
            f"  {row['preference_name']} | {row['method']}"
            f"{hyperparameter} -> lambda=[{lambda_text}], "
            f"min_score={float(row['min_relationship_score']):.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse relationship-matrix input and result output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute HelpSteer2 direct-preference, thesis M1, and thesis C1 coefficients."
        )
    )
    parser.add_argument(
        "--relationship_matrix_path",
        "--relationship-matrix-path",
        dest="relationship_matrix_path",
        default="results/helpsteer2_relationship_matrix.csv",
    )
    parser.add_argument(
        "--output_csv",
        "--output-csv",
        dest="output_csv",
        default="results/helpsteer2_m1_c1_coefficients.csv",
    )
    parser.add_argument(
        "--output_metadata",
        "--output-metadata",
        dest="output_metadata",
        default="results/helpsteer2_m1_c1_coefficients_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Load R, compute all mappings, and write CSV and JSON outputs."""
    args = parse_args()
    matrix_path = resolve_project_path(args.relationship_matrix_path)
    output_csv = resolve_project_path(args.output_csv)
    output_metadata = resolve_project_path(args.output_metadata)

    relationships = load_relationship_matrix(matrix_path)
    rows = compute_rows(relationships)
    write_results(output_csv, rows)
    write_metadata(
        output_metadata,
        args.relationship_matrix_path,
        args.output_csv,
        relationships,
        len(rows),
    )

    print(f"Loaded HelpSteer2 relationship matrix from {matrix_path}")
    print_summary(rows)
    print(f"\nSaved {len(rows)} coefficient rows to {output_csv}")
    print(f"Saved metadata to {output_metadata}")


if __name__ == "__main__":
    main()
