"""Compute HelpSteer2 M1, M2, C1, C2, P1, and P2 coefficients."""

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
    c2_softmin_cagrad_mapping,
    direct_preference_mapping,
    l1_distance,
    l2_distance,
    m1_mgda_inspired_mapping,
    m2_preference_weighted_alpha_mgda_mapping,
    p1_pcgrad_reconstruction_mapping,
    p2_pcgrad_reconstruction_reverse_mapping,
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

METHOD_NAMES = ("M1", "M2", "C1", "C2", "P1", "P2")

METHOD_DEFAULTS = {
    "M1": {"rho": 1.0},
    "M2": {"rho": 1.0},
    "C1": {"c": 0.5, "eps": 1e-8},
    "C2": {"tau": 0.1, "rho": 1.0},
    "P1": {"rho": 1.0, "eps": 1e-8},
    "P2": {"rho": 1.0, "eps": 1e-8},
}

CSV_COLUMNS = [
    "preference_name",
    "method",
    "hyperparameters",
    *(f"p_{name}" for name in OBJECTIVE_NAMES),
    *(f"lambda_{name}" for name in OBJECTIVE_NAMES),
    *(f"score_{name}" for name in OBJECTIVE_NAMES),
    "min_relationship_score",
    "lambda_sum",
    "lambda_min",
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

    matrix = np.array(
        [[float(row[column]) for column in OBJECTIVE_NAMES] for row in rows],
        dtype=np.float64,
    )
    return validate_relationship_matrix(
        matrix,
        num_objectives=len(OBJECTIVE_NAMES),
    )


def compute_method_lambda(
    method: str,
    preference: np.ndarray,
    relationships: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute one method's lambda vector and return its hyperparameters."""
    defaults = METHOD_DEFAULTS[method]
    if method == "M1":
        lambdas = m1_mgda_inspired_mapping(
            preference,
            relationships,
            rho=defaults["rho"],
        )
    elif method == "M2":
        lambdas = m2_preference_weighted_alpha_mgda_mapping(
            preference,
            relationships,
            rho=defaults["rho"],
        )
    elif method == "C1":
        lambdas = c1_trust_region_cagrad_mapping(
            preference,
            relationships,
            c=defaults["c"],
            eps=defaults["eps"],
        )
    elif method == "C2":
        lambdas = c2_softmin_cagrad_mapping(
            preference,
            relationships,
            tau=defaults["tau"],
            rho=defaults["rho"],
        )
    elif method == "P1":
        lambdas = p1_pcgrad_reconstruction_mapping(
            preference,
            relationships,
            rho=defaults["rho"],
            eps=defaults["eps"],
        )
    elif method == "P2":
        lambdas = p2_pcgrad_reconstruction_reverse_mapping(
            preference,
            relationships,
            rho=defaults["rho"],
            eps=defaults["eps"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return lambdas, defaults


def validate_lambda(
    *,
    lambdas: np.ndarray,
    preference: np.ndarray,
    method: str,
    preference_name: str,
    atol: float = 1e-7,
) -> None:
    """Validate that one lambda vector is on the simplex."""
    if lambdas.shape != preference.shape:
        raise ValueError(
            f"{method}/{preference_name} lambda shape {lambdas.shape} "
            f"does not match preference shape {preference.shape}."
        )
    if not np.all(np.isfinite(lambdas)):
        raise ValueError(f"{method}/{preference_name} lambda has non-finite values.")
    if np.any(lambdas < -atol):
        raise ValueError(f"{method}/{preference_name} lambda has negative values.")
    if abs(float(lambdas.sum()) - 1.0) > atol:
        raise ValueError(
            f"{method}/{preference_name} lambda sums to "
            f"{float(lambdas.sum()):.12f}, not 1."
        )


def build_result_row(
    preference_name: str,
    method: str,
    hyperparameters: dict[str, float],
    preference: np.ndarray,
    lambdas: np.ndarray,
    relationships: np.ndarray,
) -> dict[str, float | str]:
    """Build one coefficient result row."""
    scores = relationships @ lambdas
    row: dict[str, float | str] = {
        "preference_name": preference_name,
        "method": method,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "min_relationship_score": float(np.min(scores)),
        "lambda_sum": float(lambdas.sum()),
        "lambda_min": float(np.min(lambdas)),
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
    """Compute all requested method/preference coefficient rows."""
    rows: list[dict[str, float | str]] = []
    for preference_name, preference_values in PREFERENCES.items():
        preference = validate_preference_vector(preference_values)

        direct = direct_preference_mapping(preference)
        validate_lambda(
            lambdas=direct,
            preference=preference,
            method="direct_preference",
            preference_name=preference_name,
        )

        for method in METHOD_NAMES:
            lambdas, hyperparameters = compute_method_lambda(
                method,
                preference,
                relationships,
            )
            validate_lambda(
                lambdas=lambdas,
                preference=preference,
                method=method,
                preference_name=preference_name,
            )
            rows.append(
                build_result_row(
                    preference_name,
                    method,
                    hyperparameters,
                    preference,
                    lambdas,
                    relationships,
                )
            )

    validate_coverage(rows)
    return rows


def validate_coverage(rows: list[dict[str, float | str]]) -> None:
    """Ensure every method/preference pair is present exactly once."""
    expected = {
        (preference_name, method)
        for preference_name in PREFERENCES
        for method in METHOD_NAMES
    }
    observed = {
        (str(row["preference_name"]), str(row["method"])) for row in rows
    }
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            f"Coefficient coverage mismatch. Missing={missing}, extra={extra}."
        )


def write_results(output_path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write the coefficient rows to CSV."""
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
    relationship_matrix_path: str,
    output_csv_path: str,
    relationships: np.ndarray,
    rows: list[dict[str, float | str]],
) -> None:
    """Write compact coefficient metadata to JSON."""
    metadata = {
        "relationship_matrix_path": relationship_matrix_path,
        "output_csv_path": output_csv_path,
        "objective_order": list(OBJECTIVE_NAMES),
        "preferences": {
            name: list(values) for name, values in PREFERENCES.items()
        },
        "methods": {
            "M1": {
                "definition": (
                    "argmin over simplex lambda^T R lambda "
                    "+ rho ||lambda - p||_2^2"
                ),
                "hyperparameters": METHOD_DEFAULTS["M1"],
            },
            "M2": {
                "definition": (
                    "alpha-MGDA with G_p=Diag(p) R Diag(p), then "
                    "lambda=(alpha*p)/sum(alpha*p)"
                ),
                "hyperparameters": METHOD_DEFAULTS["M2"],
            },
            "P1": {
                "definition": (
                    "R-metric PCGrad projection with strongest negative "
                    "conflict ordering and simplex reconstruction"
                ),
                "hyperparameters": METHOD_DEFAULTS["P1"],
            },
            "P2": {
                "definition": (
                    "Same R-metric PCGrad equations as P1 with reverse "
                    "deterministic conflict ordering"
                ),
                "hyperparameters": METHOD_DEFAULTS["P2"],
            },
            "C1": {
                "definition": (
                    "trust-region CAGrad: maximize min_i (R lambda)_i "
                    "subject to simplex and R-trust-region constraint"
                ),
                "hyperparameters": METHOD_DEFAULTS["C1"],
            },
            "C2": {
                "definition": (
                    "soft-min CAGrad: maximize softmin_tau(R lambda) "
                    "- rho (lambda-p)^T R (lambda-p)"
                ),
                "hyperparameters": METHOD_DEFAULTS["C2"],
            },
        },
        "row_count": len(rows),
        "relationship_matrix_eigenvalues": np.linalg.eigvalsh(
            relationships
        ).tolist(),
        "validation": {
            "lambda_dimension_checked": True,
            "lambda_non_negative_checked": True,
            "lambda_sum_checked": True,
            "method_preference_coverage_checked": True,
            "tolerance": 1e-7,
        },
        "note": (
            "The coefficients are one-shot mappings from p and R inside the "
            "fixed Rewarded-Soups-style interpolation family."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def print_summary(rows: list[dict[str, float | str]]) -> None:
    """Print a compact summary table for the terminal."""
    print("\nHelpSteer2 all-method coefficient summary:")
    print(
        "preference".ljust(22)
        + "method".ljust(8)
        + "lambda".ljust(48)
        + "min_score"
    )
    for row in rows:
        lambdas = [
            float(row[f"lambda_{name}"]) for name in OBJECTIVE_NAMES
        ]
        lambda_text = "[" + ", ".join(f"{value:.3f}" for value in lambdas) + "]"
        print(
            str(row["preference_name"]).ljust(22)
            + str(row["method"]).ljust(8)
            + lambda_text.ljust(48)
            + f"{float(row['min_relationship_score']):.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Compute HelpSteer2 M1, M2, C1, C2, P1, and P2 coefficients."
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
        default="results/helpsteer2_all_method_coefficients.csv",
    )
    parser.add_argument(
        "--output_metadata",
        "--output-metadata",
        dest="output_metadata",
        default="results/helpsteer2_all_method_coefficients_metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    """Load R, compute coefficients, validate them, and write outputs."""
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
        rows,
    )
    print(f"Loaded relationship matrix from {matrix_path}")
    print_summary(rows)
    print(f"\nSaved {len(rows)} coefficient rows to {output_csv}")
    print(f"Saved metadata to {output_metadata}")


if __name__ == "__main__":
    main()
