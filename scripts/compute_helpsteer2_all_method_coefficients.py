"""Compute TinyLlama HelpSteer2 coefficient candidates and cost logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    p1_conflict_weighted_shrinkage_mapping,
    p2_pcgrad_reconstruction_mapping,
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
    "dominant_helpfulness": (0.5, 0.125, 0.125, 0.125, 0.125),
    "dominant_correctness": (0.125, 0.5, 0.125, 0.125, 0.125),
    "dominant_coherence": (0.125, 0.125, 0.5, 0.125, 0.125),
    "dominant_complexity": (0.125, 0.125, 0.125, 0.5, 0.125),
    "dominant_verbosity": (0.125, 0.125, 0.125, 0.125, 0.5),
    "only_helpfulness": (1.0, 0.0, 0.0, 0.0, 0.0),
    "only_correctness": (0.0, 1.0, 0.0, 0.0, 0.0),
    "only_coherence": (0.0, 0.0, 1.0, 0.0, 0.0),
    "only_complexity": (0.0, 0.0, 0.0, 1.0, 0.0),
    "only_verbosity": (0.0, 0.0, 0.0, 0.0, 1.0),
}

VALIDATION_TOLERANCE = 1e-7

BASELINE_GRIDS: dict[str, list[dict[str, float]]] = {
    "direct_preference": [{}],
    "uniform": [{}],
}

METHOD_GRIDS: dict[str, list[dict[str, float]]] = {
    "M1": [{"rho": value} for value in (0.1, 1.0, 10.0)],
    "M2": [{"rho": value} for value in (0.1, 1.0, 10.0)],
    "C1": [
        {"c": value, "eps": 1e-8}
        for value in (0.25, 0.5, 1.0)
    ],
    "C2": [
        {"tau": tau, "rho": rho}
        for tau in (0.05, 0.1, 0.2)
        for rho in (0.1, 1.0)
    ],
    "P1": [{"beta": value} for value in (0.5, 1.0, 2.0)],
    "P2": [
        {"rho": value, "eps": 1e-8}
        for value in (0.1, 1.0, 10.0)
    ],
}

METHOD_FAMILIES = {
    "direct_preference": "baseline",
    "uniform": "baseline",
    "M1": "MGDA-inspired",
    "M2": "MGDA-inspired",
    "P1": "PCGrad-inspired",
    "P2": "PCGrad-inspired",
    "C1": "CAGrad-inspired",
    "C2": "CAGrad-inspired",
}

COEFFICIENT_COLUMNS = [
    "preference_name",
    "method",
    "method_family",
    "hyperparameter_id",
    "hyperparameters_json",
    *(f"p_{name}" for name in OBJECTIVE_NAMES),
    *(f"lambda_{name}" for name in OBJECTIVE_NAMES),
    *(f"score_{name}" for name in OBJECTIVE_NAMES),
    "min_relationship_score",
    "lambda_sum",
    "lambda_min",
    "lambda_max",
    "l1_distance_to_p",
    "l2_distance_to_p",
    "validation_passed",
]

COST_COLUMNS = [
    "method",
    "method_family",
    "preference_name",
    "hyperparameter_id",
    "hyperparameters_json",
    "runtime_seconds",
    "peak_memory_mb",
    "solver_iterations",
    "solver_success",
    "num_objectives",
    "coefficient_dimension",
    "output_lambda_sum",
    "output_lambda_min",
    "output_lambda_max",
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


def hyperparameter_id(method: str, hyperparameters: dict[str, float]) -> str:
    """Create a compact stable identifier for one hyperparameter setting."""
    if not hyperparameters:
        return "default"
    parts = []
    for key, value in sorted(hyperparameters.items()):
        text = f"{value:g}".replace("-", "m").replace(".", "p")
        parts.append(f"{key}_{text}")
    return "__".join(parts)


def hyperparameters_json(hyperparameters: dict[str, float]) -> str:
    """Serialize hyperparameters in a stable form for CSV and metadata."""
    return json.dumps(hyperparameters, sort_keys=True)


def compute_lambda(
    method: str,
    preference: np.ndarray,
    relationships: np.ndarray,
    hyperparameters: dict[str, float],
) -> np.ndarray:
    """Compute one lambda vector for a method and hyperparameter setting."""
    if method == "direct_preference":
        return direct_preference_mapping(preference)
    if method == "uniform":
        return np.full(preference.size, 1.0 / preference.size)
    if method == "M1":
        return m1_mgda_inspired_mapping(
            preference,
            relationships,
            rho=hyperparameters["rho"],
        )
    if method == "M2":
        return m2_preference_weighted_alpha_mgda_mapping(
            preference,
            relationships,
            rho=hyperparameters["rho"],
        )
    if method == "C1":
        return c1_trust_region_cagrad_mapping(
            preference,
            relationships,
            c=hyperparameters["c"],
            eps=hyperparameters["eps"],
        )
    if method == "C2":
        return c2_softmin_cagrad_mapping(
            preference,
            relationships,
            tau=hyperparameters["tau"],
            rho=hyperparameters["rho"],
        )
    if method == "P1":
        return p1_conflict_weighted_shrinkage_mapping(
            preference,
            relationships,
            beta=hyperparameters["beta"],
        )
    if method == "P2":
        return p2_pcgrad_reconstruction_mapping(
            preference,
            relationships,
            rho=hyperparameters["rho"],
            eps=hyperparameters["eps"],
        )
    raise ValueError(f"Unknown method: {method}")


def validate_lambda(
    lambdas: np.ndarray,
    preference: np.ndarray,
    atol: float = VALIDATION_TOLERANCE,
) -> bool:
    """Return whether one lambda vector is finite and on the simplex."""
    if lambdas.shape != preference.shape:
        return False
    if not np.all(np.isfinite(lambdas)):
        return False
    if np.any(lambdas < -atol):
        return False
    return abs(float(lambdas.sum()) - 1.0) <= atol


def require_valid_lambda(
    lambdas: np.ndarray,
    preference: np.ndarray,
    label: str,
) -> None:
    """Raise a clear error if a lambda vector fails validation."""
    if not validate_lambda(lambdas, preference):
        raise ValueError(
            f"{label} produced an invalid lambda: shape={lambdas.shape}, "
            f"sum={float(np.sum(lambdas)) if lambdas.size else 'empty'}, "
            f"min={float(np.min(lambdas)) if lambdas.size else 'empty'}."
        )


def method_grid_items() -> list[tuple[str, dict[str, float]]]:
    """Return all baseline and method settings in a stable output order."""
    items: list[tuple[str, dict[str, float]]] = []
    for method, grid in BASELINE_GRIDS.items():
        items.extend((method, hyperparameters) for hyperparameters in grid)
    for method, grid in METHOD_GRIDS.items():
        items.extend((method, hyperparameters) for hyperparameters in grid)
    return items


def run_one_setting(
    *,
    preference_name: str,
    preference: np.ndarray,
    method: str,
    hyperparameters: dict[str, float],
    relationships: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute one coefficient row and one computational-cost row."""
    method_family = METHOD_FAMILIES[method]
    setting_id = hyperparameter_id(method, hyperparameters)
    hyper_json = hyperparameters_json(hyperparameters)

    tracemalloc.start()
    start_time = time.perf_counter()
    solver_success = True
    try:
        lambdas = compute_lambda(
            method=method,
            preference=preference,
            relationships=relationships,
            hyperparameters=hyperparameters,
        )
    except Exception:
        solver_success = False
        raise
    finally:
        runtime_seconds = time.perf_counter() - start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    require_valid_lambda(
        lambdas,
        preference,
        f"{preference_name}/{method}/{setting_id}",
    )
    validation_passed = validate_lambda(lambdas, preference)
    scores = relationships @ lambdas
    lambda_sum = float(lambdas.sum())
    lambda_min = float(np.min(lambdas))
    lambda_max = float(np.max(lambdas))

    coefficient_row: dict[str, Any] = {
        "preference_name": preference_name,
        "method": method,
        "method_family": method_family,
        "hyperparameter_id": setting_id,
        "hyperparameters_json": hyper_json,
        "min_relationship_score": float(np.min(scores)),
        "lambda_sum": lambda_sum,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "l1_distance_to_p": l1_distance(lambdas, preference),
        "l2_distance_to_p": l2_distance(lambdas, preference),
        "validation_passed": validation_passed,
    }
    coefficient_row.update(
        {
            f"p_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, preference)
        }
    )
    coefficient_row.update(
        {
            f"lambda_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, lambdas)
        }
    )
    coefficient_row.update(
        {
            f"score_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, scores)
        }
    )

    cost_row: dict[str, Any] = {
        "method": method,
        "method_family": method_family,
        "preference_name": preference_name,
        "hyperparameter_id": setting_id,
        "hyperparameters_json": hyper_json,
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
        "solver_iterations": "",
        "solver_success": solver_success and validation_passed,
        "num_objectives": len(OBJECTIVE_NAMES),
        "coefficient_dimension": lambdas.size,
        "output_lambda_sum": lambda_sum,
        "output_lambda_min": lambda_min,
        "output_lambda_max": lambda_max,
    }
    if method in {"direct_preference", "uniform", "P1"}:
        cost_row["solver_iterations"] = 0

    return coefficient_row, cost_row


def compute_rows(
    relationships: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute all coefficient rows and computational-cost rows."""
    coefficient_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    for preference_name, preference_values in PREFERENCES.items():
        preference = validate_preference_vector(preference_values)
        for method, hyperparameters in method_grid_items():
            coefficient_row, cost_row = run_one_setting(
                preference_name=preference_name,
                preference=preference,
                method=method,
                hyperparameters=hyperparameters,
                relationships=relationships,
            )
            coefficient_rows.append(coefficient_row)
            cost_rows.append(cost_row)

    validate_coverage(coefficient_rows)
    return coefficient_rows, cost_rows


def validate_coverage(rows: list[dict[str, Any]]) -> None:
    """Ensure every method/preference/hyperparameter setting appears once."""
    expected = {
        (
            preference_name,
            method,
            hyperparameter_id(method, hyperparameters),
        )
        for preference_name in PREFERENCES
        for method, hyperparameters in method_grid_items()
    }
    observed = {
        (
            str(row["preference_name"]),
            str(row["method"]),
            str(row["hyperparameter_id"]),
        )
        for row in rows
    }
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            f"Coefficient coverage mismatch. Missing={missing}, extra={extra}."
        )


def format_csv_value(value: Any) -> str | bool:
    """Format floats consistently while preserving strings and booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if value == "":
        return ""
    return f"{float(value):.10f}"


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    """Write rows to a small UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_csv_value(row[column]) for column in columns})


def write_metadata(
    output_path: Path,
    relationship_matrix_path: str,
    output_csv_path: str,
    output_costs_path: str,
    relationships: np.ndarray,
    rows: list[dict[str, Any]],
) -> None:
    """Write compact coefficient metadata to JSON."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "relationship_matrix_path": relationship_matrix_path,
        "output_csv_path": output_csv_path,
        "output_costs_path": output_costs_path,
        "objective_order": list(OBJECTIVE_NAMES),
        "preferences": {
            name: list(values) for name, values in PREFERENCES.items()
        },
        "method_families": METHOD_FAMILIES,
        "hyperparameter_grids": {
            **BASELINE_GRIDS,
            **METHOD_GRIDS,
        },
        "row_count": len(rows),
        "relationship_matrix_eigenvalues": np.linalg.eigvalsh(
            relationships
        ).tolist(),
        "validation": {
            "lambda_dimension_checked": True,
            "lambda_finite_checked": True,
            "lambda_non_negative_checked": True,
            "lambda_sum_checked": True,
            "method_preference_hyperparameter_coverage_checked": True,
            "tolerance": VALIDATION_TOLERANCE,
        },
        "note": (
            "The coefficients are one-shot mappings from p and R inside the "
            "fixed Rewarded-Soups-style interpolation family. Cost metrics are "
            "lightweight local runtime and tracemalloc measurements."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a compact summary table for the terminal."""
    print("\nHelpSteer2 coefficient summary:")
    print(
        "preference".ljust(22)
        + "method".ljust(20)
        + "setting".ljust(24)
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
            + str(row["method"]).ljust(20)
            + str(row["hyperparameter_id"]).ljust(24)
            + lambda_text.ljust(48)
            + f"{float(row['min_relationship_score']):.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute TinyLlama HelpSteer2 coefficient grids and lightweight "
            "cost logs."
        )
    )
    parser.add_argument(
        "--relationship_matrix_path",
        "--relationship-matrix-path",
        dest="relationship_matrix_path",
        default="results/tinyllama_helpsteer2_relationship_matrix.csv",
    )
    parser.add_argument(
        "--output_csv",
        "--output-csv",
        dest="output_csv",
        default="results/tinyllama_helpsteer2_all_method_coefficients.csv",
    )
    parser.add_argument(
        "--output_metadata",
        "--output-metadata",
        dest="output_metadata",
        default=(
            "results/tinyllama_helpsteer2_all_method_coefficients_metadata.json"
        ),
    )
    parser.add_argument(
        "--output_costs",
        "--output-costs",
        dest="output_costs",
        default="results/tinyllama_helpsteer2_method_costs.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Load R, compute coefficients, validate them, and write outputs."""
    args = parse_args()
    matrix_path = resolve_project_path(args.relationship_matrix_path)
    output_csv = resolve_project_path(args.output_csv)
    output_metadata = resolve_project_path(args.output_metadata)
    output_costs = resolve_project_path(args.output_costs)

    relationships = load_relationship_matrix(matrix_path)
    coefficient_rows, cost_rows = compute_rows(relationships)
    write_csv(output_csv, coefficient_rows, COEFFICIENT_COLUMNS)
    write_csv(output_costs, cost_rows, COST_COLUMNS)
    write_metadata(
        output_metadata,
        args.relationship_matrix_path,
        args.output_csv,
        args.output_costs,
        relationships,
        coefficient_rows,
    )
    print(f"Loaded relationship matrix from {matrix_path}")
    print_summary(coefficient_rows)
    print(f"\nSaved {len(coefficient_rows)} coefficient rows to {output_csv}")
    print(f"Saved {len(cost_rows)} cost rows to {output_costs}")
    print(f"Saved metadata to {output_metadata}")


if __name__ == "__main__":
    main()
