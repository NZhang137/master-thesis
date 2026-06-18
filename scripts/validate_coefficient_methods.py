"""Validate core coefficient-mapping formulas and generated coefficients."""

from __future__ import annotations

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
    m1_mgda_inspired_mapping,
    m2_preference_weighted_alpha_mgda_mapping,
    p1_conflict_weighted_shrinkage_mapping,
    p2_pcgrad_reconstruction_mapping,
)


OBJECTIVES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
METHODS = ("M1", "M2", "C1", "C2", "P1", "P2")
BASELINES = ("direct_preference", "uniform")


def assert_simplex(values: np.ndarray, label: str, atol: float = 1e-7) -> None:
    """Assert that a coefficient vector lies on the simplex."""
    if values.ndim != 1:
        raise AssertionError(f"{label} is not one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"{label} contains non-finite values.")
    if np.any(values < -atol):
        raise AssertionError(f"{label} contains negative values: {values}")
    if abs(float(values.sum()) - 1.0) > atol:
        raise AssertionError(f"{label} sums to {float(values.sum())}, not 1.")


def validate_p1_closed_form() -> None:
    """Check P1 against the thesis shrinkage formula on a hand-built matrix."""
    p = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    R = np.array(
        [
            [1.0, -0.5, 0.2],
            [-0.5, 1.0, -0.1],
            [0.2, -0.1, 1.0],
        ],
        dtype=np.float64,
    )
    beta = 2.0
    kappa = np.array([0.5, 0.6, 0.1], dtype=np.float64)
    shrinkage = 1.0 / (1.0 + beta * kappa)
    expected = p * shrinkage
    expected = expected / float(expected.sum())
    observed = p1_conflict_weighted_shrinkage_mapping(p, R, beta=beta)
    if not np.allclose(observed, expected, atol=1e-10):
        raise AssertionError(f"P1 mismatch: observed={observed}, expected={expected}")
    assert_simplex(observed, "P1 hand-built output")


def validate_p2_no_conflict_identity() -> None:
    """Check that P2 returns p when R has no negative conflicts."""
    p = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    observed = p2_pcgrad_reconstruction_mapping(p, R, rho=1.0, eps=1e-8)
    if not np.allclose(observed, p, atol=1e-7):
        raise AssertionError(f"P2 no-conflict case should return p, got {observed}")
    assert_simplex(observed, "P2 no-conflict output")


def validate_all_methods_return_simplex() -> None:
    """Run all coefficient methods on a small PSD matrix."""
    p = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    R = np.array(
        [
            [1.0, -0.2, 0.1],
            [-0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    outputs = {
        "M1": m1_mgda_inspired_mapping(p, R, rho=1.0),
        "M2": m2_preference_weighted_alpha_mgda_mapping(p, R, rho=1.0),
        "C1": c1_trust_region_cagrad_mapping(p, R, c=0.5, eps=1e-8),
        "C2": c2_softmin_cagrad_mapping(p, R, tau=0.1, rho=1.0),
        "P1": p1_conflict_weighted_shrinkage_mapping(p, R, beta=1.0),
        "P2": p2_pcgrad_reconstruction_mapping(p, R, rho=1.0, eps=1e-8),
    }
    for method, values in outputs.items():
        assert_simplex(values, f"{method} output")


def validate_generated_csv(csv_path: Path, metadata_path: Path) -> None:
    """Validate method coverage, objective order, and simplex rows in CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing coefficient CSV: {csv_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing coefficient metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if tuple(metadata.get("objective_order", ())) != OBJECTIVES:
        raise AssertionError("Metadata objective_order does not match HelpSteer2 order.")

    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise AssertionError("Coefficient CSV contains no rows.")

    preferences = sorted({row["preference_name"] for row in rows})
    expected = {
        (preference, method)
        for preference in preferences
        for method in (*BASELINES, *METHODS)
    }
    observed = {(row["preference_name"], row["method"]) for row in rows}
    if observed != expected:
        raise AssertionError(
            f"Method/preference coverage mismatch. Missing={expected - observed}, "
            f"extra={observed - expected}"
        )

    for row in rows:
        lambdas = np.array(
            [float(row[f"lambda_{objective}"]) for objective in OBJECTIVES],
            dtype=np.float64,
        )
        assert_simplex(lambdas, f"{row['preference_name']}/{row['method']}")


def main() -> None:
    """Run all lightweight coefficient validation checks."""
    validate_p1_closed_form()
    validate_p2_no_conflict_identity()
    validate_all_methods_return_simplex()
    validate_generated_csv(
        PROJECT_ROOT / "results/helpsteer2_all_method_coefficients.csv",
        PROJECT_ROOT / "results/helpsteer2_all_method_coefficients_metadata.json",
    )
    print("Coefficient method validation passed.")


if __name__ == "__main__":
    main()
