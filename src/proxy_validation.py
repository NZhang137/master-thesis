"""Utilities for validating LoRA-geometry proxies against reward utility.

The functions in this module keep the RQ2 proxy-validity experiment split into
four steps:

1. build a fixed search set of merge coefficients,
2. check geometry-score dynamic range before loading reward models,
3. collect an expensive reward matrix with caching,
4. compare geometry scores with preference-weighted reward utility.

ArmoRM scores are evaluation-only. They must not feed back into adapter
selection, relationship-matrix construction, or checkpoint choice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def validate_simplex_vector(values: Sequence[float], *, name: str = "vector") -> np.ndarray:
    """Return a finite non-negative vector that sums to one."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} contains negative values.")
    total = float(array.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{name} sums to {total:.12g}, not 1.")
    return array


def build_search_set(
    m: int,
    *,
    n_dirichlet: int = 64,
    dirichlet_alpha: float = 1.0,
    preferences: Sequence[Sequence[float]] | None = None,
    seed: int = 137,
    include_vertices: bool = True,
    include_uniform: bool = True,
    dedup_tol: float = 1e-6,
) -> np.ndarray:
    """Return an ``(N, m)`` array of merge coefficients on the simplex."""
    if m < 2:
        raise ValueError("m must be at least 2.")
    if n_dirichlet < 0:
        raise ValueError("n_dirichlet must be non-negative.")
    if dirichlet_alpha <= 0 or not math.isfinite(dirichlet_alpha):
        raise ValueError("dirichlet_alpha must be finite and positive.")

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    if include_vertices:
        rows.extend(np.eye(m, dtype=np.float64))
    if include_uniform:
        rows.append(np.full(m, 1.0 / m, dtype=np.float64))
    if preferences is not None:
        for index, preference in enumerate(preferences):
            rows.append(validate_simplex_vector(preference, name=f"preference[{index}]"))
    if n_dirichlet:
        rows.extend(
            rng.dirichlet(np.full(m, dirichlet_alpha, dtype=np.float64), size=n_dirichlet)
        )

    kept: list[np.ndarray] = []
    for row in rows:
        if not any(np.linalg.norm(row - previous) < dedup_tol for previous in kept):
            kept.append(row)
    return np.asarray(kept, dtype=np.float64)


def load_labeled_matrix_csv(path: str | Path, attributes: Sequence[str]) -> np.ndarray:
    """Load a labeled square matrix CSV in the requested attribute order."""
    matrix_path = Path(path)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Matrix CSV not found: {matrix_path}")
    frame = pd.read_csv(matrix_path, index_col="adapter")
    missing_rows = [name for name in attributes if name not in frame.index]
    missing_columns = [name for name in attributes if name not in frame.columns]
    if missing_rows or missing_columns:
        raise ValueError(
            f"Matrix {matrix_path} does not contain the expected labels; "
            f"missing rows={missing_rows}, missing columns={missing_columns}."
        )
    matrix = frame.loc[list(attributes), list(attributes)].to_numpy(dtype=np.float64)
    validate_square_matrix(matrix, label=str(matrix_path))
    return matrix


def validate_square_matrix(matrix: np.ndarray, *, label: str) -> np.ndarray:
    """Validate and symmetrize a finite square matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{label} must be a square matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values.")
    if not np.allclose(values, values.T, atol=1e-8):
        raise ValueError(f"{label} must be symmetric.")
    return 0.5 * (values + values.T)


def geometry_scores(
    coefficients: np.ndarray,
    relationship_matrix: np.ndarray,
    preference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return worst-case and preference-average geometry scores over coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    relationship_matrix = validate_square_matrix(
        relationship_matrix,
        label="relationship_matrix",
    )
    response = coefficients @ relationship_matrix
    score_worst = response.min(axis=1)
    score_avg = response @ preference if preference is not None else None
    return score_worst, score_avg


def analyze_geometry_only(
    coefficients: np.ndarray,
    R_cos: np.ndarray,
    R_gram: np.ndarray,
    preferences: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Summarize dynamic range of geometry scores before reward-model collection."""
    output: dict[str, Any] = {"per_R": {}}
    for matrix_name, matrix in (("cosine", R_cos), ("gram", R_gram)):
        worst, _ = geometry_scores(coefficients, matrix)
        avg_ranges = {}
        for preference_name, preference_values in preferences.items():
            preference = validate_simplex_vector(preference_values, name=preference_name)
            _, avg = geometry_scores(coefficients, matrix, preference)
            assert avg is not None
            avg_ranges[preference_name] = float(avg.max() - avg.min())
        worst_range = float(worst.max() - worst.min())
        output["per_R"][matrix_name] = {
            "score_worst_min": float(worst.min()),
            "score_worst_max": float(worst.max()),
            "score_worst_range": worst_range,
            "score_worst_rel_range": worst_range / (abs(float(worst.mean())) + 1e-12),
            "score_avg_ranges_by_preference": avg_ranges,
            "score_avg_mean_range_over_preferences": float(np.mean(list(avg_ranges.values()))),
        }
    return output


def collect_reward_matrix(
    coefficients: np.ndarray,
    reward_of_lambda: Callable[[np.ndarray], np.ndarray],
    cache_path: str | Path,
) -> np.ndarray:
    """Collect or resume a reward matrix with one row per coefficient vector."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a two-dimensional array.")
    num_rows, num_objectives = coefficients.shape

    completed: dict[str, list[float]] = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[str(record["key"])] = list(record["reward"])

    rewards = np.full((num_rows, num_objectives), np.nan, dtype=np.float64)
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for index, coefficient in enumerate(coefficients):
            key = coefficient_key(coefficient)
            if key in completed:
                rewards[index] = completed[key]
                continue
            reward = np.asarray(reward_of_lambda(coefficient), dtype=np.float64)
            if reward.shape != (num_objectives,):
                raise ValueError(
                    f"reward_of_lambda must return shape ({num_objectives},), got {reward.shape}."
                )
            if not np.all(np.isfinite(reward)):
                raise ValueError(f"reward_of_lambda returned non-finite values: {reward}")
            rewards[index] = reward
            cache_file.write(
                json.dumps(
                    {
                        "key": key,
                        "index": index,
                        "lambda": coefficient.tolist(),
                        "reward": reward.tolist(),
                    }
                )
                + "\n"
            )
            cache_file.flush()
            print(
                f"[reward] {index + 1}/{num_rows} "
                f"lambda={np.round(coefficient, 3)} reward={np.round(reward, 4)}"
            )
    if np.isnan(rewards).any():
        raise RuntimeError("Reward matrix still has NaNs after collection.")
    return rewards


def coefficient_key(coefficient: Sequence[float]) -> str:
    """Return a stable cache key for a coefficient vector."""
    return "|".join(f"{float(value):.8f}" for value in coefficient)


def normalize_rewards(
    reward_matrix: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize each objective using the evaluated search set."""
    rewards = np.asarray(reward_matrix, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError("reward_matrix must be a two-dimensional array.")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("reward_matrix contains non-finite values.")
    z_best = rewards.max(axis=0)
    z_worst = rewards.min(axis=0)
    denominator = np.where((z_best - z_worst) < eps, eps, z_best - z_worst)
    return (rewards - z_worst) / denominator, z_best, z_worst


def run_spearman_analysis(
    coefficients: np.ndarray,
    reward_matrix: np.ndarray,
    R_cos: np.ndarray,
    R_gram: np.ndarray,
    preferences: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Compare geometry scores with preference-weighted reward utility."""
    normalized_rewards, z_best, z_worst = normalize_rewards(reward_matrix)
    aggregate_values: dict[tuple[str, str], list[float]] = {
        ("cosine", "worst"): [],
        ("cosine", "avg"): [],
        ("gram", "worst"): [],
        ("gram", "avg"): [],
    }
    per_preference: dict[str, Any] = {}

    for preference_name, preference_values in preferences.items():
        preference = validate_simplex_vector(preference_values, name=preference_name)
        utility = normalized_rewards @ preference
        best_index = int(np.argmax(utility))
        entry: dict[str, Any] = {
            "lambda_best": coefficients[best_index].tolist(),
            "U_best": float(utility[best_index]),
            "U_at_p_baseline": None,
            "rho": {},
        }
        baseline_matches = np.where(np.linalg.norm(coefficients - preference, axis=1) < 1e-6)[0]
        if baseline_matches.size:
            baseline_index = int(baseline_matches[0])
            entry["U_at_p_baseline"] = float(utility[baseline_index])
            entry["gap_best_vs_p"] = entry["U_best"] - entry["U_at_p_baseline"]

        for matrix_name, matrix in (("cosine", R_cos), ("gram", R_gram)):
            worst, avg = geometry_scores(coefficients, matrix, preference)
            assert avg is not None
            for score_name, score_values in (("worst", worst), ("avg", avg)):
                rho = safe_spearman(score_values, utility)
                entry["rho"][f"{matrix_name}_{score_name}"] = rho
                if rho is not None and np.isfinite(rho):
                    aggregate_values[(matrix_name, score_name)].append(float(rho))
        per_preference[preference_name] = entry

    aggregate = {}
    for (matrix_name, score_name), values in aggregate_values.items():
        if values:
            aggregate[f"{matrix_name}_{score_name}"] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "worst": float(np.min(values)),
                "n": len(values),
            }

    return {
        "per_preference": per_preference,
        "aggregate": aggregate,
        "z_best": z_best.tolist(),
        "z_worst": z_worst.tolist(),
        "note": "rho is computed against min-max normalized rewards; constants are estimated from B.",
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Spearman rho, or None when one side is constant."""
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    rho, _ = spearmanr(x, y)
    return float(rho)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_numpy(path: str | Path, values: np.ndarray) -> None:
    """Write an array to an ``.npy`` file, creating parents as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(values))

