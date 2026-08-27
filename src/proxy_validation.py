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
    # The SFT exporter names the first column ``adapter``; NB09.1 run1 left the
    # same label column unnamed. Its position, not its header spelling, is the
    # stable part of the format, so accept both without rewriting hashed data.
    frame = pd.read_csv(matrix_path, index_col=0)
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


def preference_utility(
    reward_matrix: np.ndarray,
    preference: Sequence[float],
    normalization: str = "identity",
    eps: float = 1e-8,
) -> np.ndarray:
    """Return one preference-weighted utility per reward-matrix row.

    ``identity`` is NB10's primary raw-score metric. ``minmax`` and ``rank``
    reuse the same evaluated matrix and are reported only as robustness checks.
    """
    rewards = np.asarray(reward_matrix, dtype=np.float64)
    if rewards.ndim != 2:
        raise ValueError("reward_matrix must be a two-dimensional array.")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("reward_matrix contains non-finite values.")
    p = validate_simplex_vector(preference, name="preference")
    if p.size != rewards.shape[1]:
        raise ValueError(
            f"preference has length {p.size}, reward_matrix has "
            f"{rewards.shape[1]} objectives."
        )

    if normalization == "identity":
        normalized = rewards
    elif normalization == "minmax":
        normalized, _, _ = normalize_rewards(rewards, eps=eps)
    elif normalization == "rank":
        from scipy.stats import rankdata

        normalized = np.column_stack(
            [
                rankdata(rewards[:, index]) / rewards.shape[0]
                for index in range(rewards.shape[1])
            ]
        )
    else:
        raise ValueError(
            f"Unknown normalization {normalization!r}; use 'identity', "
            "'minmax' or 'rank'."
        )
    return normalized @ p


def normalization_agreement(
    reward_matrix: np.ndarray,
    preference: Sequence[float],
) -> dict[str, Any]:
    """Compare orderings induced by identity, min-max and rank utility."""
    utilities = {
        name: preference_utility(reward_matrix, preference, normalization=name)
        for name in ("identity", "minmax", "rank")
    }
    result: dict[str, Any] = {
        "argmax_index": {
            name: int(np.argmax(values)) for name, values in utilities.items()
        },
        "spearman": {},
    }
    names = list(utilities)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result["spearman"][f"{left}_vs_{right}"] = safe_spearman(
                utilities[left], utilities[right]
            )
    result["argmax_agrees"] = len(set(result["argmax_index"].values())) == 1
    return result


def coefficient_key(coefficient: Sequence[float]) -> str:
    """Return a stable cache key for a coefficient vector."""
    # Round FIRST, then normalize the sign: `+ 0.0` turns -0.0 into +0.0,
    # but leaves -1e-12 untouched, which still formats as "-0.00000000".
    return "|".join(f"{round(float(value), 8) + 0.0:.8f}" for value in coefficient)


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


# --- append to src/proxy_validation.py ---------------------------------------
# `collect_reward_matrix` above stores only the per-lambda MEAN over prompts.
# The pre-registered error layer is a PAIRED BOOTSTRAP OVER PROMPTS, which needs
# the per-prompt values. Once Phase B has run with the mean-only collector, those
# values are gone and the CI cannot be recovered without re-running ~9.4 h of
# scoring. Use this collector for Phase B; `collect_reward_matrix` remains for
# callers that genuinely only need means.


def collect_reward_tensor(
    coefficients: np.ndarray,
    rewards_of_lambda: Callable[[np.ndarray], np.ndarray],
    cache_path: str | Path,
    num_prompts: int,
    binding_sha256: str,
) -> np.ndarray:
    """Collect per-prompt rewards with one slice per coefficient vector.

    ``rewards_of_lambda`` must return an array of shape ``(num_prompts,
    num_objectives)`` - the UNAVERAGED scores. The returned tensor has shape
    ``(num_lambda, num_prompts, num_objectives)``; ``tensor.mean(axis=1)``
    reproduces exactly what ``collect_reward_matrix`` would have returned, so
    every downstream consumer of the mean matrix keeps working.

    Caching and resumption follow ``collect_reward_matrix``: one JSONL line per
    completed coefficient, keyed by ``coefficient_key``, flushed immediately so
    an interrupted Colab session resumes instead of restarting.

    ``binding_sha256`` ties the cache to one frozen run. It is written as the
    first line and re-checked on resume. Without it, a cache written under one
    set of adapters, prompts or scorer settings would be silently reused under
    another - the resumption logic would then be actively harmful rather than
    merely useless.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a two-dimensional array.")
    num_rows, num_objectives = coefficients.shape
    if num_prompts < 1:
        raise ValueError("num_prompts must be at least 1.")
    expected_shape = (num_prompts, num_objectives)

    header = {"binding_sha256": str(binding_sha256), "num_prompts": int(num_prompts),
              "num_objectives": int(num_objectives)}
    completed: dict[str, list[list[float]]] = {}
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        lines = [line for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = json.loads(lines[0])
        if "binding_sha256" not in first:
            raise ValueError(
                f"{cache_path} has no binding header. It was written before the run was "
                "bound to a pre-registration. Start a new cache file."
            )
        if first["binding_sha256"] != header["binding_sha256"]:
            raise ValueError(
                "Reward cache belongs to a different run.\n"
                f"  cache:   {first['binding_sha256'][:16]}\n"
                f"  current: {header['binding_sha256'][:16]}\n"
                "Adapters, prompts, matrices, scorer or grids changed since this cache was "
                "written. Use a new RUN_TAG rather than mixing two runs in one file."
            )
        for line in lines[1:]:
            record = json.loads(line)
            completed[str(record["key"])] = record["scores"]
    else:
        cache_path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    tensor = np.full((num_rows, num_prompts, num_objectives), np.nan, dtype=np.float64)
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for index, coefficient in enumerate(coefficients):
            key = coefficient_key(coefficient)
            if key in completed:
                cached = np.asarray(completed[key], dtype=np.float64)
                if cached.shape != expected_shape:
                    raise ValueError(
                        f"Cached entry {key} has shape {cached.shape}, expected "
                        f"{expected_shape}. The prompt set changed since this cache "
                        "was written; start a new cache file rather than mixing them."
                    )
                tensor[index] = cached
                continue

            scores = np.asarray(rewards_of_lambda(coefficient), dtype=np.float64)
            if scores.shape != expected_shape:
                raise ValueError(
                    f"rewards_of_lambda must return shape {expected_shape}, "
                    f"got {scores.shape}."
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("rewards_of_lambda returned non-finite values.")
            tensor[index] = scores
            cache_file.write(
                json.dumps({
                    "key": key,
                    "index": index,
                    "lambda": coefficient.tolist(),
                    "scores": scores.tolist(),
                    # `reward` is the per-lambda mean under the key that
                    # `collect_reward_matrix` used. Cells 30 and 34 read exactly
                    # this key, so they keep working without modification.
                    "reward": scores.mean(axis=0).tolist(),
                })
                + "\n"
            )
            cache_file.flush()
            print(
                f"[reward] {index + 1}/{num_rows} "
                f"lambda={np.round(coefficient, 3)} "
                f"mean={np.round(scores.mean(axis=0), 4)}"
            )

    if np.isnan(tensor).any():
        raise RuntimeError("Reward tensor still has NaNs after collection.")
    return tensor
