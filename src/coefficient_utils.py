"""Preference-to-coefficient utilities for the first M1 prototype.

M1 applies a one-shot relationship-softmax correction inside the existing
model-merging family. It does not use coefficient-space optimization.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_simplex(
    x: Sequence[float] | np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Validate a non-negative vector and normalize it to sum to one."""
    if eps <= 0:
        raise ValueError("eps must be positive.")

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values.")
    if np.any(values < 0):
        raise ValueError("x must contain only non-negative values.")

    total = float(values.sum())
    if total <= eps:
        raise ValueError("x must contain at least one positive value.")

    normalized = values / total
    return normalized / normalized.sum()


def relationship_softmax_mapping(
    p: Sequence[float] | np.ndarray,
    R: Sequence[Sequence[float]] | np.ndarray,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute M1 coefficients from preferences and a relationship matrix.

    The mapping is ``lambda_i proportional to p_i * exp(tau * (R @ p)_i)``.
    Zero preference entries remain zero. Setting ``tau=0`` returns the
    normalized preference vector exactly.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be a finite, non-negative value.")

    preferences = normalize_simplex(p, eps=eps)
    relationships = np.asarray(R, dtype=np.float64)
    expected_shape = (preferences.size, preferences.size)

    if relationships.shape != expected_shape:
        raise ValueError(f"R must have shape {expected_shape}.")
    if not np.all(np.isfinite(relationships)):
        raise ValueError("R must contain only finite values.")
    if tau == 0:
        return preferences.copy()

    scores = relationships @ preferences
    positive_mask = preferences > 0

    # Work in log space and shift the active support for numerical stability.
    log_weights = np.full_like(preferences, -np.inf)
    log_weights[positive_mask] = (
        np.log(preferences[positive_mask]) + tau * scores[positive_mask]
    )
    if not np.all(np.isfinite(log_weights[positive_mask])):
        raise ValueError("The scaled relationship scores are not finite.")

    shift = float(np.max(log_weights[positive_mask]))
    unnormalized = np.zeros_like(preferences)
    unnormalized[positive_mask] = np.exp(log_weights[positive_mask] - shift)

    return normalize_simplex(unnormalized, eps=eps)


def l1_distance(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Return the L1 distance between equally shaped finite vectors."""
    vec_a, vec_b = _validate_distance_vectors(a, b)
    return float(np.linalg.norm(vec_a - vec_b, ord=1))


def l2_distance(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Return the Euclidean distance between equally shaped finite vectors."""
    vec_a, vec_b = _validate_distance_vectors(a, b)
    return float(np.linalg.norm(vec_a - vec_b, ord=2))


def _validate_distance_vectors(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two vectors used by the distance helpers."""
    vec_a = np.asarray(a, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)

    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError("Distance inputs must be one-dimensional vectors.")
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Distance inputs must have the same shape, got "
            f"{vec_a.shape} and {vec_b.shape}."
        )
    if vec_a.size == 0:
        raise ValueError("Distance inputs must not be empty.")
    if not np.all(np.isfinite(vec_a)) or not np.all(np.isfinite(vec_b)):
        raise ValueError("Distance inputs must contain only finite values.")

    return vec_a, vec_b
