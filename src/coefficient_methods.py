"""Generic preference-to-coefficient mappings for model-merging prototypes.

The functions in this module operate on an arbitrary number of objectives.
They provide the direct-preference baseline, the M1 relationship-softmax
mapping, and the C1 CAGrad-inspired one-shot mapping.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ArrayLike1D = Sequence[float] | np.ndarray
ArrayLike2D = Sequence[Sequence[float]] | np.ndarray


def normalize_simplex(
    x: ArrayLike1D,
    eps: float = 1e-12,
) -> np.ndarray:
    """Validate and normalize a non-negative vector onto the simplex."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive value.")

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
    return normalized / float(normalized.sum())


def validate_preference_vector(
    p: ArrayLike1D,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return a finite, non-negative preference vector on the simplex."""
    return normalize_simplex(p, eps=eps)


def validate_relationship_matrix(
    R: ArrayLike2D,
    num_objectives: int | None = None,
) -> np.ndarray:
    """Validate a square finite matrix and return its symmetric part."""
    relationships = np.asarray(R, dtype=np.float64)
    if relationships.ndim != 2 or relationships.shape[0] == 0:
        raise ValueError("R must be a non-empty two-dimensional matrix.")
    if relationships.shape[0] != relationships.shape[1]:
        raise ValueError("R must be square.")
    if num_objectives is not None:
        if num_objectives < 1:
            raise ValueError("num_objectives must be positive.")
        expected_shape = (num_objectives, num_objectives)
        if relationships.shape != expected_shape:
            raise ValueError(f"R must have shape {expected_shape}.")
    if not np.all(np.isfinite(relationships)):
        raise ValueError("R must contain only finite values.")

    return 0.5 * (relationships + relationships.T)


def l1_distance(a: ArrayLike1D, b: ArrayLike1D) -> float:
    """Return the L1 distance between equally shaped finite vectors."""
    vec_a, vec_b = _validate_distance_vectors(a, b)
    return float(np.linalg.norm(vec_a - vec_b, ord=1))


def l2_distance(a: ArrayLike1D, b: ArrayLike1D) -> float:
    """Return the Euclidean distance between equally shaped finite vectors."""
    vec_a, vec_b = _validate_distance_vectors(a, b)
    return float(np.linalg.norm(vec_a - vec_b, ord=2))


def make_psd_matrix(
    R: ArrayLike2D,
    eigenvalue_floor: float = 0.0,
) -> np.ndarray:
    """Return a symmetric positive-semidefinite-safe version of ``R``.

    The matrix is symmetrized and reconstructed after clipping its eigenvalues
    to ``eigenvalue_floor``. The default floor of zero produces a
    positive-semidefinite matrix up to floating-point precision.
    """
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor < 0:
        raise ValueError("eigenvalue_floor must be finite and non-negative.")

    symmetric = validate_relationship_matrix(R)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    psd = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (psd + psd.T)


def direct_preference_mapping(
    p: ArrayLike1D,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the normalized direct-preference baseline ``lambda = p``."""
    return validate_preference_vector(p, eps=eps).copy()


def relationship_softmax_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the numerically stable M1 relationship-softmax mapping.

    The mapping uses ``scores = R @ p`` and
    ``lambda_i proportional to p_i * exp(tau * scores_i)``. Zero preference
    entries remain zero, and ``tau=0`` returns the normalized preference
    vector exactly.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be a finite, non-negative value.")

    preferences = validate_preference_vector(p, eps=eps)
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )
    if tau == 0:
        return preferences.copy()

    scores = relationships @ preferences
    positive = preferences > 0
    log_weights = np.full(preferences.shape, -np.inf, dtype=np.float64)
    log_weights[positive] = (
        np.log(preferences[positive]) + tau * scores[positive]
    )
    if not np.all(np.isfinite(log_weights[positive])):
        raise ValueError("The scaled M1 scores must be finite.")

    shift = float(np.max(log_weights[positive]))
    weights = np.zeros_like(preferences)
    weights[positive] = np.exp(log_weights[positive] - shift)
    return normalize_simplex(weights, eps=eps)


def c1_cagrad_inspired_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    rho: float = 1.0,
    eps: float = 1e-12,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute the C1 CAGrad-inspired one-shot coefficient mapping.

    C1 maximizes

    ``min_i (R @ lambda)_i - rho * (lambda - p)^T R_psd (lambda - p)``

    over the probability simplex. The SLSQP problem introduces one auxiliary
    variable for the minimum relationship score and starts from ``p``. If
    SciPy raises, reports failure, or returns an invalid point, a deterministic
    fallback selects the best candidate among ``p``, the uniform vector, and
    all simplex vertices under the same objective.
    """
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be a finite, non-negative value.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    preferences = validate_preference_vector(p, eps=eps)
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )
    relationships_psd = make_psd_matrix(relationships)
    fallback = _c1_deterministic_fallback(
        preferences,
        relationships,
        relationships_psd,
        rho,
    )

    try:
        from scipy.optimize import minimize

        initial_score = float(np.min(relationships @ preferences))
        initial = np.concatenate([preferences, [initial_score]])
        objective_count = preferences.size

        def objective(variables: np.ndarray) -> float:
            lambdas = variables[:objective_count]
            worst_score = variables[-1]
            difference = lambdas - preferences
            penalty = float(
                difference @ relationships_psd @ difference
            )
            return float(-worst_score + rho * penalty)

        constraints = (
            {
                "type": "eq",
                "fun": lambda variables: (
                    float(np.sum(variables[:objective_count])) - 1.0
                ),
            },
            {
                "type": "ineq",
                "fun": lambda variables: (
                    relationships @ variables[:objective_count] - variables[-1]
                ),
            },
        )
        bounds = [(0.0, 1.0)] * objective_count + [(None, None)]
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )

        if not result.success or not np.all(np.isfinite(result.x)):
            return fallback

        candidate = np.clip(result.x[:objective_count], 0.0, None)
        candidate = normalize_simplex(candidate, eps=eps)
        if _c1_objective(
            candidate,
            preferences,
            relationships,
            relationships_psd,
            rho,
        ) + 1e-10 < _c1_objective(
            fallback,
            preferences,
            relationships,
            relationships_psd,
            rho,
        ):
            return fallback
        return candidate
    except Exception:
        return fallback


def _c1_objective(
    lambdas: np.ndarray,
    preferences: np.ndarray,
    relationships: np.ndarray,
    relationships_psd: np.ndarray,
    rho: float,
) -> float:
    """Evaluate the C1 maximization objective for a simplex vector."""
    difference = lambdas - preferences
    penalty = float(difference @ relationships_psd @ difference)
    return float(np.min(relationships @ lambdas) - rho * penalty)


def _c1_deterministic_fallback(
    preferences: np.ndarray,
    relationships: np.ndarray,
    relationships_psd: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Select the best deterministic feasible candidate for C1."""
    objective_count = preferences.size
    candidates = [
        preferences.copy(),
        np.full(objective_count, 1.0 / objective_count, dtype=np.float64),
        *np.eye(objective_count, dtype=np.float64),
    ]
    values = [
        _c1_objective(
            candidate,
            preferences,
            relationships,
            relationships_psd,
            rho,
        )
        for candidate in candidates
    ]
    return candidates[int(np.argmax(values))].copy()


def _validate_distance_vectors(
    a: ArrayLike1D,
    b: ArrayLike1D,
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
