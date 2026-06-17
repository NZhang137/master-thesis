"""Generic preference-to-coefficient mappings for model-merging prototypes.

The functions in this module operate on an arbitrary number of objectives and
return merge coefficients on the simplex. They implement one-shot mappings of
the form ``lambda = f(p, R)`` from a preference vector and a static relationship
matrix. They do not implement training-time coefficient-space gradients.
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


def m1_mgda_inspired_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    rho: float = 1.0,
    eps: float = 1e-12,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute M1: the MGDA-inspired one-shot mapping.

    Given ``p`` and ``R``, solve

    ``argmin_lambda lambda.T @ R @ lambda + rho * ||lambda - p||_2^2``

    over the probability simplex.
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")

    preferences = validate_preference_vector(p, eps=eps)
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )

    def objective(lambdas: np.ndarray) -> float:
        difference = lambdas - preferences
        return float(
            lambdas @ relationships @ lambdas
            + rho * np.dot(difference, difference)
        )

    fallback = preferences.copy()
    return _minimize_on_simplex(
        objective=objective,
        initial=preferences,
        fallback=fallback,
        eps=eps,
        max_iterations=max_iterations,
    )


def m2_preference_weighted_alpha_mgda_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    rho: float = 1.0,
    eps: float = 1e-12,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute M2: the preference-weighted alpha-MGDA variant.

    Let ``G_p = Diag(p) @ R @ Diag(p)`` and ``u = 1/m``. Solve

    ``argmin_alpha alpha.T @ G_p @ alpha + rho * ||alpha - u||_2^2``

    over the simplex, then return

    ``lambda = (alpha * p) / sum(alpha * p)``.
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")

    preferences = validate_preference_vector(p, eps=eps)
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )
    objective_count = preferences.size
    uniform = np.full(objective_count, 1.0 / objective_count, dtype=np.float64)
    preference_diagonal = np.diag(preferences)
    weighted_relationships = preference_diagonal @ relationships @ preference_diagonal

    def objective(alpha: np.ndarray) -> float:
        difference = alpha - uniform
        return float(
            alpha @ weighted_relationships @ alpha
            + rho * np.dot(difference, difference)
        )

    alpha = _minimize_on_simplex(
        objective=objective,
        initial=uniform,
        fallback=uniform,
        eps=eps,
        max_iterations=max_iterations,
    )
    multiplicative = alpha * preferences
    if float(multiplicative.sum()) <= eps:
        return preferences.copy()
    return normalize_simplex(multiplicative, eps=eps)


def p1_pcgrad_reconstruction_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    rho: float = 1.0,
    eps: float = 1e-8,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute P1: R-metric PCGrad with strongest-conflict ordering.

    This implements the thesis PCGrad-inspired mapping using the equivalent
    ``R``-metric representation. Conflicting ordered pairs are processed from
    strongest to weakest negative conflict, then coefficients are reconstructed
    by solving the simplex least-squares problem from the definition.
    """
    return _pcgrad_reconstruction_mapping(
        p=p,
        R=R,
        rho=rho,
        eps=eps,
        reverse_order=False,
        max_iterations=max_iterations,
    )


def p2_pcgrad_reconstruction_reverse_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    rho: float = 1.0,
    eps: float = 1e-8,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute P2: the reverse deterministic PCGrad order variant.

    The thesis notes that the PCGrad-inspired mapping may depend on projection
    order. P2 keeps the same R-metric projection and reconstruction equations
    as P1, but processes the deterministic strongest-conflict list in reverse.
    """
    return _pcgrad_reconstruction_mapping(
        p=p,
        R=R,
        rho=rho,
        eps=eps,
        reverse_order=True,
        max_iterations=max_iterations,
    )


def c1_trust_region_cagrad_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    c: float = 0.5,
    eps: float = 1e-8,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute C1: the trust-region CAGrad-inspired method.

    Solve

    ``max_lambda min_i (R @ lambda)_i``

    over the simplex subject to

    ``(lambda - p).T @ R @ (lambda - p) <= c^2 * max(p.T @ R @ p, eps)``.
    """
    if not np.isfinite(c) or c < 0:
        raise ValueError("c must be finite and non-negative.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    preferences = validate_preference_vector(p, eps=eps)
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )
    objective_count = preferences.size
    trust_radius = c**2 * max(
        float(preferences @ relationships @ preferences),
        eps,
    )
    fallback = _best_feasible_min_score_candidate(
        preferences,
        relationships,
        trust_radius,
        eps=eps,
    )

    try:
        from scipy.optimize import minimize

        initial_t = float(np.min(relationships @ preferences))
        initial = np.concatenate([preferences, [initial_t]])

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
            {
                "type": "ineq",
                "fun": lambda variables: (
                    trust_radius
                    - float(
                        (variables[:objective_count] - preferences)
                        @ relationships
                        @ (variables[:objective_count] - preferences)
                    )
                ),
            },
        )
        bounds = [(0.0, 1.0)] * objective_count + [(None, None)]
        result = minimize(
            lambda variables: float(-variables[-1]),
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return fallback
        candidate = normalize_simplex(
            np.clip(result.x[:objective_count], 0.0, None),
            eps=eps,
        )
        if not _is_trust_region_feasible(
            candidate,
            preferences,
            relationships,
            trust_radius,
            eps=eps,
        ):
            return fallback
        if float(np.min(relationships @ candidate)) + 1e-10 < float(
            np.min(relationships @ fallback)
        ):
            return fallback
        return candidate
    except Exception:
        return fallback


def c2_softmin_cagrad_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    tau: float = 0.1,
    rho: float = 1.0,
    eps: float = 1e-12,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Compute C2: the soft-min CAGrad-inspired variant.

    Solve

    ``argmax_lambda softmin_tau(R @ lambda)
       - rho * (lambda - p).T @ R @ (lambda - p)``

    over the probability simplex.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive.")
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and non-negative.")

    preferences = validate_preference_vector(p, eps=eps)
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )

    def objective(lambdas: np.ndarray) -> float:
        scores = relationships @ lambdas
        softmin = _stable_softmin(scores, tau=tau)
        difference = lambdas - preferences
        penalty = float(difference @ relationships @ difference)
        return float(-(softmin - rho * penalty))

    return _minimize_on_simplex(
        objective=objective,
        initial=preferences,
        fallback=preferences,
        eps=eps,
        max_iterations=max_iterations,
    )


def relationship_softmax_mapping(
    p: ArrayLike1D,
    R: ArrayLike2D,
    tau: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the legacy relationship-softmax prototype mapping.

    The mapping uses ``scores = R @ p`` and
    ``lambda_i proportional to p_i * exp(tau * scores_i)``. Zero preference
    entries remain zero, and ``tau=0`` returns the normalized preference
    vector exactly.

    This helper is kept for compatibility with early prototype scripts. It is
    not the current thesis M1 definition, which is implemented by
    :func:`m1_mgda_inspired_mapping`.
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
        raise ValueError("The scaled relationship-softmax scores must be finite.")

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
    """Backward-compatible wrapper for the previous C1 name.

    The current thesis C1 definition is the trust-region CAGrad-inspired
    method. This wrapper maps the old ``rho`` argument to a trust-region size
    to keep older scripts runnable while new scripts call
    :func:`c1_trust_region_cagrad_mapping` directly.
    """
    return c1_trust_region_cagrad_mapping(
        p=p,
        R=R,
        c=float(rho),
        eps=eps,
        max_iterations=max_iterations,
    )


def _minimize_on_simplex(
    *,
    objective,
    initial: np.ndarray,
    fallback: np.ndarray,
    eps: float,
    max_iterations: int,
) -> np.ndarray:
    """Minimize a scalar objective over the probability simplex with SLSQP."""
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    initial = normalize_simplex(initial, eps=eps)
    fallback = normalize_simplex(fallback, eps=eps)
    objective_count = initial.size

    try:
        from scipy.optimize import minimize

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * objective_count,
            constraints=(
                {
                    "type": "eq",
                    "fun": lambda values: float(np.sum(values)) - 1.0,
                },
            ),
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return fallback
        candidate = normalize_simplex(np.clip(result.x, 0.0, None), eps=eps)
        if objective(candidate) > objective(fallback) + 1e-10:
            return fallback
        return candidate
    except Exception:
        return fallback


def _pcgrad_reconstruction_mapping(
    *,
    p: ArrayLike1D,
    R: ArrayLike2D,
    rho: float,
    eps: float,
    reverse_order: bool,
    max_iterations: int,
) -> np.ndarray:
    """Shared R-metric PCGrad projection and reconstruction implementation."""
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative.")

    preferences = validate_preference_vector(p, eps=max(eps, 1e-12))
    relationships = validate_relationship_matrix(
        R,
        num_objectives=preferences.size,
    )
    objective_count = preferences.size
    coefficient_directions = [
        preferences[index] * np.eye(objective_count, dtype=np.float64)[index]
        for index in range(objective_count)
    ]

    pairs = []
    for i in range(objective_count):
        for j in range(objective_count):
            if i == j:
                continue
            conflict = float(
                coefficient_directions[i]
                @ relationships
                @ coefficient_directions[j]
            )
            if conflict < 0:
                pairs.append((conflict, i, j))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    if reverse_order:
        pairs.reverse()

    for _, i, j in pairs:
        direction_i = coefficient_directions[i]
        direction_j = coefficient_directions[j]
        conflict = float(direction_i @ relationships @ direction_j)
        if conflict < 0:
            denominator = float(direction_j @ relationships @ direction_j) + eps
            if denominator > 0:
                coefficient_directions[i] = (
                    direction_i - (conflict / denominator) * direction_j
                )

    reconstructed_target = np.sum(coefficient_directions, axis=0)

    def objective(lambdas: np.ndarray) -> float:
        difference = lambdas - reconstructed_target
        preference_difference = lambdas - preferences
        return float(
            difference @ relationships @ difference
            + rho * np.dot(preference_difference, preference_difference)
        )

    return _minimize_on_simplex(
        objective=objective,
        initial=preferences,
        fallback=preferences,
        eps=max(eps, 1e-12),
        max_iterations=max_iterations,
    )


def _stable_softmin(values: np.ndarray, tau: float) -> float:
    """Compute ``-tau * log(sum(exp(-values / tau)))`` stably."""
    scaled = -np.asarray(values, dtype=np.float64) / tau
    shift = float(np.max(scaled))
    return float(-tau * (shift + np.log(np.sum(np.exp(scaled - shift)))))


def _is_trust_region_feasible(
    candidate: np.ndarray,
    preferences: np.ndarray,
    relationships: np.ndarray,
    trust_radius: float,
    eps: float,
) -> bool:
    """Check C1 simplex and trust-region feasibility."""
    if candidate.shape != preferences.shape:
        return False
    if np.any(candidate < -eps):
        return False
    if abs(float(candidate.sum()) - 1.0) > 1e-7:
        return False
    difference = candidate - preferences
    distance = float(difference @ relationships @ difference)
    return distance <= trust_radius + 1e-7


def _best_feasible_min_score_candidate(
    preferences: np.ndarray,
    relationships: np.ndarray,
    trust_radius: float,
    eps: float,
) -> np.ndarray:
    """Select the best simple feasible C1 fallback candidate."""
    objective_count = preferences.size
    candidates = [
        preferences.copy(),
        np.full(objective_count, 1.0 / objective_count, dtype=np.float64),
        *np.eye(objective_count, dtype=np.float64),
    ]
    feasible = [
        candidate
        for candidate in candidates
        if _is_trust_region_feasible(
            candidate,
            preferences,
            relationships,
            trust_radius,
            eps=eps,
        )
    ]
    if not feasible:
        return preferences.copy()
    values = [float(np.min(relationships @ candidate)) for candidate in feasible]
    return feasible[int(np.argmax(values))].copy()


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
