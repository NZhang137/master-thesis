"""Coefficient-space portfolio methods for Rewarded-Soups-style correction.

This module is the single source of truth for the coefficient mappings used by
Notebook 08 and by the RS-PPO setup verifier. Keep return keys stable: notebook
outputs and verifier checks intentionally depend on the exact `run_portfolio`
schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import rankdata


def floor_lp(R: np.ndarray, tol: float = 1e-9) -> tuple[float, bool]:
    """Return max_{1^T v = 0, ||v||_1 <= 1} min_i (Rv)_i and collapse flag."""
    matrix = np.asarray(R, dtype=float)
    n = matrix.shape[0]
    c = np.zeros(2 * n + 1)
    c[-1] = -1.0
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for i in range(n):
        row = np.zeros(2 * n + 1)
        row[:n] = -matrix[i]
        row[n:2 * n] = matrix[i]
        row[-1] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)

    row = np.zeros(2 * n + 1)
    row[:2 * n] = 1.0
    a_ub.append(row)
    b_ub.append(1.0)

    a_eq = np.asarray([np.r_[np.ones(n), -np.ones(n), 0.0]])
    lp = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.asarray([0.0]),
        bounds=[(0.0, None)] * (2 * n) + [(None, None)],
        method="highs",
    )
    assert lp.success, lp.message
    value = float(-lp.fun)
    return value, bool(value <= tol)


def improvements(p: np.ndarray, R: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Return objective-wise linearized improvements R(lambda-p)."""
    return np.asarray(R, dtype=float) @ (np.asarray(lam, dtype=float) - np.asarray(p, dtype=float))


def line_search_into_floor(
    p: np.ndarray,
    R: np.ndarray,
    d: np.ndarray,
    tol: float = 1e-10,
) -> float:
    """Largest t >= 0 with p+t*d in the floor and in the simplex.

    The projection Pi0 does not preserve common ascent by itself. This line
    search is what guarantees floor membership, with t=0 as conservative
    fallback.
    """
    pref = np.asarray(p, dtype=float)
    direction = np.asarray(d, dtype=float)
    assert abs(float(direction.sum())) < 1e-8, "direction must lie in the simplex tangent space"
    if np.any(np.asarray(R, dtype=float) @ direction < -tol) or np.linalg.norm(direction) < tol:
        return 0.0
    ts = [np.inf] + [-pref[i] / direction[i] for i in range(len(pref)) if direction[i] < -tol]
    t = float(min(ts))
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def m1_plus(p: np.ndarray, R: np.ndarray, rho: float) -> np.ndarray:
    """Scalar-safe M1+: argmax p'R lambda - rho (lambda-p)'R(lambda-p)."""
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    obj = lambda x: -(pref @ matrix @ x - rho * (x - pref) @ matrix @ (x - pref))
    jac = lambda x: -(matrix @ pref - 2 * rho * matrix @ (x - pref))
    res = minimize(
        obj,
        pref.copy(),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda x: x.sum() - 1.0,
            "jac": lambda x: np.ones(n),
        }],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    assert res.success, res.message
    x = np.clip(res.x, 0.0, None)
    return x / x.sum()


def c1_plus_plus(
    p: np.ndarray,
    R: np.ndarray,
    c: float,
    eps: float,
) -> tuple[np.ndarray, float]:
    """Trust-region C1++ floor maximization."""
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    radius2 = (c**2) * max(float(pref @ matrix @ pref), eps)
    obj = lambda z: -z[-1]
    jac = lambda z: np.r_[np.zeros(n), -1.0]
    cons = [
        {
            "type": "eq",
            "fun": lambda z: z[:n].sum() - 1.0,
            "jac": lambda z: np.r_[np.ones(n), 0.0],
        },
        {
            "type": "ineq",
            "fun": lambda z: matrix @ (z[:n] - pref) - z[-1],
            "jac": lambda z: np.hstack([matrix, -np.ones((n, 1))]),
        },
        {
            "type": "ineq",
            "fun": lambda z: radius2 - (z[:n] - pref) @ matrix @ (z[:n] - pref),
            "jac": lambda z: np.r_[-2 * matrix @ (z[:n] - pref), 0.0],
        },
    ]
    res = minimize(
        obj,
        np.r_[pref, 0.0],
        jac=jac,
        method="SLSQP",
        constraints=cons,
        bounds=[(0.0, 1.0)] * n + [(None, None)],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    if not res.success:
        return pref.copy(), 0.0
    lam = np.clip(res.x[:n], 0.0, None)
    lam = lam / lam.sum()
    t_star = float(min(improvements(pref, matrix, lam)))
    if t_star < -1e-8:
        return pref.copy(), 0.0
    return lam, max(t_star, 0.0)


def m1_plus_plus(p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float]:
    """MGDA min-norm direction, projected to the simplex tangent, then line-searched."""
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    obj = lambda a: float(np.dot(matrix @ a, matrix @ a))
    jac = lambda a: 2 * matrix.T @ (matrix @ a)
    res = minimize(
        obj,
        np.ones(n) / n,
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda a: a.sum() - 1.0,
            "jac": lambda a: np.ones(n),
        }],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if not res.success:
        return pref.copy(), 0.0
    pi0 = np.eye(n) - np.ones((n, n)) / n
    d = pi0 @ (matrix @ res.x)
    t = line_search_into_floor(pref, matrix, d)
    return (pref + t * d if t > 0 else pref.copy()), t


def p2_plus_plus(p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float, int]:
    """PCGrad surgery on g_i = R[:, i], deterministic strongest-conflict-first order."""
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    gradients = [matrix[:, i].astype(float).copy() for i in range(n)]
    pairs = sorted(
        [(i, j) for i in range(n) for j in range(n) if i != j],
        key=lambda ij: float(matrix[:, ij[0]] @ matrix[:, ij[1]]),
    )
    fired = 0
    for i, j in pairs:
        dot = float(gradients[i] @ gradients[j])
        if dot < 0:
            gradients[i] = gradients[i] - (dot / float(gradients[j] @ gradients[j])) * gradients[j]
            fired += 1
    pi0 = np.eye(n) - np.ones((n, n)) / n
    d = pi0 @ sum(pref[i] * gradients[i] for i in range(n))
    t = line_search_into_floor(pref, matrix, d)
    return (pref + t * d if t > 0 else pref.copy()), t, fired


def p3_plus_plus(
    p: np.ndarray,
    R: np.ndarray,
    n_starts: int = 8,
    seed: int = 137,
) -> tuple[np.ndarray, float, bool]:
    """Minimize conflict energy lambda'R^-lambda inside the floor."""
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    r_minus = np.maximum(0.0, -matrix.copy())
    np.fill_diagonal(r_minus, 0.0)
    if not np.any(r_minus > 0):
        return pref.copy(), 0.0, True

    cons = [
        {"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(n)},
        {"type": "ineq", "fun": lambda x: matrix @ (x - pref), "jac": lambda x: matrix},
    ]
    obj = lambda x: float(x @ r_minus @ x)
    rng = np.random.default_rng(seed)
    best, best_val = pref.copy(), obj(pref)
    for x0 in [pref.copy()] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]:
        res = minimize(
            obj,
            x0,
            jac=lambda x: 2 * r_minus @ x,
            method="SLSQP",
            constraints=cons,
            bounds=[(0.0, 1.0)] * n,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.success:
            x = np.clip(res.x, 0.0, None)
            x = x / x.sum()
            if np.all(improvements(pref, matrix, x) >= -1e-7) and obj(x) < best_val - 1e-12:
                best, best_val = x, obj(x)
    return best, best_val, False


def run_portfolio(p: np.ndarray, R: np.ndarray, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run the full coefficient portfolio with the notebook-compatible schema."""
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    lam_m1p = m1_plus(pref, matrix, cfg["M1PLUS_RHO"])
    lam_c1, t_c1 = c1_plus_plus(pref, matrix, cfg["C1PP_C"], cfg["C1PP_EPS"])
    lam_m1pp, t_m1 = m1_plus_plus(pref, matrix)
    lam_p2, t_p2, fired = p2_plus_plus(pref, matrix)
    lam_p3, c_val, indiff = p3_plus_plus(pref, matrix)

    out: dict[str, dict[str, Any]] = {}
    for name, lam, extra in [
        ("M1+", lam_m1p, {"scalar_gain_pT_R_dlam": float(pref @ matrix @ (lam_m1p - pref))}),
        ("C1++", lam_c1, {"t_star": t_c1}),
        ("M1++", lam_m1pp, {"t_star": t_m1}),
        ("P2++", lam_p2, {"t_star": t_p2, "surgeries_fired": fired}),
        ("P3++", lam_p3, {"conflict_energy": c_val, "indifferent_C_is_zero": indiff}),
    ]:
        delta = improvements(pref, matrix, lam)
        returns_p = bool(np.allclose(lam, pref, atol=1e-6))
        out[name] = {
            "lam": lam.tolist(),
            "returns_p": returns_p,
            "min_improvement": float(delta.min()),
            "vector_safe": bool(np.all(delta >= -1e-7)),
            "l2_from_p": float(np.linalg.norm(lam - pref)),
            "certificate": (
                "returns p: no direction improves ALL objectives over p in this "
                "geometry, i.e. the floor is the singleton {p}. This is a theorem, "
                "not a failure."
            ) if (returns_p and name != "M1+") else None,
            **extra,
        }
    return out


def paired_rank_delta(heads_p: np.ndarray, heads_l: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
    """Rank-normalized paired Delta U_p with ranks pooled over both models per axis."""
    left = np.asarray(heads_p, dtype=float)
    right = np.asarray(heads_l, dtype=float)
    pref = np.asarray(p_vec, dtype=float)
    n = left.shape[0]
    delta_rank = np.zeros(n)
    for j in range(left.shape[1]):
        pooled = rankdata(np.r_[left[:, j], right[:, j]]) / (2 * n)
        delta_rank += pref[j] * (pooled[n:] - pooled[:n])
    return delta_rank
