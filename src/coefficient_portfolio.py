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
    """Return the global floor-existence LP value and collapse flag.

    This p-agnostic test asks whether any simplex-tangent direction can improve
    all objectives. It ignores the current preference boundary. Use
    `floor_lp_at_p` for the per-preference certificate.
    """
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


def floor_lp_at_p(R: np.ndarray, p: np.ndarray, tol: float = 1e-9) -> tuple[float, bool]:
    """Return the p-aware floor LP value and collapse flag.

    Solves max t subject to (Rv)_i >= t, 1^T v = 0, v >= -p, and ||v||_1 <= 1.
    The extra v >= -p constraint is essential for boundary preferences such as
    one-hot "only_*" points; the global floor LP is not a valid certificate
    there.
    """
    matrix = np.asarray(R, dtype=float)
    pref = np.asarray(p, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("R must be square.")
    if pref.shape != (n,):
        raise ValueError(f"p must have shape {(n,)}, got {pref.shape}.")
    if np.any(pref < -1e-12) or not np.isclose(pref.sum(), 1.0):
        raise ValueError("p must lie on the simplex.")

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

    for i in range(n):
        row = np.zeros(2 * n + 1)
        row[i] = -1.0
        row[n + i] = 1.0
        a_ub.append(row)
        b_ub.append(float(pref[i]))

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


# --------------------------------------------------------------------------
# Nomenklatur v13. Die Namen unten sind die der Arbeit; die Funktionsnamen
# darueber bleiben unveraendert, weil Vorregistrierungen, report.json-Schluessel
# und SHA-Manifeste an ihnen haengen. Umbenennen wuerde diese Kette reissen.
#
#   Avg          <- m1_plus
#   Cert         <- c1_plus_plus
#   MaxMin(c)    <- maxmin_c          (neu, v13; es gab keine Entsprechung)
#   Fair(a,eps)  <- fair_alpha_eps    (neu, v13; es gab keine Entsprechung)
#   SignConf     <- p3_plus_plus      (Diagnostik; Entfernung vorgesehen)
# --------------------------------------------------------------------------


def tangent_projection(v: np.ndarray) -> np.ndarray:
    """Project onto the simplex tangent space {v : 1'v = 0}."""
    values = np.asarray(v, dtype=float)
    return values - values.mean()


def maxmin_center(p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the MaxMin ball centre d0 = Pi0(R p) and its R-norm.

    This centre is what distinguishes MaxMin(c) from Cert. Cert centres its
    trust region on p itself (radius^2 = c^2 p'Rp); MaxMin centres the ball on
    the preference-weighted ascent direction d0. That shift is exactly why the
    c-parameter runs the other way round than in vanilla CAGrad: c = 0 pins the
    step to d0 (aggressive), while c >= 1 makes the ball contain the origin, so
    v = 0 becomes attainable and the certificate is recovered.
    """
    d0 = tangent_projection(np.asarray(R, dtype=float) @ np.asarray(p, dtype=float))
    return d0, float(np.sqrt(max(d0 @ np.asarray(R, dtype=float) @ d0, 0.0)))


def maxmin_feasible_distance(
    p: np.ndarray,
    R: np.ndarray,
    d0: np.ndarray,
) -> float:
    """Smallest R-distance from d0 to {1'v = 0, p + v in simplex}.

    The MaxMin ball can miss the simplex entirely. At a vertex p = e_k the
    direction d0 points OUTWARD -- it raises the already-dominant component --
    while the only admissible moves go inward. The minimum distance is then
    exactly ||d0||_R, so the ball first touches the feasible set at c = 1, and
    touches it only in v = 0. MaxMin(c) is therefore EMPTY at vertices for every
    c < 1. An optimizer reports that as a convergence failure, which is the wrong
    diagnosis: it is an empty feasible set, not a hard one. Callers must be able
    to tell the two apart, so this test runs before the optimization.
    """
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    centre = np.asarray(d0, dtype=float)
    n = len(pref)
    res = minimize(
        lambda v: float((v - centre) @ matrix @ (v - centre)),
        np.zeros(n),
        jac=lambda v: 2.0 * matrix @ (v - centre),
        method="SLSQP",
        bounds=[(-float(x), 1.0 - float(x)) for x in pref],
        constraints=[{
            "type": "eq",
            "fun": lambda v: float(v.sum()),
            "jac": lambda v: np.ones(n),
        }],
        options={"maxiter": 2000, "ftol": 1e-16},
    )
    if not res.success:
        return float("inf")
    return float(np.sqrt(max(res.fun, 0.0)))


def maxmin_c(
    p: np.ndarray,
    R: np.ndarray,
    c: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float, str]:
    """MaxMin(c): max_v min_i (R v)_i over an R-ball centred on d0 = Pi0(R p).

    Solves, in epigraph form,

        max t  s.t.  R v >= t,  ||v - d0||_R <= c ||d0||_R,
                     1'v = 0,   p + v in the simplex.

    Returns ``(lam, t_star, status)``. ``lam`` is None when no admissible point
    exists; ``status`` distinguishes an empty feasible set from a solver failure,
    which a silent fallback to p would conflate with a genuine collapse.
    """
    if not np.isfinite(c) or c < 0:
        raise ValueError("c must be finite and non-negative.")
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    if matrix.shape != (n, n):
        raise ValueError("R must be square and match p.")
    if np.any(pref < -1e-12) or not np.isclose(pref.sum(), 1.0):
        raise ValueError("p must lie on the simplex.")

    d0, d0_norm = maxmin_center(pref, matrix)
    if d0_norm < tol:
        return pref.copy(), 0.0, "D0_DEGENERATE"

    radius = c * d0_norm

    # c = 0 pins the step to d0 exactly, so test that point directly. Doing this
    # BEFORE the general distance test matters: at c = 0 the admissible radius is
    # exactly zero, and the numerical minimum-distance solve returns a small
    # positive residual, which would report a feasible interior preference as
    # infeasible.
    if c <= tol:
        lam = pref + d0
        if np.any(lam < -1e-9) or np.any(lam > 1.0 + 1e-9):
            return None, float("nan"), "INFEASIBLE_BALL_MISSES_SIMPLEX"
        lam = np.clip(lam, 0.0, None)
        lam = lam / lam.sum()
        return lam, float(np.min(improvements(pref, matrix, lam))), "OK_EXACT_D0"

    # Scale the slack with ||d0||_R; an absolute epsilon is meaningless here
    # because the radius itself is measured in units of ||d0||_R.
    slack = max(1e-9, 1e-7 * d0_norm)
    if maxmin_feasible_distance(pref, matrix, d0) > radius + slack:
        return None, float("nan"), "INFEASIBLE_BALL_MISSES_SIMPLEX"

    bounds = [(-float(x), 1.0 - float(x)) for x in pref] + [(None, None)]
    cons = [
        {"type": "eq",
         "fun": lambda z: float(z[:n].sum()),
         "jac": lambda z: np.r_[np.ones(n), 0.0]},
        {"type": "ineq",
         "fun": lambda z: matrix @ z[:n] - z[-1],
         "jac": lambda z: np.hstack([matrix, -np.ones((n, 1))])},
        {"type": "ineq",
         "fun": lambda z: float(radius**2 - (z[:n] - d0) @ matrix @ (z[:n] - d0)),
         "jac": lambda z: np.r_[-2.0 * matrix @ (z[:n] - d0), 0.0]},
    ]

    def _clip_tangent(v):
        clipped = np.clip(v, [b[0] for b in bounds[:n]], [b[1] for b in bounds[:n]])
        return tangent_projection(clipped)

    starts = [np.zeros(n), d0, 0.5 * d0, (1.0 - c) * d0,
              _clip_tangent(d0), _clip_tangent(0.25 * d0)]
    best = None
    for start in starts:
        z0 = np.r_[start, float(np.min(matrix @ start))]
        res = minimize(lambda z: float(-z[-1]), z0,
                       jac=lambda z: np.r_[np.zeros(n), -1.0],
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-14})
        if not res.success or not np.all(np.isfinite(res.x)):
            continue
        v = res.x[:n]
        ball = float((v - d0) @ matrix @ (v - d0) - radius**2)
        lam = pref + v
        if ball > 1e-8 or abs(float(v.sum())) > 1e-7 or np.any(lam < -1e-9):
            continue
        value = float(np.min(matrix @ v))
        if best is None or value > best[0]:
            best = (value, lam)

    if best is None:
        return None, float("nan"), "SOLVER_FAILED"
    lam = np.clip(best[1], 0.0, None)
    lam = lam / lam.sum()
    return lam, float(np.min(improvements(pref, matrix, lam))), "OK"


def alpha_fair_utility(x: np.ndarray, alpha: float) -> float:
    """Atkinson/alpha-fair utility. alpha=0 utilitarian, 1 Nash, ->inf max-min.

    The domain depends on alpha and must not be guarded uniformly. U_0 is the
    plain sum and is defined on all of R; for 0 < alpha < 1 the exponent 1-alpha
    is positive, so x = 0 is admissible; only alpha >= 1 needs x > 0 strictly.
    Requiring positivity everywhere rejects the true optimum whenever a floor
    constraint is active, which is precisely the case at a simplex vertex.
    """
    values = np.asarray(x, dtype=float)
    if abs(alpha) < 1e-12:
        return float(np.sum(values))
    if alpha >= 1.0 - 1e-9:
        if np.any(values <= 0):
            return -np.inf
        if abs(alpha - 1.0) < 1e-9:
            return float(np.sum(np.log(values)))
    elif np.any(values < 0):
        return -np.inf
    return float(np.sum(values ** (1.0 - alpha)) / (1.0 - alpha))


def fair_alpha_eps(
    p: np.ndarray,
    R: np.ndarray,
    alpha: float,
    eps: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float, str]:
    """Fair(alpha, eps): max U_alpha(Delta + eps) over the eps-relaxed floor.

        max_v  U_alpha(R v + eps)
        s.t.   R v + eps >= 0,  1'v = 0,  p + v in the simplex.

    The eps-relaxation is the whole point. On a collapsed floor the exact
    programs return p, because F_p = {p}; admitting a bounded degradation of eps
    per axis makes F_p^eps full-dimensional, so Fair escapes the collapse
    whenever Pi0(R 1) != 0. Prop 26 was corrected in v13.2 for exactly this
    reason -- Fair had been listed as a collapser.

    alpha = 1 is the Nash point; there is no separate Nash method.
    Returns ``(lam, u_value, status)``.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive; eps = 0 is the exact floor (use Cert).")
    pref = np.asarray(p, dtype=float)
    matrix = np.asarray(R, dtype=float)
    n = len(pref)
    if matrix.shape != (n, n):
        raise ValueError("R must be square and match p.")
    if np.any(pref < -1e-12) or not np.isclose(pref.sum(), 1.0):
        raise ValueError("p must lie on the simplex.")

    # alpha = 0 makes the programme a linear one. Solve it as an LP rather than
    # with SLSQP: the optimum sits on the boundary of the relaxed floor, where a
    # gradient method has nothing to descend along.
    if abs(alpha) < 1e-12:
        res = linprog(
            -(np.ones(n) @ matrix),
            A_ub=-matrix, b_ub=np.full(n, eps),
            A_eq=np.ones((1, n)), b_eq=[0.0],
            bounds=[(-float(x), 1.0 - float(x)) for x in pref],
            method="highs",
        )
        if not res.success:
            return None, float("nan"), "SOLVER_FAILED"
        lam = np.clip(pref + res.x, 0.0, None)
        lam = lam / lam.sum()
        return lam, float(np.sum(matrix @ res.x + eps)), "OK"

    pad = 1e-10

    def negative_utility(v):
        shifted = matrix @ np.asarray(v, dtype=float) + eps
        if np.any(shifted <= pad):
            return 1e12
        return -alpha_fair_utility(shifted, alpha)

    def negative_jacobian(v):
        shifted = matrix @ np.asarray(v, dtype=float) + eps
        if np.any(shifted <= pad):
            return np.zeros(n)
        return -(matrix @ (shifted ** (-alpha)))

    bounds = [(-float(x), 1.0 - float(x)) for x in pref]
    cons = [
        {"type": "eq",
         "fun": lambda v: float(v.sum()),
         "jac": lambda v: np.ones(n)},
        {"type": "ineq",
         "fun": lambda v: matrix @ np.asarray(v, dtype=float) + eps,
         "jac": lambda v: matrix},
    ]
    d0 = tangent_projection(matrix @ pref)
    # alpha = 0 makes the objective linear, so the optimum sits on the boundary of
    # the relaxed floor and a single interior start is not enough; the extra scaled
    # and clipped starts exist for that case.
    def _clip_tangent(v):
        clipped = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
        return tangent_projection(clipped)

    ones_dir = tangent_projection(np.linalg.solve(matrix, np.ones(n)))
    starts = [np.zeros(n), 0.01 * d0, -0.01 * d0, 0.05 * d0,
              _clip_tangent(0.1 * ones_dir), _clip_tangent(-0.1 * ones_dir),
              _clip_tangent(0.5 * d0)]
    # At a vertex only one coordinate can decrease, so the interior starts above
    # sit on the boundary of the bounds and SLSQP stalls there. Adding the
    # feasible edge directions gives it somewhere to start from.
    for i in range(n):
        for j in range(n):
            if i != j and pref[j] > 1e-9:
                step = np.zeros(n)
                step[i] = min(0.01, 1.0 - float(pref[i]))
                step[j] = -min(0.01, float(pref[j]))
                starts.append(step)

    best = None
    for start in starts:
        res = minimize(negative_utility, start, jac=negative_jacobian,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-14})
        if not res.success or not np.all(np.isfinite(res.x)):
            continue
        v = res.x
        lam = pref + v
        if np.any(matrix @ v + eps < -1e-8) or np.any(lam < -1e-9) or abs(float(v.sum())) > 1e-7:
            continue
        value = -negative_utility(v)
        if best is None or value > best[0]:
            best = (value, lam)

    if best is None:
        return None, float("nan"), "SOLVER_FAILED"
    lam = np.clip(best[1], 0.0, None)
    lam = lam / lam.sum()
    return lam, float(best[0]), "OK"


def _infeasible_row(status: str, extra: dict[str, Any]) -> dict[str, Any]:
    """A portfolio row for a method whose feasible set is empty.

    Carries the full legacy field set so downstream consumers can iterate blindly.
    ``lam`` stays None on purpose: substituting p here would report an empty
    feasible set as a collapse, and those are different statements.
    """
    return {
        "lam": None,
        "returns_p": False,
        "min_improvement": float("nan"),
        "vector_safe": False,
        "l2_from_p": float("nan"),
        "certificate": None,
        "status": status,
        "infeasible": True,
        **extra,
    }


# Aliases in thesis nomenclature. Thin wrappers, no behaviour change.
avg = m1_plus
cert = c1_plus_plus


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
    entries = [
        ("M1+", lam_m1p, {"scalar_gain_pT_R_dlam": float(pref @ matrix @ (lam_m1p - pref))}),
        ("C1++", lam_c1, {"t_star": t_c1}),
        ("M1++", lam_m1pp, {"t_star": t_m1}),
        ("P2++", lam_p2, {"t_star": t_p2, "surgeries_fired": fired}),
        ("P3++", lam_p3, {"conflict_energy": c_val, "indifferent_C_is_zero": indiff}),
    ]

    # Additive v13 extension. Existing keys and their schema are untouched, because
    # notebook outputs and verifier checks depend on them. New methods appear only
    # when their grids are configured, so old cfg dicts keep working unchanged.
    for c_value in cfg.get("MAXMIN_C_GRID", []):
        lam_mm, t_mm, status_mm = maxmin_c(pref, matrix, float(c_value))
        out_key = f"MaxMin(c={c_value})"
        if lam_mm is None:
            # Schema-complete even when there is no lambda. Consumers iterate over
            # every row and read the legacy fields unconditionally (the verifier's
            # G4 gate does exactly that), so a short row would raise a KeyError far
            # away from its cause. vector_safe=False makes the G4 guard
            # short-circuit before the NaN gain is ever compared.
            out[out_key] = _infeasible_row(status_mm, {"t_star": None, "c": float(c_value)})
            continue
        entries.append((out_key, lam_mm, {"t_star": t_mm, "status": status_mm, "c": float(c_value)}))

    for alpha_value in cfg.get("FAIR_ALPHA_GRID", []):
        for eps_value in cfg.get("FAIR_EPS_GRID", []):
            lam_f, u_f, status_f = fair_alpha_eps(pref, matrix, float(alpha_value), float(eps_value))
            out_key = f"Fair(alpha={alpha_value},eps={eps_value})"
            if lam_f is None:
                out[out_key] = _infeasible_row(status_f, {
                    "u_alpha": None, "alpha": float(alpha_value), "eps": float(eps_value)})
                continue
            entries.append((out_key, lam_f, {"u_alpha": u_f, "status": status_f,
                                             "alpha": float(alpha_value), "eps": float(eps_value)}))

    for name, lam, extra in entries:
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
            ) if (returns_p and not name.startswith(("M1+", "MaxMin", "Fair"))) else None,
            **extra,
        }
    return out


def paired_rank_delta(heads_p: np.ndarray, heads_l: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
    """Rank-normalized paired linear-utility gain Delta U_p with pooled ranks.

    NOT the primary metric any more. Since v15 the primary is the RAW
    preference-weighted sum under the identity declaration r~_i := r_i; see
    ``src.proxy_validation.preference_utility``. Rank normalization is retained
    as a robustness comparison, which is what this function now serves.
    """
    left = np.asarray(heads_p, dtype=float)
    right = np.asarray(heads_l, dtype=float)
    pref = np.asarray(p_vec, dtype=float)
    n = left.shape[0]
    delta_rank = np.zeros(n)
    for j in range(left.shape[1]):
        pooled = rankdata(np.r_[left[:, j], right[:, j]]) / (2 * n)
        delta_rank += pref[j] * (pooled[n:] - pooled[:n])
    return delta_rank
