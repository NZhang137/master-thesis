"""Invariants for the v13 coefficient portfolio.

These tests encode theorems, not observations. Each one should fail loudly if a
future edit breaks the mathematics rather than merely changing a number.

Run: python -m pytest tests/test_coefficient_portfolio_v13.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_portfolio import (  # noqa: E402
    avg,
    c1_plus_plus,
    cert,
    fair_alpha_eps,
    floor_lp_at_p,
    improvements,
    m1_plus,
    maxmin_c,
    maxmin_center,
    run_portfolio,
    tangent_projection,
)

ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
M = 5


def conflict_free_R() -> np.ndarray:
    """Near-isotropic, entrywise positive R in the shape of the RS-PPO geometry.

    Off-diagonals in [0.164, 0.206] and a dominant eigenvalue share near 35
    percent, i.e. the regime in which the common-ascent floor collapses.

    The off-diagonals must NOT be constant. A perfectly equicorrelated matrix
    satisfies R 1 = const * 1, hence Pi0(R 1) = 0, which is the exact knife-edge
    where Fair also collapses and where d0 vanishes for uniform p. That case is
    real and is tested separately in ``equicorrelated_R``; using it as the generic
    fixture would silently test the degenerate branch instead of the intended one.
    """
    R = np.array([
        [1.0000, 0.2056, 0.1783, 0.1912, 0.1640],
        [0.2056, 1.0000, 0.1701, 0.1834, 0.2011],
        [0.1783, 0.1701, 1.0000, 0.1955, 0.1872],
        [0.1912, 0.1834, 0.1955, 1.0000, 0.1739],
        [0.1640, 0.2011, 0.1872, 0.1739, 1.0000],
    ])
    return 0.5 * (R + R.T)


def equicorrelated_R(rho: float = 0.19) -> np.ndarray:
    """Perfectly equicorrelated R, the degenerate case Pi0(R 1) = 0."""
    R = np.full((M, M), rho)
    np.fill_diagonal(R, 1.0)
    return R


def conflicting_R() -> np.ndarray:
    """R with genuine negative off-diagonals, so the floor need not collapse."""
    R = np.array([
        [1.00, -0.40, 0.10, 0.05, -0.20],
        [-0.40, 1.00, -0.30, 0.15, 0.05],
        [0.10, -0.30, 1.00, -0.25, 0.10],
        [0.05, 0.15, -0.25, 1.00, -0.35],
        [-0.20, 0.05, 0.10, -0.35, 1.00],
    ])
    return 0.5 * (R + R.T)


PREFERENCES = {
    "uniform": np.full(M, 1.0 / M),
    "quality_focused": np.array([0.15, 0.35, 0.25, 0.10, 0.15]),
    "dominant_helpfulness": np.array([0.5, 0.125, 0.125, 0.125, 0.125]),
    "only_verbosity": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
}
INTERIOR = {k: v for k, v in PREFERENCES.items() if np.all(v > 0)}
VERTICES = [np.eye(M)[i] for i in range(M)]


def assert_simplex(lam, label):
    assert lam is not None, f"{label}: no lambda returned"
    assert np.all(lam >= -1e-9), f"{label}: negative entries {lam}"
    assert abs(float(lam.sum()) - 1.0) < 1e-7, f"{label}: sums to {lam.sum()}"


# --------------------------------------------------------------- aliases ----

def test_aliases_are_the_same_objects():
    """The v13 names must not become a second implementation."""
    assert avg is m1_plus
    assert cert is c1_plus_plus


# ------------------------------------------------------------------ Avg ----

@pytest.mark.parametrize("name,p", list(PREFERENCES.items()))
def test_avg_returns_simplex(name, p):
    assert_simplex(avg(p, conflict_free_R(), 0.5), f"Avg/{name}")


def test_avg_is_globally_optimal_from_one_start():
    """The Avg objective is concave for positive definite R and rho > 0.

    A single start is therefore not a shortcut but sufficient. If this ever fails,
    either R stopped being positive definite or the objective changed sign.
    """
    from scipy.optimize import minimize

    R = conflict_free_R()
    assert np.linalg.eigvalsh(R).min() > 0
    p = PREFERENCES["quality_focused"]
    rho = 0.5

    def negative(x):
        return -(p @ R @ x - rho * (x - p) @ R @ (x - p))

    rng = np.random.default_rng(0)
    best = None
    for start in [p, np.full(M, 1.0 / M), *np.eye(M), *rng.dirichlet(np.ones(M), 20)]:
        res = minimize(negative, start, method="SLSQP", bounds=[(0.0, 1.0)] * M,
                       constraints=({"type": "eq", "fun": lambda x: x.sum() - 1.0},),
                       options={"maxiter": 800, "ftol": 1e-14})
        if res.success and (best is None or negative(res.x) < best):
            best = negative(res.x)
    assert negative(avg(p, R, rho)) <= best + 1e-9


# ----------------------------------------------------------------- Cert ----

@pytest.mark.parametrize("name,p", list(PREFERENCES.items()))
def test_cert_returns_p_on_collapsed_floor(name, p):
    """Prop 26 as a class result: on a trivial floor every exact method gives p."""
    R = conflict_free_R()
    _, collapsed = floor_lp_at_p(R, p)
    assert collapsed, "fixture is supposed to have a collapsed floor"
    lam, t_star = cert(p, R, 0.5, 1e-8)
    assert np.allclose(lam, p, atol=1e-6), f"Cert moved on a collapsed floor: {lam}"
    assert t_star <= 1e-9


def test_cert_agrees_with_the_floor_lp():
    """Two independent routes to the same statement must not diverge."""
    for R in (conflict_free_R(), conflicting_R()):
        for p in PREFERENCES.values():
            t_lp, collapsed = floor_lp_at_p(R, p)
            lam, t_cert = cert(p, R, 1.0, 1e-8)
            if collapsed:
                assert np.allclose(lam, p, atol=1e-6)
            assert t_cert <= max(t_lp, 0.0) + 1e-6


# ----------------------------------------------------------- MaxMin(c) -----

@pytest.mark.parametrize("name,p", list(INTERIOR.items()))
def test_maxmin_collapses_to_p_at_c_one(name, p):
    """At c >= 1 the ball contains the origin, so v = 0 is attainable.

    On a collapsed floor no v != 0 improves every axis, so v = 0 is optimal and
    MaxMin coincides with the certificate. This is Prop 30(a) made testable.
    """
    R = conflict_free_R()
    for c in (1.0, 1.5, 3.0):
        lam, t_star, status = maxmin_c(p, R, c)
        assert status.startswith("OK"), f"{name}/c={c}: {status}"
        assert np.linalg.norm(lam - p) < 1e-6, f"{name}/c={c} moved by {np.linalg.norm(lam - p):.2e}"


@pytest.mark.parametrize("name,p", list(INTERIOR.items()))
def test_maxmin_moves_strictly_below_c_one(name, p):
    """For c < 1 the ball excludes the origin, so the step cannot be zero."""
    R = conflict_free_R()
    lam, t_star, status = maxmin_c(p, R, 0.5)
    assert status.startswith("OK"), f"{name}: {status}"
    assert np.linalg.norm(lam - p) > 1e-6


@pytest.mark.parametrize("p", VERTICES)
def test_maxmin_is_infeasible_at_vertices_below_c_one(p):
    """At a vertex d0 points outward while only inward moves are admissible.

    The minimum R-distance from d0 to the admissible set equals ||d0||_R exactly,
    so the ball first touches at c = 1, in v = 0 alone. Reporting this as a solver
    failure, or silently returning p, would misread an empty feasible set as a
    collapse.
    """
    R = conflict_free_R()
    d0, d0_norm = maxmin_center(p, R)
    for c in (0.0, 0.25, 0.5, 0.9):
        lam, t_star, status = maxmin_c(p, R, c)
        assert status == "INFEASIBLE_BALL_MISSES_SIMPLEX", f"c={c}: got {status}"
        assert lam is None
    lam, t_star, status = maxmin_c(p, R, 1.0)
    assert status.startswith("OK") and np.linalg.norm(lam - p) < 1e-6


def test_maxmin_c_zero_is_exactly_d0_when_admissible():
    p = PREFERENCES["uniform"]
    R = conflict_free_R()
    d0, _ = maxmin_center(p, R)
    lam, t_star, status = maxmin_c(p, R, 0.0)
    assert status == "OK_EXACT_D0"
    assert np.allclose(lam, p + d0, atol=1e-9)


def test_maxmin_centre_lies_in_the_tangent_space():
    for p in PREFERENCES.values():
        d0, _ = maxmin_center(p, conflict_free_R())
        assert abs(float(d0.sum())) < 1e-12


def test_maxmin_step_shrinks_monotonically_in_c():
    """Larger c means more freedom to retreat towards the certificate."""
    p = PREFERENCES["dominant_helpfulness"]
    R = conflict_free_R()
    distances = []
    for c in (0.25, 0.5, 0.75, 0.9, 1.0):
        lam, _, status = maxmin_c(p, R, c)
        assert status.startswith("OK")
        distances.append(np.linalg.norm(lam - p))
    assert all(a >= b - 1e-7 for a, b in zip(distances, distances[1:])), distances


# ----------------------------------------------------- Fair(alpha, eps) ----

@pytest.mark.parametrize("name,p", list(PREFERENCES.items()))
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_fair_returns_simplex(name, p, alpha):
    lam, _, status = fair_alpha_eps(p, conflict_free_R(), alpha, 0.05)
    assert status == "OK", f"Fair(a={alpha})/{name}: {status}"
    assert_simplex(lam, f"Fair(a={alpha})/{name}")


@pytest.mark.parametrize("name,p", list(INTERIOR.items()))
def test_fair_escapes_the_collapse(name, p):
    """Prop 26 as corrected in v13.2: Fair is not a collapser.

    The exact floor is the singleton {p}, but the eps-relaxed floor is
    full-dimensional, so Fair must move. Listing Fair among the collapsers was
    the error this test exists to prevent from returning.
    """
    R = conflict_free_R()
    _, collapsed = floor_lp_at_p(R, p)
    assert collapsed
    lam, _, status = fair_alpha_eps(p, R, 1.0, 0.05)
    assert status == "OK"
    assert np.linalg.norm(lam - p) > 1e-6, "Fair collapsed onto p"


@pytest.mark.parametrize("name,p", list(INTERIOR.items()))
def test_fair_step_grows_with_eps(name, p):
    R = conflict_free_R()
    distances = []
    for eps in (0.01, 0.05, 0.2):
        lam, _, status = fair_alpha_eps(p, R, 1.0, eps)
        assert status == "OK"
        distances.append(np.linalg.norm(lam - p))
    assert all(a <= b + 1e-9 for a, b in zip(distances, distances[1:])), distances


def test_fair_objective_does_not_depend_on_p():
    """Fair is the fairness axis, and that axis is orthogonal to the preference.

    Its objective U_alpha(R v + eps) is a function of the displacement alone; p
    enters only through the simplex bounds. For interior preferences the
    displacement is therefore identical. This is a property of the definition, not
    a bug, but it means Fair is NOT preference-aware in the sense of RQ1 and must
    be described as such wherever lambda = f(p, R) is claimed.
    """
    R = conflict_free_R()
    displacements = []
    for p in INTERIOR.values():
        lam, _, status = fair_alpha_eps(p, R, 1.0, 0.05)
        assert status == "OK"
        displacements.append(lam - p)
    for other in displacements[1:]:
        assert np.allclose(displacements[0], other, atol=1e-6)


def test_fair_rejects_eps_zero():
    with pytest.raises(ValueError):
        fair_alpha_eps(PREFERENCES["uniform"], conflict_free_R(), 1.0, 0.0)


# ------------------------------------------------- cross-method invariant ---

def test_every_mover_degrades_at_least_one_axis_on_a_collapsed_floor():
    """The operational meaning of a trivial floor.

    If a method moves at all, some axis must lose. A mover with all improvements
    non-negative would mean the floor is not the singleton {p}, contradicting the
    certificate.
    """
    R = conflict_free_R()
    for p in INTERIOR.values():
        candidates = [avg(p, R, 0.5)]
        for c in (0.25, 0.5, 0.9):
            lam, _, status = maxmin_c(p, R, c)
            if status.startswith("OK"):
                candidates.append(lam)
        for alpha in (0.0, 1.0, 2.0):
            lam, _, status = fair_alpha_eps(p, R, alpha, 0.05)
            if status == "OK":
                candidates.append(lam)
        for lam in candidates:
            if np.linalg.norm(lam - p) <= 1e-8:
                continue
            assert float(np.min(improvements(p, R, lam))) < 1e-9


def test_tangent_projection_is_idempotent():
    rng = np.random.default_rng(3)
    v = rng.normal(size=M)
    once = tangent_projection(v)
    assert np.allclose(once, tangent_projection(once))
    assert abs(float(once.sum())) < 1e-12


# ------------------------------------------------------- run_portfolio ------

def test_run_portfolio_keeps_the_legacy_schema():
    """Old keys and their fields must survive; verifier checks depend on them."""
    p = PREFERENCES["quality_focused"]
    out = run_portfolio(p, conflict_free_R(), {"M1PLUS_RHO": 0.5, "C1PP_C": 0.5, "C1PP_EPS": 1e-8})
    for key in ("M1+", "C1++", "M1++", "P2++", "P3++"):
        assert key in out
        for field in ("lam", "returns_p", "min_improvement", "vector_safe", "l2_from_p"):
            assert field in out[key], f"{key} lost field {field}"
    assert not any(k.startswith(("MaxMin", "Fair")) for k in out), \
        "new methods must not appear without their grids"


def test_run_portfolio_adds_new_methods_only_when_configured():
    p = PREFERENCES["quality_focused"]
    out = run_portfolio(p, conflict_free_R(), {
        "M1PLUS_RHO": 0.5, "C1PP_C": 0.5, "C1PP_EPS": 1e-8,
        "MAXMIN_C_GRID": [0.5, 1.0],
        "FAIR_ALPHA_GRID": [1.0], "FAIR_EPS_GRID": [0.05],
    })
    assert "MaxMin(c=0.5)" in out and "MaxMin(c=1.0)" in out
    assert "Fair(alpha=1.0,eps=0.05)" in out
    assert out["MaxMin(c=1.0)"]["returns_p"] is True
    assert out["MaxMin(c=0.5)"]["returns_p"] is False


def test_certificate_text_is_not_attached_to_movers():
    """Only exact floor methods certify. Avg, MaxMin and Fair never do."""
    p = PREFERENCES["quality_focused"]
    out = run_portfolio(p, conflict_free_R(), {
        "M1PLUS_RHO": 0.5, "C1PP_C": 0.5, "C1PP_EPS": 1e-8,
        "MAXMIN_C_GRID": [1.0], "FAIR_ALPHA_GRID": [1.0], "FAIR_EPS_GRID": [0.05],
    })
    assert out["C1++"]["certificate"] is not None
    assert out["MaxMin(c=1.0)"]["certificate"] is None
    assert out["M1+"]["certificate"] is None


# ------------------------------------------------------ conflicting case ---

def test_methods_still_run_on_a_conflicting_geometry():
    """Guard against silently assuming the collapsed regime everywhere."""
    R = conflicting_R()
    p = PREFERENCES["uniform"]
    assert_simplex(avg(p, R, 0.5), "Avg/conflicting")
    lam, t_star = cert(p, R, 0.5, 1e-8)
    assert_simplex(lam, "Cert/conflicting")
    lam, _, status = fair_alpha_eps(p, R, 1.0, 0.05)
    if status == "OK":
        assert_simplex(lam, "Fair/conflicting")


# ------------------------------------------------- the degenerate knife-edge --

def test_equicorrelated_R_has_vanishing_ascent_direction():
    """Pi0(R 1) = 0 exactly when the off-diagonals are constant."""
    R = equicorrelated_R()
    assert np.allclose(tangent_projection(R @ np.ones(M)), 0.0, atol=1e-12)
    assert not np.allclose(tangent_projection(conflict_free_R() @ np.ones(M)), 0.0)


def test_maxmin_reports_degenerate_centre_on_equicorrelated_R():
    """For uniform p the ball centre d0 vanishes, so there is nothing to move along."""
    R = equicorrelated_R()
    p = np.full(M, 1.0 / M)
    d0, d0_norm = maxmin_center(p, R)
    assert d0_norm < 1e-12
    lam, t_star, status = maxmin_c(p, R, 0.5)
    assert status == "D0_DEGENERATE"
    assert np.allclose(lam, p)


def test_fair_collapses_when_the_escape_direction_vanishes():
    """Fair escapes the collapse only if Pi0(R 1) != 0.

    The eps-relaxation opens the floor, but opening it is useless when every
    admissible direction is orthogonal to the gain. This is the precise condition
    under which the v13.2 correction to Prop 26 does NOT apply, and it must be
    stated with the correction rather than left implicit.
    """
    R = equicorrelated_R()
    p = np.full(M, 1.0 / M)
    lam, _, status = fair_alpha_eps(p, R, 1.0, 0.05)
    assert status == "OK"
    assert np.linalg.norm(lam - p) < 1e-6, "Fair should not move when Pi0(R 1) = 0"


# --------------------------------------------- verifier schema contract -----

def test_infeasible_rows_are_schema_complete():
    """G4 in verify_rs_ppo_setup.py iterates over every row and reads the legacy
    fields unconditionally. A row that omits them would raise a KeyError far from
    its cause, so infeasible rows must carry the full field set.
    """
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0])   # vertex: MaxMin(c<1) is empty here
    out = run_portfolio(p, conflict_free_R(), {
        "M1PLUS_RHO": 0.5, "C1PP_C": 0.5, "C1PP_EPS": 1e-8,
        "MAXMIN_C_GRID": [0.5], "FAIR_ALPHA_GRID": [1.0], "FAIR_EPS_GRID": [0.05],
    })
    infeasible = [k for k, v in out.items() if v.get("infeasible")]
    assert infeasible, "fixture is supposed to produce an infeasible row"
    for key in infeasible:
        row = out[key]
        for field in ("lam", "returns_p", "min_improvement", "vector_safe",
                      "l2_from_p", "certificate"):
            assert field in row, f"{key} lost field {field}"
        assert row["lam"] is None, "an empty feasible set must not be reported as p"
        assert row["vector_safe"] is False


def test_g4_gate_survives_infeasible_rows():
    """Replay the verifier's scalar-gain sweep verbatim over a vertex preference."""
    R = conflict_free_R()
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    out = run_portfolio(p, R, {
        "M1PLUS_RHO": 0.5, "C1PP_C": 0.5, "C1PP_EPS": 1e-8,
        "MAXMIN_C_GRID": [0.5, 1.0], "FAIR_ALPHA_GRID": [1.0], "FAIR_EPS_GRID": [0.05],
    })
    for name, row in out.items():
        lam = np.asarray(row["lam"], dtype=float)
        gain = float(p @ R @ (lam - p))
        assert not (row["vector_safe"] and gain < -1e-8), name

