"""Verify the RS-PPO/ArmoRM setup before expensive runs.

The script is intentionally conservative: it performs no training and exits
with code 0 only when every selected check passes. GPU/model stages are
explicitly marked SKIP when prerequisites are missing, which still blocks
training for the selected stage.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "results" / "verify_rs_ppo_setup" / "report.json"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_rs_ppo_armorm_circular_colab.ipynb"
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_rs_ppo.py"


@dataclass
class CheckResult:
    """One PASS/FAIL/SKIP result row."""

    check_id: str
    stage: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def read_text(path: Path) -> str:
    """Read UTF-8 text from a repository file."""
    return path.read_text(encoding="utf-8")


def notebook_cells() -> list[dict[str, Any]]:
    """Load notebook cells."""
    nb = json.loads(read_text(NOTEBOOK_PATH))
    return list(nb.get("cells", []))


def notebook_source() -> str:
    """Return all notebook source text concatenated."""
    chunks: list[str] = []
    for cell in notebook_cells():
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def notebook_config() -> dict[str, Any]:
    """Extract the literal CONFIG dict from the notebook setup cell."""
    for cell in notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        filtered = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            tree = ast.parse(filtered)
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets)
            ):
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise TypeError("Notebook CONFIG is not a dict.")
                return value
    raise RuntimeError("Could not find literal CONFIG dict in notebook.")


def write_report(results: list[CheckResult]) -> None:
    """Write report.json."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
        "summary": {
            "pass": sum(result.status == "PASS" for result in results),
            "fail": sum(result.status == "FAIL" for result in results),
            "skip": sum(result.status == "SKIP" for result in results),
        },
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_table(results: list[CheckResult]) -> None:
    """Print compact console table."""
    print("\nRS-PPO setup verification")
    print("=" * 88)
    print(f"{'CHECK':<6} {'STAGE':<5} {'STATUS':<6} MESSAGE")
    print("-" * 88)
    for result in results:
        print(f"{result.check_id:<6} {result.stage:<5} {result.status:<6} {result.message}")
    print("-" * 88)
    print(f"Report: {REPORT_PATH}")


def run_check(
    results: list[CheckResult],
    check_id: str,
    stage: str,
    fn: Callable[[], tuple[str, dict[str, Any] | None]],
) -> None:
    """Run one check and append a result."""
    t0 = time.time()
    try:
        message, details = fn()
        status = "PASS"
    except SkipCheck as exc:
        status = "SKIP"
        message = str(exc)
        details = exc.details
    except Exception as exc:  # noqa: BLE001 - verifier must capture all failures.
        status = "FAIL"
        message = str(exc)
        details = {"exception_type": type(exc).__name__}
    results.append(
        CheckResult(
            check_id=check_id,
            stage=stage,
            status=status,
            message=message,
            details=details or {},
            elapsed_s=time.time() - t0,
        )
    )


class SkipCheck(RuntimeError):
    """A skipped check with structured details."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


def import_train_module():
    """Import and reload train_rs_ppo after ensuring project root is importable."""
    module = importlib.import_module("scripts.train_rs_ppo")
    return importlib.reload(module)


def train_tree() -> ast.Module:
    """Parse train_rs_ppo.py once for static checks."""
    return ast.parse(read_text(TRAIN_SCRIPT_PATH), filename=str(TRAIN_SCRIPT_PATH))


def selected_train_namespace(names: set[str]) -> dict[str, Any]:
    """Execute selected train_rs_ppo definitions without importing torch/models."""
    from src.helpsteer2_utils import HELPSTEER2_ATTRIBUTES

    tree = train_tree()
    selected: list[ast.stmt] = []
    wanted = set(names)
    for node in tree.body:
        node_names: set[str] = set()
        if isinstance(node, ast.Assign):
            node_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_names = {node.name}
        elif isinstance(node, ast.Assert):
            if "ATTRIBUTES" in wanted:
                node_names = {"ATTRIBUTES"}
        if node_names & wanted:
            selected.append(node)

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "HELPSTEER2_ATTRIBUTES": HELPSTEER2_ATTRIBUTES,
        "warnings": __import__("warnings"),
    }
    exec(compile(module, str(TRAIN_SCRIPT_PATH), "exec"), namespace)
    return namespace


def check_s1_axis_order() -> tuple[str, dict[str, Any]]:
    """S1: one source of truth for axis order."""
    from src.helpsteer2_utils import HELPSTEER2_ATTRIBUTES

    train_ns = selected_train_namespace({"ATTRIBUTES"})
    nb_config = notebook_config()
    orders = {
        "train_rs_ppo.ATTRIBUTES": list(train_ns["ATTRIBUTES"]),
        "HELPSTEER2_ATTRIBUTES": list(HELPSTEER2_ATTRIBUTES),
        "notebook.CONFIG.ATTRIBUTES": list(nb_config["ATTRIBUTES"]),
    }
    if not (
        tuple(train_ns["ATTRIBUTES"])
        == tuple(HELPSTEER2_ATTRIBUTES)
        == tuple(nb_config["ATTRIBUTES"])
    ):
        raise AssertionError(f"axis order drift: {orders}")
    return "Axis order is shared by train script, HelpSteer2 utils, and notebook.", orders


def check_s2_no_policy_save() -> tuple[str, dict[str, Any]]:
    """S2: policy.save_pretrained must not appear."""
    source = read_text(TRAIN_SCRIPT_PATH)
    count = source.count("policy.save_pretrained(")
    if count:
        raise AssertionError("policy.save_pretrained( still appears in train_rs_ppo.py")
    return "No policy.save_pretrained( call found.", {"occurrences": count}


def check_s3_device_maps() -> tuple[str, dict[str, Any]]:
    """S3: PPO policy constructor has no device_map, ArmoRM loader does."""
    tree = train_tree()
    policy_calls: list[ast.Call] = []
    armorm_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                name = f"{func.value.id}.{func.attr}"
            elif isinstance(func.value, ast.Attribute):
                name = f"{func.value.attr}.{func.attr}"
        if name == "AutoModelForCausalLMWithValueHead.from_pretrained":
            policy_calls.append(node)
        if name == "AutoModelForSequenceClassification.from_pretrained":
            armorm_calls.append(node)
    if not policy_calls:
        raise AssertionError("PPO policy constructor not found.")
    if any(keyword.arg == "device_map" for call in policy_calls for keyword in call.keywords):
        raise AssertionError("PPO policy constructor still has a device_map keyword.")
    if not armorm_calls:
        raise AssertionError("ArmoRM AutoModelForSequenceClassification loader not found.")
    if not any(keyword.arg == "device_map" for call in armorm_calls for keyword in call.keywords):
        raise AssertionError("ArmoRM loader does not expose device_map.")
    return "Policy has no device_map; ArmoRM loader keeps device_map.", {
        "policy_calls": len(policy_calls),
        "armorm_calls": len(armorm_calls),
    }


def check_s4_apply_overrides() -> tuple[str, dict[str, Any]]:
    """S4: apply_overrides mutates the expected CFG keys."""
    train_ns = selected_train_namespace({"ARMORM_MODEL", "CFG", "OVERRIDABLE", "apply_overrides"})
    cfg = train_ns["CFG"]
    apply_overrides = train_ns["apply_overrides"]
    old = dict(cfg)
    overrides = {
        "out_dir": "results/verify_rs_ppo_setup/_tmp_rs_runs",
        "batch_size": 17,
        "total_ppo_steps": 19,
        "n_prompts": 23,
    }
    try:
        returned = apply_overrides(**overrides)
        for key, value in overrides.items():
            if cfg[key] != value or returned[key] != value:
                raise AssertionError(f"apply_overrides failed for {key!r}")
    finally:
        cfg.clear()
        cfg.update(old)
    return "apply_overrides writes and exposes out_dir/batch_size/steps/prompts.", overrides


def check_s5_firewall() -> tuple[str, dict[str, Any]]:
    """S5: circular ArmoRM firewall must block unless acknowledged."""
    train_ns = selected_train_namespace({
        "BASE_MODEL",
        "ARMORM_MODEL",
        "ATTRIBUTES",
        "CIRCULARITY_WARNING",
        "is_armorm_model",
        "check_reward_firewall",
    })
    check_reward_firewall = train_ns["check_reward_firewall"]
    blocked = False
    try:
        check_reward_firewall(
            "helpfulness",
            train_ns["ARMORM_MODEL"],
            circular_armorm_acknowledged=False,
        )
    except AssertionError:
        blocked = True
    if not blocked:
        raise AssertionError("Firewall did not block unacknowledged ArmoRM PPO.")
    acknowledged = check_reward_firewall(
        "helpfulness",
        train_ns["ARMORM_MODEL"],
        circular_armorm_acknowledged=True,
    )
    retired = acknowledged.get("retired_research_questions", [])
    if "RQ2 (proxy validity)" not in retired:
        raise AssertionError(f"Acknowledged firewall did not retire RQ2: {acknowledged}")
    return "Firewall blocks unacknowledged ArmoRM and retires RQ2 when acknowledged.", acknowledged


def check_s6_preregistration_consistency() -> tuple[str, dict[str, Any]]:
    """S6: decision metrics must be named in preregistration source."""
    source = notebook_source()
    required = {
        "delta_U_p_mean_decision": "Delta U_p",
        "ci_excludes_zero_decision": "bootstrap 95% CI",
        "wall_a_threshold": "wall_A_R2_threshold",
        "wall_a_rule": "Wall A stands iff max R2 over the quality axes < threshold.",
    }
    missing = [name for name, needle in required.items() if needle not in source]
    if missing:
        raise AssertionError(f"Preregistration source is missing: {missing}")
    merge_required = ["delta_U_p_mean", "ci_excludes_zero", "WALL_A_R2_THRESHOLD"]
    missing_merge = [needle for needle in merge_required if needle not in source]
    if missing_merge:
        raise AssertionError(f"Notebook decision code is missing expected names: {missing_merge}")
    return "Decision metrics are named in the preregistration source.", {
        "required_preregistration_terms": required,
    }


def check_s7_equal_n() -> tuple[str, dict[str, Any]]:
    """S7: seeds and step count are global, not per-axis."""
    train_ns = selected_train_namespace({"ARMORM_MODEL", "CFG"})
    cfg = train_ns["CFG"]
    values = {
        "prompt_seed": cfg["prompt_seed"],
        "train_seed": cfg["train_seed"],
        "total_ppo_steps": cfg["total_ppo_steps"],
    }
    if not all(isinstance(value, int) for value in values.values()):
        raise AssertionError(f"Equal-N keys must be integer scalars: {values}")
    source = read_text(TRAIN_SCRIPT_PATH)
    forbidden_patterns = [
        "total_ppo_steps_by_axis",
        "prompt_seed_by_axis",
        "train_seed_by_axis",
        "per_axis_steps",
        "per_axis_seed",
    ]
    found = [pattern for pattern in forbidden_patterns if pattern in source]
    if found:
        raise AssertionError(f"Per-axis Equal-N override pattern found: {found}")
    return "prompt_seed, train_seed, and total_ppo_steps are global scalar settings.", values


# ---------------------------------------------------------------------------
# Stage 1: synthetic geometry / portfolio
# ---------------------------------------------------------------------------


PI0 = lambda m: np.eye(m) - np.ones((m, m)) / m


def floor_lp(R: np.ndarray, tol: float = 1e-9) -> tuple[float, bool]:
    """max_{1^T v = 0, ||v||_1 <= 1} min_i (Rv)_i."""
    from scipy.optimize import linprog

    n = R.shape[0]
    c = np.zeros(2 * n + 1)
    c[-1] = -1.0
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(2 * n + 1)
        row[:n] = -R[i]
        row[n:2 * n] = R[i]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    row = np.zeros(2 * n + 1)
    row[:2 * n] = 1.0
    A_ub.append(row)
    b_ub.append(1.0)
    A_eq = np.asarray([np.r_[np.ones(n), -np.ones(n), 0.0]])
    lp = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=A_eq,
        b_eq=np.asarray([0.0]),
        bounds=[(0.0, None)] * (2 * n) + [(None, None)],
        method="highs",
    )
    if not lp.success:
        raise AssertionError(lp.message)
    value = float(-lp.fun)
    return value, bool(value <= tol)


def improvements(p: np.ndarray, R: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Return R(lam-p)."""
    return R @ (np.asarray(lam, float) - np.asarray(p, float))


def line_search_into_floor(p: np.ndarray, R: np.ndarray, d: np.ndarray, tol: float = 1e-10) -> float:
    """Largest t >= 0 such that p+t*d remains in the floor and simplex."""
    p = np.asarray(p, float)
    d = np.asarray(d, float)
    if abs(float(d.sum())) > 1e-8:
        raise AssertionError("direction must lie in simplex tangent space")
    if np.any(R @ d < -tol) or np.linalg.norm(d) < tol:
        return 0.0
    ts = [np.inf] + [-p[i] / d[i] for i in range(len(p)) if d[i] < -tol]
    t = float(min(ts))
    return 0.0 if not np.isfinite(t) else max(0.0, t)


def m1_plus(p: np.ndarray, R: np.ndarray, rho: float) -> np.ndarray:
    """Scalar-safe M1+ mapping."""
    from scipy.optimize import minimize

    p = np.asarray(p, float)
    n = len(p)
    obj = lambda x: -(p @ R @ x - rho * (x - p) @ R @ (x - p))
    jac = lambda x: -(R @ p - 2 * rho * R @ (x - p))
    res = minimize(
        obj,
        p.copy(),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(n)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise AssertionError(res.message)
    x = np.clip(res.x, 0.0, None)
    return x / x.sum()


def c1_plus_plus(p: np.ndarray, R: np.ndarray, c: float, eps: float) -> tuple[np.ndarray, float]:
    """C1++ with trust region."""
    from scipy.optimize import minimize

    p = np.asarray(p, float)
    n = len(p)
    radius2 = (c ** 2) * max(float(p @ R @ p), eps)
    obj = lambda z: -z[-1]
    jac = lambda z: np.r_[np.zeros(n), -1.0]
    cons = [
        {"type": "eq", "fun": lambda z: z[:n].sum() - 1.0, "jac": lambda z: np.r_[np.ones(n), 0.0]},
        {"type": "ineq", "fun": lambda z: R @ (z[:n] - p) - z[-1],
         "jac": lambda z: np.hstack([R, -np.ones((n, 1))])},
        {"type": "ineq", "fun": lambda z: radius2 - (z[:n] - p) @ R @ (z[:n] - p),
         "jac": lambda z: np.r_[-2 * R @ (z[:n] - p), 0.0]},
    ]
    res = minimize(
        obj,
        np.r_[p, 0.0],
        jac=jac,
        method="SLSQP",
        constraints=cons,
        bounds=[(0.0, 1.0)] * n + [(None, None)],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    if not res.success:
        return p.copy(), 0.0
    lam = np.clip(res.x[:n], 0.0, None)
    lam = lam / lam.sum()
    t_star = float(min(improvements(p, R, lam)))
    if t_star < -1e-8:
        return p.copy(), 0.0
    return lam, max(t_star, 0.0)


def m1_plus_plus(p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float]:
    """M1++ min-norm direction followed by line search."""
    from scipy.optimize import minimize

    p = np.asarray(p, float)
    n = len(p)
    obj = lambda a: float(np.dot(R @ a, R @ a))
    jac = lambda a: 2 * R.T @ (R @ a)
    res = minimize(
        obj,
        np.ones(n) / n,
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1.0, "jac": lambda a: np.ones(n)}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if not res.success:
        return p.copy(), 0.0
    d = PI0(n) @ (R @ res.x)
    t = line_search_into_floor(p, R, d)
    return (p + t * d if t > 0 else p.copy()), t


def p2_plus_plus(p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float, int]:
    """P2++ PCGrad surgery followed by line search."""
    p = np.asarray(p, float)
    n = len(p)
    G = [R[:, i].astype(float).copy() for i in range(n)]
    pairs = sorted(
        [(i, j) for i in range(n) for j in range(n) if i != j],
        key=lambda ij: float(R[:, ij[0]] @ R[:, ij[1]]),
    )
    fired = 0
    for i, j in pairs:
        dot = float(G[i] @ G[j])
        if dot < 0:
            G[i] = G[i] - (dot / float(G[j] @ G[j])) * G[j]
            fired += 1
    d = PI0(n) @ sum(p[i] * G[i] for i in range(n))
    t = line_search_into_floor(p, R, d)
    return (p + t * d if t > 0 else p.copy()), t, fired


def p3_plus_plus(p: np.ndarray, R: np.ndarray, n_starts: int = 8) -> tuple[np.ndarray, float, bool]:
    """P3++ conflict-energy minimization in the floor."""
    from scipy.optimize import minimize

    p = np.asarray(p, float)
    n = len(p)
    Rm = np.maximum(0.0, -R.copy())
    np.fill_diagonal(Rm, 0.0)
    if not np.any(Rm > 0):
        return p.copy(), 0.0, True
    cons = [
        {"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(n)},
        {"type": "ineq", "fun": lambda x: R @ (x - p), "jac": lambda x: R},
    ]
    obj = lambda x: float(x @ Rm @ x)
    rng = np.random.default_rng(137)
    best, best_val = p.copy(), obj(p)
    for x0 in [p.copy()] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]:
        res = minimize(
            obj,
            x0,
            jac=lambda x: 2 * Rm @ x,
            method="SLSQP",
            constraints=cons,
            bounds=[(0.0, 1.0)] * n,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res.success:
            x = np.clip(res.x, 0.0, None)
            x = x / x.sum()
            if np.all(improvements(p, R, x) >= -1e-7) and obj(x) < best_val - 1e-12:
                best, best_val = x, obj(x)
    return best, best_val, False


def run_portfolio(p: np.ndarray, R: np.ndarray) -> dict[str, dict[str, Any]]:
    """Run all five methods on a synthetic geometry."""
    lam_m1p = m1_plus(p, R, rho=0.5)
    lam_c1, t_c1 = c1_plus_plus(p, R, c=0.5, eps=1e-8)
    lam_m1pp, t_m1 = m1_plus_plus(p, R)
    lam_p2, t_p2, fired = p2_plus_plus(p, R)
    lam_p3, c_val, indiff = p3_plus_plus(p, R)
    methods = {
        "M1+": (lam_m1p, {"scalar_gain": float(p @ R @ (lam_m1p - p))}),
        "C1++": (lam_c1, {"t_star": t_c1}),
        "M1++": (lam_m1pp, {"t_star": t_m1}),
        "P2++": (lam_p2, {"t_star": t_p2, "surgeries_fired": fired}),
        "P3++": (lam_p3, {"conflict_energy": c_val, "indifferent_C_is_zero": indiff}),
    }
    out = {}
    for name, (lam, extra) in methods.items():
        delta = improvements(p, R, lam)
        out[name] = {
            "lambda": lam,
            "returns_p": bool(np.allclose(lam, p, atol=1e-6)),
            "vector_safe": bool(np.all(delta >= -1e-7)),
            "scalar_gain": float(p @ R @ (lam - p)),
            "min_improvement": float(delta.min()),
            **extra,
        }
    return out


def conflict_free_R(n: int = 5) -> np.ndarray:
    """PSD correlation-like R with all positive off-diagonals."""
    return np.full((n, n), 0.3) + np.eye(n) * 0.7


def negative_offdiag_collapsed_R(n: int = 5) -> np.ndarray:
    """PSD R with negative off-diagonals but collapsed floor."""
    return np.full((n, n), -0.1) + np.eye(n) * 1.1


def test_preference() -> np.ndarray:
    """Non-uniform preference used in synthetic tests."""
    return np.asarray([0.5, 0.125, 0.125, 0.125, 0.125], dtype=float)


def check_g1_conflict_free_floor() -> tuple[str, dict[str, Any]]:
    """G1: conflict-free R has collapsed floor; vector-safe methods return p."""
    R = conflict_free_R()
    p = test_preference()
    value, collapsed = floor_lp(R)
    if abs(value) > 1e-8 or not collapsed:
        raise AssertionError(f"Expected collapsed floor with LP value 0, got {value}.")
    portfolio = run_portfolio(p, R)
    for name in ("C1++", "M1++", "P2++", "P3++"):
        if not portfolio[name]["returns_p"]:
            raise AssertionError(f"{name} should return p in collapsed floor: {portfolio[name]}")
    if portfolio["M1+"]["returns_p"]:
        raise AssertionError("M1+ should move on the scalar objective in this test.")
    if portfolio["M1+"]["vector_safe"]:
        raise AssertionError("M1+ should not be vector-safe in this conflict-free test.")
    return "Conflict-free R collapses the floor; vector-safe methods return p; M1+ moves.", {
        "floor_lp_value": value,
        "m1plus": {
            "scalar_gain": portfolio["M1+"]["scalar_gain"],
            "min_improvement": portfolio["M1+"]["min_improvement"],
        },
    }


def check_g2_negative_offdiag_not_sufficient() -> tuple[str, dict[str, Any]]:
    """G2: negative off-diagonals do not automatically open the floor."""
    R = negative_offdiag_collapsed_R()
    eig = np.linalg.eigvalsh(R)
    offdiag = R[~np.eye(R.shape[0], dtype=bool)]
    value, collapsed = floor_lp(R)
    if not np.all(eig >= -1e-10):
        raise AssertionError(f"Synthetic R is not PSD: {eig}")
    if not np.any(offdiag < 0):
        raise AssertionError("Synthetic R has no negative off-diagonal.")
    if abs(value) > 1e-8 or not collapsed:
        raise AssertionError(f"R- != 0 should still collapse here; got LP value {value}.")
    return "PSD R has negative off-diagonals, but floor remains collapsed; LP is the criterion.", {
        "min_eigenvalue": float(eig.min()),
        "floor_lp_value": value,
    }


def check_g3_vector_safe_methods() -> tuple[str, dict[str, Any]]:
    """G3: vector-safe methods always produce R(lambda-p)>=0 in synthetic cases."""
    p = test_preference()
    cases = {"conflict_free": conflict_free_R(), "negative_collapsed": negative_offdiag_collapsed_R()}
    details: dict[str, Any] = {}
    for case, R in cases.items():
        portfolio = run_portfolio(p, R)
        details[case] = {}
        for name in ("C1++", "M1++", "P2++", "P3++"):
            min_imp = portfolio[name]["min_improvement"]
            details[case][name] = min_imp
            if min_imp < -1e-7:
                raise AssertionError(f"{case}/{name} is not vector-safe: {min_imp}")
    return "C1++/M1++/P2++/P3++ satisfy R(lambda-p)>=0 in synthetic cases.", details


def check_g4_vector_implies_scalar() -> tuple[str, dict[str, Any]]:
    """G4: vector-safe implies scalar-safe."""
    p = test_preference()
    details: dict[str, Any] = {}
    for case, R in {"conflict_free": conflict_free_R(), "negative_collapsed": negative_offdiag_collapsed_R()}.items():
        portfolio = run_portfolio(p, R)
        details[case] = {}
        for name, row in portfolio.items():
            if row["vector_safe"] and row["scalar_gain"] < -1e-8:
                raise AssertionError(f"{case}/{name}: vector-safe but scalar gain negative.")
            details[case][name] = {
                "vector_safe": row["vector_safe"],
                "scalar_gain": row["scalar_gain"],
            }
    return "Every vector-safe synthetic lambda is scalar-safe.", details


def paired_rank_delta(heads_p: np.ndarray, heads_l: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
    """Rank-normalized paired per-prompt delta."""
    from scipy.stats import rankdata

    n = heads_p.shape[0]
    deltas = np.zeros(n)
    for j in range(heads_p.shape[1]):
        pooled = rankdata(np.r_[heads_p[:, j], heads_l[:, j]]) / (2 * n)
        deltas += p_vec[j] * (pooled[n:] - pooled[:n])
    return deltas


def check_g5_paired_rank_shape() -> tuple[str, dict[str, Any]]:
    """G5: paired_rank_delta is per-prompt and therefore bootstrappable."""
    rng = np.random.default_rng(137)
    n_prompts = 13
    heads_p = rng.normal(size=(n_prompts, 5))
    heads_l = rng.normal(size=(n_prompts, 5))
    p = np.ones(5) / 5
    delta = paired_rank_delta(heads_p, heads_l, p)
    if delta.shape != (n_prompts,):
        raise AssertionError(f"Expected shape {(n_prompts,)}, got {delta.shape}")
    return "paired_rank_delta returns one value per prompt.", {"shape": list(delta.shape)}


# ---------------------------------------------------------------------------
# Stage 2 and 3: GPU checks
# ---------------------------------------------------------------------------


def require_cuda():
    """Import torch and require CUDA."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SkipCheck("PyTorch is unavailable; GPU checks cannot run.", error=repr(exc)) from exc
    if not torch.cuda.is_available():
        raise SkipCheck("CUDA is unavailable; GPU check skipped.")
    return torch


def ensure_theta_sft_snapshot(train, config: dict[str, Any]) -> Path:
    """Return the theta_SFT path, creating a base snapshot when configured."""
    torch = require_cuda()
    sft_path = PROJECT_ROOT / config["OUTPUT_DIR"] / "rs_runs" / "theta_sft" / "merged"
    if sft_path.exists():
        return sft_path
    if not config.get("USE_BASE_AS_THETA_SFT", False):
        raise SkipCheck("theta_SFT snapshot is missing and USE_BASE_AS_THETA_SFT is False.", path=str(sft_path))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft_path.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(config["BASE_MODEL"])
    model = AutoModelForCausalLM.from_pretrained(config["BASE_MODEL"], torch_dtype=torch.bfloat16)
    model.save_pretrained(str(sft_path), safe_serialization=True)
    tok.save_pretrained(str(sft_path))
    del model, tok
    torch.cuda.empty_cache()
    return sft_path


def load_sanity_rows(n: int = 200):
    """Load deterministic HelpSteer2 validation rows."""
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split="validation")
    rng = np.random.default_rng(137)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    return [ds[int(i)] for i in idx]


def check_a1_armorm_head_indices() -> tuple[str, dict[str, Any]]:
    """A1: ArmoRM id2label resolves all HelpSteer2 objectives by name."""
    require_cuda()
    train = import_train_module()
    scorer = train.ArmoRMHeadScorer(axis="helpfulness", model_id=train.ARMORM_MODEL)
    indices = scorer.helpsteer_indices()
    if len(indices) != 5 or len(set(indices)) != 5:
        raise AssertionError(f"HelpSteer2 head indices are not pairwise distinct: {indices}")
    details = {
        "indices": indices,
        "labels": {str(index): scorer.id2label[index] for index in indices},
    }
    del scorer
    return "ArmoRM exposes all five HelpSteer2 heads by name with distinct indices.", details


def check_a2_armorm_batching() -> tuple[str, dict[str, Any]]:
    """A2: batched scoring equals single-example scoring or falls back to batch_size=1."""
    require_cuda()
    train = import_train_module()
    rows = load_sanity_rows(8)
    pairs = [(str(row["prompt"]), "This is a short probe response.") for row in rows]
    scorer = train.ArmoRMHeadScorer(axis="helpfulness", model_id=train.ARMORM_MODEL)
    report = scorer.validate_batching(pairs, atol=1e-3)
    ok = (report.get("fallback_to_batch_size_1") is True) or (
        report.get("resolved_padding_side") in {"left", "right"}
        and min(
            report.get("max_abs_diff_left", math.inf),
            report.get("max_abs_diff_right", math.inf),
        )
        < 1e-3
    )
    if not ok:
        raise AssertionError(f"Batching validation failed: {report}")
    del scorer
    return "ArmoRM batched==single check passed or correctly fell back to batch_size=1.", report


def check_a3_armorm_degeneracy() -> tuple[str, dict[str, Any]]:
    """A3: ArmoRM heads are non-degenerate on theta_SFT generations."""
    torch = require_cuda()
    train = import_train_module()
    config = notebook_config()
    sft_path = ensure_theta_sft_snapshot(train, config)
    rows = load_sanity_rows(200)
    prompts = [str(row["prompt"]) for row in rows]
    scorer = train.ArmoRMHeadScorer(axis="helpfulness", model_id=train.ARMORM_MODEL)
    scorer.validate_batching([(prompt, "probe response") for prompt in prompts[:8]])
    responses = train.generate_responses(str(sft_path), prompts, seed=137)
    scores = scorer.score_all_heads(prompts, responses)
    stds = scores.std(axis=0)
    uniques = np.asarray([len(np.unique(scores[:, i])) for i in range(scores.shape[1])])
    failures = [
        f"{axis}: std={stds[i]:.3e}, unique={uniques[i]}"
        for i, axis in enumerate(train.ATTRIBUTES)
        if stds[i] <= 1e-6 or uniques[i] <= 10
    ]
    del scorer
    torch.cuda.empty_cache()
    if failures:
        raise AssertionError("Degenerate ArmoRM heads: " + "; ".join(failures))
    return "All ArmoRM heads vary on theta_SFT generations.", {
        "std": dict(zip(train.ATTRIBUTES, stds.tolist())),
        "unique": dict(zip(train.ATTRIBUTES, uniques.tolist())),
    }


def check_a4_armorm_axis_discriminance() -> tuple[str, dict[str, Any]]:
    """A4: report ArmoRM head-vs-label cross matrix."""
    require_cuda()
    from scipy.stats import pearsonr

    train = import_train_module()
    config = notebook_config()
    sft_path = ensure_theta_sft_snapshot(train, config)
    rows = load_sanity_rows(200)
    prompts = [str(row["prompt"]) for row in rows]
    labels = np.asarray([[float(row[axis]) for axis in train.ATTRIBUTES] for row in rows])
    scorer = train.ArmoRMHeadScorer(axis="helpfulness", model_id=train.ARMORM_MODEL)
    scorer.validate_batching([(prompt, "probe response") for prompt in prompts[:8]])
    responses = train.generate_responses(str(sft_path), prompts, seed=137)
    scores = scorer.score_all_heads(prompts, responses)
    cross = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            if scores[:, i].std() > 0 and labels[:, j].std() > 0:
                cross[i, j] = float(pearsonr(scores[:, i], labels[:, j]).statistic)
    own_argmax = [bool(np.nanargmax(cross[i]) == i) for i in range(5)]
    del scorer
    return "Axis discriminance matrix computed and reported.", {
        "attributes": list(train.ATTRIBUTES),
        "pearson": cross.tolist(),
        "diagonal": dict(zip(train.ATTRIBUTES, np.diag(cross).tolist())),
        "n_self_argmax": int(sum(own_argmax)),
        "self_argmax": dict(zip(train.ATTRIBUTES, own_argmax)),
    }


def merge_theta(
    lmbda: np.ndarray,
    adapter_paths: dict[str, Path],
    base_path: Path,
):
    """Merge weighted LoRA adapters into theta_SFT and return the dense model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.bfloat16, device_map="auto")
    names = list(adapter_paths.keys())
    peft_model = PeftModel.from_pretrained(model, str(adapter_paths[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        peft_model.load_adapter(str(adapter_paths[name]), adapter_name=name)
    peft_model.add_weighted_adapter(
        adapters=names,
        weights=[float(x) for x in lmbda],
        adapter_name="merged",
        combination_type="linear",
    )
    peft_model.set_adapter("merged")
    merged = peft_model.merge_and_unload()
    merged.eval()
    return merged


def stage3_paths() -> tuple[Path, dict[str, Path]]:
    """Resolve theta_SFT and PPO adapter paths."""
    config = notebook_config()
    sft_path = PROJECT_ROOT / config["OUTPUT_DIR"] / "rs_runs" / "theta_sft" / "merged"
    adapter_paths = {
        axis: PROJECT_ROOT / config["OUTPUT_DIR"] / "rs_runs" / f"ppo_{axis}" / "adapter"
        for axis in config["ATTRIBUTES"]
    }
    missing = [str(path) for path in [sft_path, *adapter_paths.values()] if not path.is_dir()]
    if missing:
        raise SkipCheck("Stage 3 requires existing theta_SFT and five PPO adapters.", missing=missing)
    return sft_path, adapter_paths


def state_fingerprint(model, max_tensors: int = 24) -> dict[str, float]:
    """Compute lightweight deterministic weight sums for target-module tensors."""
    import torch

    result: dict[str, float] = {}
    with torch.inference_mode():
        for name, tensor in model.state_dict().items():
            if not tensor.is_floating_point():
                continue
            if not any(part in name for part in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
                continue
            result[name] = float(tensor.detach().float().abs().sum().cpu())
            if len(result) >= max_tensors:
                break
    if not result:
        raise AssertionError("No target-module tensors found for fingerprinting.")
    return result


def max_fingerprint_diff(left: dict[str, float], right: dict[str, float]) -> float:
    """Maximum absolute difference over common fingerprint keys."""
    keys = sorted(set(left) & set(right))
    if not keys:
        raise AssertionError("No common fingerprint keys.")
    return max(abs(left[key] - right[key]) for key in keys)


def check_m1_vertex_merge_reproduces_adapter() -> tuple[str, dict[str, Any]]:
    """M1: merge_theta(e_k) matches the direct adapter-k merge path."""
    torch = require_cuda()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    config = notebook_config()
    sft_path, adapter_paths = stage3_paths()
    axis = config["ATTRIBUTES"][0]
    e0 = np.eye(5)[0]
    weighted = merge_theta(e0, adapter_paths, sft_path)
    base = AutoModelForCausalLM.from_pretrained(str(sft_path), torch_dtype=torch.bfloat16, device_map="auto")
    direct = PeftModel.from_pretrained(base, str(adapter_paths[axis])).merge_and_unload()
    diff = max_fingerprint_diff(state_fingerprint(weighted), state_fingerprint(direct))
    del weighted, direct, base
    torch.cuda.empty_cache()
    if diff > 1e-3:
        raise AssertionError(f"merge_theta(e_k) differs from direct adapter merge: {diff}")
    return "merge_theta(e_k) reproduces the direct adapter-k merge path.", {"axis": axis, "max_diff": diff}


def check_m2_zero_merge_is_base() -> tuple[str, dict[str, Any]]:
    """M2: merge_theta(lambda=0) equals theta_SFT."""
    torch = require_cuda()
    from transformers import AutoModelForCausalLM

    sft_path, adapter_paths = stage3_paths()
    zero = np.zeros(5)
    merged = merge_theta(zero, adapter_paths, sft_path)
    base = AutoModelForCausalLM.from_pretrained(str(sft_path), torch_dtype=torch.bfloat16, device_map="auto")
    diff = max_fingerprint_diff(state_fingerprint(merged), state_fingerprint(base))
    del merged, base
    torch.cuda.empty_cache()
    if diff > 1e-3:
        raise AssertionError(f"merge_theta(lambda=0) differs from theta_SFT: {diff}")
    return "merge_theta(lambda=0) matches theta_SFT weights.", {"max_diff": diff}


def check_m3_different_lambdas_change_weights() -> tuple[str, dict[str, Any]]:
    """M3: two different lambdas create different weights."""
    torch = require_cuda()
    sft_path, adapter_paths = stage3_paths()
    e0 = np.eye(5)[0]
    uniform = np.ones(5) / 5
    model_a = merge_theta(e0, adapter_paths, sft_path)
    model_b = merge_theta(uniform, adapter_paths, sft_path)
    diff = max_fingerprint_diff(state_fingerprint(model_a), state_fingerprint(model_b))
    del model_a, model_b
    torch.cuda.empty_cache()
    if diff <= 1e-6:
        raise AssertionError("Different lambda values produced indistinguishable weights.")
    return "Different lambda values produce different merged weights.", {"max_diff": diff}


STAGE_CHECKS: dict[str, list[tuple[str, Callable[[], tuple[str, dict[str, Any] | None]]]]] = {
    "0": [
        ("S1", check_s1_axis_order),
        ("S2", check_s2_no_policy_save),
        ("S3", check_s3_device_maps),
        ("S4", check_s4_apply_overrides),
        ("S5", check_s5_firewall),
        ("S6", check_s6_preregistration_consistency),
        ("S7", check_s7_equal_n),
    ],
    "1": [
        ("G1", check_g1_conflict_free_floor),
        ("G2", check_g2_negative_offdiag_not_sufficient),
        ("G3", check_g3_vector_safe_methods),
        ("G4", check_g4_vector_implies_scalar),
        ("G5", check_g5_paired_rank_shape),
    ],
    "2": [
        ("A1", check_a1_armorm_head_indices),
        ("A2", check_a2_armorm_batching),
        ("A3", check_a3_armorm_degeneracy),
        ("A4", check_a4_armorm_axis_discriminance),
    ],
    "3": [
        ("M1", check_m1_vertex_merge_reproduces_adapter),
        ("M2", check_m2_zero_merge_is_base),
        ("M3", check_m3_different_lambdas_change_weights),
    ],
}


def selected_stages(stage: str) -> list[str]:
    """Resolve stage argument to stages to run."""
    if stage == "all":
        return ["0", "1", "2", "3"]
    return [stage]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify RS-PPO setup before training.")
    parser.add_argument(
        "--stage",
        choices=["0", "1", "2", "3", "all"],
        default="0",
        help="Verification stage. Stage 0 and 1 are GPU-free.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected checks and exit nonzero unless every selected check passes."""
    args = parse_args()
    results: list[CheckResult] = []
    for stage in selected_stages(args.stage):
        for check_id, fn in STAGE_CHECKS[stage]:
            run_check(results, check_id, stage, fn)
    write_report(results)
    print_table(results)
    if any(result.status != "PASS" for result in results):
        print("\nDO NOT TRAIN -- fix the failing checks first.")
        raise SystemExit(1)
    print("\nAll selected checks passed.")


if __name__ == "__main__":
    main()
