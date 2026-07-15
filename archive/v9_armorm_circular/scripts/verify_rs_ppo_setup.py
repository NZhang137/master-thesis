"""Verify the RS-PPO/ArmoRM setup before expensive runs.

The script is intentionally conservative: it performs no training and exits
with code 0 only when every selected check passes. GPU/model stages are
explicitly marked SKIP when prerequisites are missing, which still blocks
training for the selected stage.

Negative sanity tests for maintainers:
- if `src.coefficient_portfolio.m1_plus` is temporarily changed to always return
  p, G1 must FAIL;
- if `src.merge.effective_deltas` is temporarily changed to use sqrt scaling,
  if B@A is transposed, or if `resolve_base_module` maps to a wrong module,
  G7 must FAIL.
These checks confirm the verifier is testing production implementations, not
local copies.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_portfolio import (  # noqa: E402 - project root is set above.
    floor_lp,
    paired_rank_delta,
    run_portfolio,
)
from src.merge import merge_theta  # noqa: E402 - project root is set above.
from src.preferences import PREFERENCES  # noqa: E402 - project root is set above.

REPORT_PATH = PROJECT_ROOT / "results" / "verify_rs_ppo_setup" / "report.json"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_rs_ppo_armorm_circular_colab.ipynb"
NOTEBOOK_09_PATH = PROJECT_ROOT / "notebooks" / "09_final_merge_test_colab.ipynb"
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


def notebook_cells(path: Path = NOTEBOOK_PATH) -> list[dict[str, Any]]:
    """Load notebook cells."""
    nb = json.loads(read_text(path))
    return list(nb.get("cells", []))


def notebook_source(path: Path = NOTEBOOK_PATH) -> str:
    """Return all notebook source text concatenated."""
    chunks: list[str] = []
    for cell in notebook_cells(path):
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def notebook_config(path: Path = NOTEBOOK_PATH, variable_name: str = "CONFIG") -> dict[str, Any]:
    """Extract the literal CONFIG dict from the notebook setup cell."""
    for cell in notebook_cells(path):
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
                and any(isinstance(t, ast.Name) and t.id == variable_name for t in node.targets)
            ):
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise TypeError(f"Notebook {variable_name} is not a dict.")
                return value
    raise RuntimeError(f"Could not find literal {variable_name} dict in notebook {path}.")


def notebook_code_cell_containing(*needles: str, path: Path = NOTEBOOK_PATH) -> str:
    """Return the first code cell containing all given strings."""
    for cell in notebook_cells(path):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if all(needle in text for needle in needles):
            return text
    raise RuntimeError(f"Could not find notebook code cell containing: {needles}")


def parse_notebook_code(text: str) -> ast.Module:
    """Parse a notebook code cell after removing Colab shell/magic lines."""
    filtered = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    return ast.parse(filtered)


class _DummyDateTime:
    """Minimal datetime stand-in for evaluating the preregistration dict."""

    @classmethod
    def now(cls, _tz: object) -> "_DummyDateTime":
        return cls()

    def isoformat(self) -> str:
        return "STATIC_VERIFIER_TIMESTAMP"


class _DummyTimezone:
    """Minimal timezone stand-in for evaluating the preregistration dict."""

    utc = object()


def extract_preregistration_dict() -> dict[str, Any]:
    """Evaluate only the notebook's pre_registration dict expression."""
    text = notebook_code_cell_containing("pre_registration =", "PREREGISTRATION_PATH")
    tree = parse_notebook_code(text)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pre_registration"
                    for target in node.targets)
        ):
            expr = ast.Expression(node.value)
            ast.fix_missing_locations(expr)
            safe_locals = {
                "CONFIG": notebook_config(),
                "datetime": _DummyDateTime,
                "timezone": _DummyTimezone,
            }
            value = eval(compile(expr, "<notebook-pre-registration>", "eval"), {"__builtins__": {}}, safe_locals)
            if not isinstance(value, dict):
                raise TypeError("pre_registration did not evaluate to a dict.")
            return value
    raise RuntimeError("Could not extract pre_registration dict from notebook.")


def collect_subscript_keys(node: ast.AST, variable_name: str) -> set[str]:
    """Collect string keys from expressions like variable_name['key']."""
    keys: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.value, ast.Name) or child.value.id != variable_name:
            continue
        slice_node = child.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            keys.add(slice_node.value)
    return keys


def raw_delta_check(path: Path) -> bool:
    """Return whether a notebook computes per_prompt as (heads_l - heads_p) @ p_vec."""
    text = notebook_code_cell_containing("per_prompt =", "delta_U_p_mean", path=path)
    tree = parse_notebook_code(text)
    raw_delta_seen = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "per_prompt" in target_names:
                value = node.value
                if (
                    isinstance(value, ast.BinOp)
                    and isinstance(value.op, ast.MatMult)
                    and isinstance(value.right, ast.Name)
                    and value.right.id == "p_vec"
                    and isinstance(value.left, ast.BinOp)
                    and isinstance(value.left.op, ast.Sub)
                    and isinstance(value.left.left, ast.Name)
                    and value.left.left.id == "heads_l"
                    and isinstance(value.left.right, ast.Name)
                    and value.left.right.id == "heads_p"
                ):
                    raw_delta_seen = True
    return raw_delta_seen


def nb09_metric_and_holm_check() -> dict[str, Any]:
    """Validate NB09 metric provenance and Holm-corrected verdict wiring."""
    prereg_text = notebook_code_cell_containing("PREREG_ADDENDUM_PATH", "PRIMARY_METRIC", path=NOTEBOOK_09_PATH)
    prereg_tree = parse_notebook_code(prereg_text)
    primary_from_nb08 = False
    addendum_uses_primary = False
    addendum_reports_unweighted_delta_m = False
    addendum_context_weighted_delta_m = False
    for node in ast.walk(prereg_tree):
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "PRIMARY_METRIC" in targets:
                source = ast.unparse(node.value)
                primary_from_nb08 = "nb08_prereg['primary']['metric']" in source
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "metric":
                    addendum_uses_primary = isinstance(value, ast.Name) and value.id == "PRIMARY_METRIC"
                if isinstance(key, ast.Constant) and key.value == "reported_secondary_metric":
                    addendum_reports_unweighted_delta_m = (
                        isinstance(value, ast.Constant)
                        and value.value == "delta_m_percent_unweighted_gain"
                    )
                if isinstance(key, ast.Constant) and key.value == "context_metric":
                    addendum_context_weighted_delta_m = (
                        isinstance(value, ast.Constant)
                        and value.value == "delta_m_percent_pref_weighted_gain"
                    )

    analysis_text = notebook_code_cell_containing(
        "n_quality_significant_after_holm",
        "upper_bound_verdict",
        path=NOTEBOOK_09_PATH,
    )
    analysis_tree = parse_notebook_code(analysis_text)
    verdict_uses_holm_count = False
    raw_quality_count_drives_verdict = False
    analysis_reports_unweighted_delta_m = False
    analysis_reports_weighted_delta_m = False
    holm_uses_testable_rows = False
    analysis_checks_scoring_status = False
    for node in ast.walk(analysis_tree):
        if isinstance(node, ast.If):
            cond = ast.unparse(node.test)
            assigns_verdict = any(
                isinstance(child, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "upper_bound_verdict"
                        for target in child.targets)
                for child in node.body + node.orelse
            )
            if assigns_verdict:
                if "n_quality_significant_after_holm" in cond:
                    verdict_uses_holm_count = True
                if "q_sig" in cond or "n_quality_significant" in cond and "after_holm" not in cond:
                    raw_quality_count_drives_verdict = True
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "reported_secondary_metric":
                    analysis_reports_unweighted_delta_m = (
                        isinstance(value, ast.Constant)
                        and value.value == "delta_m_percent_unweighted_gain"
                    )
                if isinstance(key, ast.Constant) and key.value == "context_secondary_metric":
                    analysis_reports_weighted_delta_m = (
                        isinstance(value, ast.Constant)
                        and value.value == "delta_m_percent_pref_weighted_gain"
                    )
        if isinstance(node, ast.ListComp):
            text = ast.unparse(node)
            if "row['testable']" in text and "row['preference'] in family" in text:
                holm_uses_testable_rows = True
    analysis_source = ast.unparse(analysis_tree)
    analysis_checks_scoring_status = all(
        needle in analysis_source
        for needle in ("SCORING_STATUS_PATH", "n_gen_per_prompt", "merge_dtype", "CONFIG_09['MERGE_DTYPE']")
    )

    raw_delta_seen = raw_delta_check(NOTEBOOK_09_PATH)
    if not primary_from_nb08:
        raise AssertionError("NB09 PRIMARY_METRIC is not sourced from NB08 preregistration primary.metric.")
    if not addendum_uses_primary:
        raise AssertionError("NB09 preregistration addendum does not write PRIMARY_METRIC as its metric.")
    if not raw_delta_seen:
        raise AssertionError("NB09 analysis does not compute per_prompt as (heads_l - heads_p) @ p_vec.")
    if not verdict_uses_holm_count or raw_quality_count_drives_verdict:
        raise AssertionError("NB09 upper_bound_verdict is not driven by the Holm-corrected quality count.")
    if not (addendum_reports_unweighted_delta_m and addendum_context_weighted_delta_m):
        raise AssertionError("NB09 addendum does not freeze the unweighted Delta m% secondary metric.")
    if not (analysis_reports_unweighted_delta_m and analysis_reports_weighted_delta_m):
        raise AssertionError("NB09 analysis does not report both Delta m% variants with distinct names.")
    if not holm_uses_testable_rows:
        raise AssertionError("NB09 Holm correction does not appear to exclude no-movement rows.")
    if not analysis_checks_scoring_status:
        raise AssertionError("NB09 analysis does not gate heads_by_lambda.npz on scoring status provenance.")
    return {
        "primary_metric_from_nb08": primary_from_nb08,
        "addendum_uses_primary_metric": addendum_uses_primary,
        "raw_delta_seen": raw_delta_seen,
        "verdict_uses_holm_count": verdict_uses_holm_count,
        "delta_m_secondary_metric": "delta_m_percent_unweighted_gain",
        "holm_moving_preferences_only": holm_uses_testable_rows,
        "heads_by_lambda_scoring_status_gate": analysis_checks_scoring_status,
    }


def write_report(results: list[CheckResult], requested_stages: list[str]) -> None:
    """Write report.json."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_stages = sorted({result.stage for result in results})
    all_stages = sorted(STAGE_CHECKS)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "requested": requested_stages,
            "run": run_stages,
            "not_run": [stage for stage in all_stages if stage not in run_stages],
            "gpu_free": ["0", "1"],
            "gpu_required": ["2", "3"],
        },
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
    """S6: preregistered metrics must match the NB09 decision code."""
    prereg = extract_preregistration_dict()
    expected_superseder = "preregistration_addendum_nb09.json (Holm-corrected, per-family)"
    if prereg.get("superseded_by") != expected_superseder:
        raise AssertionError(
            "NB08 preregistration must point to the Holm-corrected NB09 addendum; "
            f"got {prereg.get('superseded_by')!r}."
        )
    primary = prereg.get("primary", {})
    metric = str(primary.get("metric", ""))
    success = str(primary.get("success", ""))

    metric_upper = metric.upper()
    required_metric_terms = ["PAIRED", "PER-PROMPT", "NATIVE ARMORM REWARD SCALE"]
    missing_terms = [term for term in required_metric_terms if term not in metric_upper]
    if missing_terms:
        raise AssertionError(
            "Pre-registration primary.metric does not describe the actual raw paired scale; "
            f"missing terms: {missing_terms}. metric={metric!r}"
        )
    if "BOOTSTRAP 95% CI" not in success.upper():
        raise AssertionError(f"Pre-registration primary.success does not name bootstrap 95% CI: {success!r}")

    config_threshold = float(notebook_config()["WALL_A_R2_THRESHOLD"])
    prereg_threshold = float(prereg["secondary_non_circular"]["wall_A_R2_threshold"])
    if abs(config_threshold - prereg_threshold) > 1e-12:
        raise AssertionError(
            "Wall-A threshold mismatch: "
            f"CONFIG={config_threshold}, preregistration={prereg_threshold}"
        )
    expected_rule = "Wall A stands iff max R2 over the quality axes < threshold."
    if prereg["secondary_non_circular"].get("wall_A_rule") != expected_rule:
        raise AssertionError("Wall-A rule in preregistration does not match the frozen rule.")

    nb09 = nb09_metric_and_holm_check()
    return "NB08 preregistration, NB09 addendum, raw paired scale, Holm verdict, and Wall-A threshold match.", {
        "primary_metric": metric,
        "primary_success": success,
        "wall_A_R2_threshold": config_threshold,
        "nb09": nb09,
    }


def check_s8_notebook08_no_primary_endpoint() -> tuple[str, dict[str, Any]]:
    """S8: Notebook 08 must no longer contain the binding primary endpoint."""
    source = notebook_source(NOTEBOOK_PATH)
    forbidden = ["RUN_FINAL_MERGE", "MERGE_RESULTS_PATH"]
    found = [needle for needle in forbidden if needle in source]
    if found:
        raise AssertionError(f"Notebook 08 still contains primary-endpoint markers: {found}")
    return "Notebook 08 no longer contains RUN_FINAL_MERGE or MERGE_RESULTS_PATH.", {
        "forbidden": forbidden,
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


def portfolio_cfg() -> dict[str, float]:
    """Read portfolio hyperparameters from the notebook CONFIG."""
    config = notebook_config()
    return {
        "M1PLUS_RHO": float(config["M1PLUS_RHO"]),
        "C1PP_C": float(config["C1PP_C"]),
        "C1PP_EPS": float(config["C1PP_EPS"]),
    }


def conflict_free_R(n: int = 5) -> np.ndarray:
    """PSD correlation-like R with all positive off-diagonals."""
    return np.full((n, n), 0.3) + np.eye(n) * 0.7


def negative_offdiag_collapsed_R(n: int = 5) -> np.ndarray:
    """PSD R with negative off-diagonals but collapsed floor."""
    return np.full((n, n), -0.1) + np.eye(n) * 1.1


def test_preference() -> np.ndarray:
    """Non-uniform preference used in synthetic tests."""
    return np.asarray([0.5, 0.125, 0.125, 0.125, 0.125], dtype=float)


def portfolio_for(p: np.ndarray, R: np.ndarray) -> dict[str, dict[str, Any]]:
    """Run the production portfolio using notebook hyperparameters."""
    return run_portfolio(p, R, portfolio_cfg())


def scalar_gain(p: np.ndarray, R: np.ndarray, row: dict[str, Any]) -> float:
    """Compute p^T R(lambda-p) from a portfolio row."""
    lam = np.asarray(row["lam"], dtype=float)
    return float(np.asarray(p, dtype=float) @ np.asarray(R, dtype=float) @ (lam - np.asarray(p, dtype=float)))


def check_g1_conflict_free_floor() -> tuple[str, dict[str, Any]]:
    """G1: conflict-free R has collapsed floor; vector-safe methods return p."""
    R = conflict_free_R()
    p = test_preference()
    value, collapsed = floor_lp(R, tol=1e-9)
    if abs(value) > 1e-8 or not collapsed:
        raise AssertionError(f"Expected collapsed floor with LP value 0, got {value}.")
    portfolio = portfolio_for(p, R)
    for name in ("C1++", "M1++", "P2++", "P3++"):
        if not portfolio[name]["returns_p"]:
            raise AssertionError(f"{name} should return p in collapsed floor: {portfolio[name]}")
    if portfolio["M1+"]["returns_p"]:
        raise AssertionError("M1+ should move on the scalar objective in this test.")
    if portfolio["M1+"]["vector_safe"]:
        raise AssertionError("M1+ should not be vector-safe in this conflict-free test.")
    return "Production portfolio: conflict-free R collapses floor; vector-safe methods return p; M1+ moves.", {
        "floor_lp_value": value,
        "m1plus": {
            "scalar_gain_pT_R_dlam": portfolio["M1+"]["scalar_gain_pT_R_dlam"],
            "min_improvement": portfolio["M1+"]["min_improvement"],
        },
    }


def check_g2_negative_offdiag_not_sufficient() -> tuple[str, dict[str, Any]]:
    """G2: negative off-diagonals do not automatically open the floor."""
    R = negative_offdiag_collapsed_R()
    eig = np.linalg.eigvalsh(R)
    offdiag = R[~np.eye(R.shape[0], dtype=bool)]
    value, collapsed = floor_lp(R, tol=1e-9)
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
        portfolio = portfolio_for(p, R)
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
        portfolio = portfolio_for(p, R)
        details[case] = {}
        for name, row in portfolio.items():
            gain = scalar_gain(p, R, row)
            if row["vector_safe"] and gain < -1e-8:
                raise AssertionError(f"{case}/{name}: vector-safe but scalar gain negative.")
            details[case][name] = {
                "vector_safe": row["vector_safe"],
                "scalar_gain": gain,
            }
    return "Every vector-safe synthetic lambda is scalar-safe.", details


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


TOY_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def require_toy_merge_deps() -> dict[str, Any]:
    """Import CPU-only model/PEFT dependencies for the toy merge verifier."""
    cache_root = PROJECT_ROOT / "results" / "verify_rs_ppo_setup" / "hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from safetensors.torch import load_file
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:  # noqa: BLE001
        raise SkipCheck(
            "Toy merge checks require torch, transformers, peft, and safetensors.",
            error=repr(exc),
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "load_file": load_file,
        "LlamaConfig": LlamaConfig,
        "LlamaForCausalLM": LlamaForCausalLM,
    }


def independent_find_weight_module(model: Any, module_name: str) -> Any:
    """Resolve a module name without using src.merge."""
    modules = dict(model.named_modules())
    candidates = [module_name]
    if module_name.startswith("base_model.model."):
        candidates.append(module_name[len("base_model.model."):])
    if module_name.startswith("model."):
        candidates.append(module_name)
    for candidate in dict.fromkeys(candidates):
        if candidate in modules and hasattr(modules[candidate], "weight"):
            return modules[candidate]
    matches = [
        module for name, module in modules.items()
        if name.endswith(f".{module_name}") and hasattr(module, "weight")
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Could not independently resolve module {module_name!r}.")


def adapter_weight_path(adapter_path: Path) -> tuple[Path, str]:
    """Return adapter weight path and storage format."""
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        return safetensors_path, "safetensors"
    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        return bin_path, "bin"
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def split_lora_module_key(key: str, side: str) -> tuple[str, str] | None:
    """Return (base_module_name, matching_suffix) for one LoRA key."""
    suffixes = (f".lora_{side}.default.weight", f".lora_{side}.weight")
    for suffix in suffixes:
        if key.endswith(suffix):
            module_name = key[:-len(suffix)]
            if module_name.startswith("base_model.model."):
                module_name = module_name[len("base_model.model."):]
            return module_name, suffix
    return None


def independent_effective_deltas(adapter_path: Path) -> dict[str, Any]:
    """Read a PEFT adapter directly and compute (alpha/r) B@A without src.merge."""
    deps = require_toy_merge_deps()
    torch = deps["torch"]
    load_file = deps["load_file"]

    cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    rank = cfg["r"]
    alpha = cfg["lora_alpha"]
    if isinstance(rank, dict) or isinstance(alpha, dict):
        raise ValueError("The verifier expects scalar LoRA r and lora_alpha.")
    scaling = float(alpha) / float(rank)
    weight_path, fmt = adapter_weight_path(adapter_path)
    state = load_file(str(weight_path)) if fmt == "safetensors" else torch.load(weight_path, map_location="cpu")

    deltas: dict[str, Any] = {}
    for key, value in state.items():
        parsed = split_lora_module_key(key, "A")
        if parsed is None:
            continue
        module_name, suffix = parsed
        b_key = key[:-len(suffix)] + suffix.replace("lora_A", "lora_B")
        if b_key not in state:
            raise KeyError(f"Missing matching LoRA B tensor for {key}")
        lora_a = value.detach().cpu().to(dtype=torch.float64)
        lora_b = state[b_key].detach().cpu().to(dtype=torch.float64)
        deltas[module_name] = scaling * (lora_b @ lora_a)
    if not deltas:
        raise AssertionError(f"No LoRA A/B pairs found in {adapter_path}")
    return deltas


def independent_expected_weights(
    base_model: Any,
    adapter_paths: dict[str, Path],
    lmbda: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute expected target weights from raw adapter files, independent of src.merge."""
    names = list(adapter_paths)
    weights = np.asarray(lmbda, dtype=np.float64)
    raw_deltas = {name: independent_effective_deltas(adapter_paths[name]) for name in names}
    module_names = sorted(raw_deltas[names[0]])
    for name in names[1:]:
        if sorted(raw_deltas[name]) != module_names:
            raise AssertionError(f"Adapter {name!r} has a different LoRA target set.")

    base_weights: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for module_name in module_names:
        module = independent_find_weight_module(base_model, module_name)
        weight = module.weight.detach().cpu().to(dtype=raw_deltas[names[0]][module_name].dtype)
        base_weights[module_name] = weight.clone()
        total = weight.clone()
        for adapter_name, coeff in zip(names, weights):
            total = total + float(coeff) * raw_deltas[adapter_name][module_name]
        expected[module_name] = total
    return expected, base_weights


def max_abs_target_error(model: Any, expected: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Return max absolute error on expected target weights."""
    errors: dict[str, float] = {}
    for module_name, expected_weight in expected.items():
        actual = independent_find_weight_module(model, module_name).weight.detach().cpu().to(dtype=expected_weight.dtype)
        errors[module_name] = float((actual - expected_weight).abs().max().item())
    return max(errors.values()), errors


def relative_delta_error(model: Any, expected: dict[str, Any], base_weights: dict[str, Any]) -> float:
    """Relative error against the true target delta, not against full base weights."""
    numerator = 0.0
    denominator = 0.0
    for module_name, expected_weight in expected.items():
        actual = independent_find_weight_module(model, module_name).weight.detach().cpu().to(dtype=expected_weight.dtype)
        diff = actual - expected_weight
        true_delta = expected_weight - base_weights[module_name]
        numerator += float((diff * diff).sum().item())
        denominator += float((true_delta * true_delta).sum().item())
    return float(math.sqrt(numerator / denominator)) if denominator > 0 else float("inf")


def build_toy_lora_case(tmp_root: Path, n_adapters: int = 3) -> tuple[Path, dict[str, Path], Any]:
    """Build a tiny LlamaForCausalLM and real random PEFT LoRA adapters on CPU."""
    deps = require_toy_merge_deps()
    torch = deps["torch"]
    LlamaConfig = deps["LlamaConfig"]
    LlamaForCausalLM = deps["LlamaForCausalLM"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]

    torch.manual_seed(137)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    base_model = LlamaForCausalLM(config).to(dtype=torch.float32)
    base_path = tmp_root / "theta_0"
    base_model.save_pretrained(str(base_path), safe_serialization=True)

    adapter_paths: dict[str, Path] = {}
    for adapter_idx in range(n_adapters):
        model = LlamaForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
        peft_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=TOY_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, peft_cfg)
        generator = torch.Generator(device="cpu").manual_seed(137 + adapter_idx)
        with torch.no_grad():
            for module in peft_model.modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    for adapter_name in module.lora_A:
                        module.lora_A[adapter_name].weight.normal_(mean=0.0, std=0.05, generator=generator)
                        module.lora_B[adapter_name].weight.normal_(mean=0.0, std=0.05, generator=generator)
        adapter_dir = tmp_root / f"adapter_{adapter_idx}"
        peft_model.save_pretrained(str(adapter_dir), safe_serialization=True)
        adapter_paths[f"a{adapter_idx}"] = adapter_dir
    return base_path, adapter_paths, config


def prepare_toy_dir(name: str) -> Path:
    """Create a fresh short-path toy directory for model/adapters."""
    root = (Path(os.environ.get("TEMP", PROJECT_ROOT / "results")) / "rsppo_verify_toy").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{name}_{os.getpid()}_{uuid.uuid4().hex[:8]}").resolve()
    if root not in path.parents:
        raise RuntimeError(f"Refusing to clear toy path outside verifier directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_g6_no_duplicate_portfolio() -> tuple[str, dict[str, Any]]:
    """G6: portfolio and merge functions must only be defined in src modules."""
    names = (
        "m1_plus",
        "c1_plus_plus",
        "run_portfolio",
        "floor_lp",
        "merge_theta",
        "effective_deltas",
        "score_lambda",
        "lambda_key",
        "holm_adjust",
        "delta_m_percent",
        "score_one_lambda",
    )
    needles = tuple(f"def {name}(" for name in names)
    allowed_roots = {
        (PROJECT_ROOT / "src" / "coefficient_portfolio.py").resolve(),
        (PROJECT_ROOT / "src" / "merge.py").resolve(),
        (PROJECT_ROOT / "src" / "metrics.py").resolve(),
        (PROJECT_ROOT / "src" / "lambda_utils.py").resolve(),
    }
    hits: list[dict[str, Any]] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".ipynb"}:
            continue
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        text = read_text(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(needle in line for needle in needles):
                hits.append({"path": str(path.relative_to(PROJECT_ROOT)), "line": line_no, "text": line.strip()})
    duplicates = [hit for hit in hits if (PROJECT_ROOT / hit["path"]).resolve() not in allowed_roots]
    if duplicates:
        raise AssertionError(f"Duplicate portfolio/merge definitions found outside src/: {duplicates}")
    return "No duplicate portfolio or merge definitions outside the shared src modules.", {"hits": hits}


def check_g7_synthetic_lora_linearity() -> tuple[str, dict[str, Any]]:
    """G7: production merge_theta matches an independent toy-PEFT oracle."""
    deps = require_toy_merge_deps()
    torch = deps["torch"]
    LlamaForCausalLM = deps["LlamaForCausalLM"]
    PeftModel = deps["PeftModel"]

    tmp_root = prepare_toy_dir("g7_merge")
    base_path, adapter_paths, _config = build_toy_lora_case(tmp_root)
    weights = np.asarray([0.15, 0.35, 0.50], dtype=np.float64)

    base = LlamaForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
    expected, base_weights = independent_expected_weights(base, adapter_paths, weights)
    expected_count = len(TOY_TARGET_MODULES) * 2
    if len(expected) != expected_count:
        raise AssertionError(f"Expected {expected_count} touched modules, got {len(expected)}.")

    merged = merge_theta(weights, adapter_paths, base_path, dtype=torch.float32)
    max_abs, module_errors = max_abs_target_error(merged, expected)
    moved = {
        name: float((independent_find_weight_module(merged, name).weight.detach().cpu().double()
                     - base_weights[name]).norm().item())
        for name in expected
    }
    if max_abs >= 1e-5:
        raise AssertionError(f"merge_theta does not match independent oracle: max_abs={max_abs}")
    if any(value <= 0.0 for value in moved.values()):
        raise AssertionError("At least one target module was not touched by merge_theta.")

    peft_base = LlamaForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
    names = list(adapter_paths)
    peft_model = PeftModel.from_pretrained(peft_base, str(adapter_paths[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        peft_model.load_adapter(str(adapter_paths[name]), adapter_name=name)
    peft_model.add_weighted_adapter(
        adapters=names,
        weights=weights.tolist(),
        adapter_name="merged",
        combination_type="linear",
    )
    peft_model.set_adapter("merged")
    peft_linear = peft_model.merge_and_unload()
    peft_rel_error = relative_delta_error(peft_linear, expected, base_weights)
    if peft_rel_error <= 0.1:
        raise AssertionError(f"PEFT linear control was not wrong enough: {peft_rel_error}")

    vertex_errors: dict[str, float] = {}
    for idx, name in enumerate(names):
        vertex = np.zeros(len(names), dtype=np.float64)
        vertex[idx] = 1.0
        vertex_merged = merge_theta(vertex, adapter_paths, base_path, dtype=torch.float32)
        direct_base = LlamaForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
        direct = PeftModel.from_pretrained(direct_base, str(adapter_paths[name])).merge_and_unload()
        direct_expected = {
            module_name: independent_find_weight_module(direct, module_name).weight.detach().cpu().double()
            for module_name in expected
        }
        vertex_max_abs, _ = max_abs_target_error(vertex_merged, direct_expected)
        vertex_errors[name] = vertex_max_abs
        if vertex_max_abs >= 1e-5:
            raise AssertionError(f"Vertex merge for {name} differs from PEFT direct merge: {vertex_max_abs}")

    return "Toy PEFT end-to-end merge matches independent oracle; PEFT-linear control fails.", {
        "max_abs_oracle_error": max_abs,
        "peft_linear_relative_error": peft_rel_error,
        "n_touched_modules": len(module_errors),
        "vertex_max_abs": vertex_errors,
    }


def check_g8_toy_bf16_precision_diagnostic() -> tuple[str, dict[str, Any]]:
    """G8: report bf16 storage noise on a toy lambda* vs p displacement."""
    deps = require_toy_merge_deps()
    torch = deps["torch"]
    LlamaForCausalLM = deps["LlamaForCausalLM"]

    tmp_root = prepare_toy_dir("g8_precision")
    base_path, adapter_paths, _config = build_toy_lora_case(tmp_root)
    base = LlamaForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
    p = np.asarray([0.20, 0.30, 0.50], dtype=np.float64)
    lam = np.asarray([0.25, 0.35, 0.40], dtype=np.float64)
    expected_p, _base_weights = independent_expected_weights(base, adapter_paths, p)
    expected_lam, _ = independent_expected_weights(base, adapter_paths, lam)
    true_parts = []
    bf16_parts = []
    for module_name in sorted(expected_p):
        w_p = expected_p[module_name]
        w_l = expected_lam[module_name]
        true_parts.append((w_l - w_p).reshape(-1))
        bf16_p = w_p.to(dtype=torch.bfloat16).to(dtype=torch.float64)
        bf16_l = w_l.to(dtype=torch.bfloat16).to(dtype=torch.float64)
        bf16_parts.append((bf16_l - bf16_p).reshape(-1))
    d_true = torch.cat(true_parts)
    d_bf16 = torch.cat(bf16_parts)
    denom = float(torch.linalg.norm(d_true).item())
    noise_ratio = float(torch.linalg.norm(d_bf16 - d_true).item() / denom)
    cosine = float(torch.dot(d_true, d_bf16).item() / (denom * float(torch.linalg.norm(d_bf16).item())))
    return "Toy bf16 storage diagnostic recorded; no pass/fail threshold on toy scales.", {
        "lambda_l2": float(np.linalg.norm(lam - p)),
        "bf16_noise_over_true_displacement": noise_ratio,
        "bf16_cosine_with_true_displacement": cosine,
    }


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


def check_a1_head_indices_by_name() -> tuple[str, dict[str, Any]]:
    """A1: head order is pinned by the AUTHORS' golden sample, not by our own code.

    ArmoRM ships NO id2label -- its config.json contains {"0": "LABEL_0"} and
    "num_objectives": 19. The names live only in the model card. So the index cannot be
    "read from the model"; it is resolved by name against src/armorm_objectives.py and
    then VERIFIED against the five HelpSteer-scale numbers the authors publish for a
    fixed prompt/response. That expected value does not come from our code, which is
    exactly what makes this check non-self-referential.
    """
    require_cuda()
    from src.armorm_objectives import ARMORM_HELPSTEER_OBJECTIVE_NAMES, N_ARMORM_OBJECTIVES

    scorer = rs_ppo.ArmoRMHeadScorer(axis="helpfulness", model_id=notebook_config()["ARMORM_MODEL"])
    indices = scorer.helpsteer_indices()
    if len(set(indices)) != len(indices):
        raise AssertionError(f"HelpSteer2 head indices are not distinct: {indices}")
    if int(scorer.model.config.num_objectives) != N_ARMORM_OBJECTIVES:
        raise AssertionError(
            f"num_objectives={scorer.model.config.num_objectives}, expected {N_ARMORM_OBJECTIVES}")

    golden = scorer.verify_against_model_card()   # raises on mismatch

    return (
        f"Head order pinned against the published golden sample "
        f"(max_abs={golden['max_abs_error']:.4f} <= {golden['atol']}).",
        {
            "indices": {a: int(i) for a, i in zip(HELPSTEER2_ATTRIBUTES, indices)},
            "objective_names": {a: ARMORM_HELPSTEER_OBJECTIVE_NAMES[a] for a in HELPSTEER2_ATTRIBUTES},
            "num_objectives": int(scorer.model.config.num_objectives),
            "golden_sample": golden,
            "note": "ArmoRM has no id2label; names come from the model card, the mapping "
                    "is verified against the authors' published numbers.",
        },
    )

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
    weighted = merge_theta(e0, adapter_paths, sft_path, dtype=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(str(sft_path), torch_dtype=torch.float32, device_map="auto")
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
    merged = merge_theta(zero, adapter_paths, sft_path, dtype=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(str(sft_path), torch_dtype=torch.float32, device_map="auto")
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
    model_a = merge_theta(e0, adapter_paths, sft_path, dtype=torch.float32)
    model_b = merge_theta(uniform, adapter_paths, sft_path, dtype=torch.float32)
    diff = max_fingerprint_diff(state_fingerprint(model_a), state_fingerprint(model_b))
    del model_a, model_b
    torch.cuda.empty_cache()
    if diff <= 1e-6:
        raise AssertionError("Different lambda values produced indistinguishable weights.")
    return "Different lambda values produce different merged weights.", {"max_diff": diff}


def check_m4_real_adapter_effective_merge() -> tuple[str, dict[str, Any]]:
    """M4: real interior merge equals sum_i lambda_i delta_i on weights."""
    torch = require_cuda()
    from transformers import AutoModelForCausalLM

    sft_path, adapter_paths = stage3_paths()
    lmbda = np.asarray(PREFERENCES["dominant_complexity"], dtype=np.float64)
    merged = merge_theta(lmbda, adapter_paths, sft_path, dtype=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(str(sft_path), torch_dtype=torch.float32, device_map="auto")
    expected, _base_weights = independent_expected_weights(base, adapter_paths, lmbda)
    max_abs = 0.0
    with torch.inference_mode():
        for module_name, expected_delta in expected.items():
            actual = independent_find_weight_module(merged, module_name).weight.detach().cpu().double()
            diff = actual - expected_delta
            max_abs = max(max_abs, float(diff.abs().max().detach().cpu()))
    del merged, base
    torch.cuda.empty_cache()
    if max_abs >= 1e-3:
        raise AssertionError(f"Merged model weights do not match effective delta sum: max_abs={max_abs}")
    return "Real dominant_complexity merge matches sum_i lambda_i delta_i.", {
        "max_abs_weight_error": max_abs,
    }


def check_m5_real_bf16_precision() -> tuple[str, dict[str, Any]]:
    """M5: real bf16 storage must preserve lambda* vs p displacement direction."""
    torch = require_cuda()
    from transformers import AutoModelForCausalLM

    config = notebook_config()
    r_path = PROJECT_ROOT / config["OUTPUT_DIR"] / "R_cos.npy"
    if not r_path.exists():
        raise SkipCheck("M5 requires R_cos.npy from Notebook 08.", missing=str(r_path))
    sft_path, adapter_paths = stage3_paths()
    R = np.load(r_path)
    p = np.asarray(PREFERENCES["dominant_complexity"], dtype=np.float64)
    lam = np.asarray(run_portfolio(p, R, portfolio_cfg())["M1+"]["lam"], dtype=np.float64)
    base = AutoModelForCausalLM.from_pretrained(str(sft_path), torch_dtype=torch.float32, device_map="auto")
    expected_p, _base_weights = independent_expected_weights(base, adapter_paths, p)
    expected_lam, _ = independent_expected_weights(base, adapter_paths, lam)
    true_parts = [(expected_lam[name] - expected_p[name]).reshape(-1) for name in sorted(expected_p)]
    d_true = torch.cat(true_parts)

    merged_p = merge_theta(p, adapter_paths, sft_path, dtype=torch.bfloat16)
    merged_lam = merge_theta(lam, adapter_paths, sft_path, dtype=torch.bfloat16)
    bf16_parts = []
    for module_name in sorted(expected_p):
        w_p = independent_find_weight_module(merged_p, module_name).weight.detach().cpu().double()
        w_lam = independent_find_weight_module(merged_lam, module_name).weight.detach().cpu().double()
        bf16_parts.append((w_lam - w_p).reshape(-1))
    d_bf16 = torch.cat(bf16_parts)
    true_norm = float(torch.linalg.norm(d_true).item())
    bf16_norm = float(torch.linalg.norm(d_bf16).item())
    if true_norm <= 0 or bf16_norm <= 0:
        raise AssertionError("M5 displacement norm is zero.")
    cosine = float(torch.dot(d_true, d_bf16).item() / (true_norm * bf16_norm))
    noise_ratio = float(torch.linalg.norm(d_bf16 - d_true).item() / true_norm)
    del base, merged_p, merged_lam
    torch.cuda.empty_cache()
    if cosine < 0.999:
        raise AssertionError(f"bf16 storage distorts real lambda displacement: cosine={cosine}")
    return "Real bf16 storage preserves dominant_complexity lambda displacement direction.", {
        "preference": "dominant_complexity",
        "lambda_l2": float(np.linalg.norm(lam - p)),
        "bf16_cosine_with_true_displacement": cosine,
        "bf16_noise_over_true_displacement": noise_ratio,
    }


STAGE_CHECKS: dict[str, list[tuple[str, Callable[[], tuple[str, dict[str, Any] | None]]]]] = {
    "0": [
        ("S1", check_s1_axis_order),
        ("S2", check_s2_no_policy_save),
        ("S3", check_s3_device_maps),
        ("S4", check_s4_apply_overrides),
        ("S5", check_s5_firewall),
        ("S6", check_s6_preregistration_consistency),
        ("S7", check_s7_equal_n),
        ("S8", check_s8_notebook08_no_primary_endpoint),
    ],
    "1": [
        ("G1", check_g1_conflict_free_floor),
        ("G2", check_g2_negative_offdiag_not_sufficient),
        ("G3", check_g3_vector_safe_methods),
        ("G4", check_g4_vector_implies_scalar),
        ("G5", check_g5_paired_rank_shape),
        ("G6", check_g6_no_duplicate_portfolio),
        ("G7", check_g7_synthetic_lora_linearity),
        ("G8", check_g8_toy_bf16_precision_diagnostic),
    ],
    "2": [
        ("A1", check_a1_head_indices_by_name),
        ("A2", check_a2_armorm_batching),
        ("A3", check_a3_armorm_degeneracy),
        ("A4", check_a4_armorm_axis_discriminance),
    ],
    "3": [
        ("M1", check_m1_vertex_merge_reproduces_adapter),
        ("M2", check_m2_zero_merge_is_base),
        ("M3", check_m3_different_lambdas_change_weights),
        ("M4", check_m4_real_adapter_effective_merge),
        ("M5", check_m5_real_bf16_precision),
    ],
}


def selected_stages(stage: str | list[str] | None) -> list[str]:
    """Resolve stage argument to stages to run."""
    if stage is None:
        return ["0", "1"]
    raw = stage if isinstance(stage, list) else [stage]
    if any(item == "all" for item in raw):
        return ["0", "1", "2", "3"]
    stages: list[str] = []
    for item in raw:
        stages.extend(part.strip() for part in item.split(",") if part.strip())
    valid = set(STAGE_CHECKS)
    invalid = [part for part in stages if part not in valid]
    if invalid or not stages:
        raise ValueError(f"Invalid --stage value {stage!r}; use comma-separated stages from {sorted(valid)} or 'all'.")
    return stages


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify RS-PPO setup before training.")
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Verification stage(s), e.g. 0, 1, 0,1, 2, 3, or all. Stage 0 and 1 are GPU-free.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected checks and exit nonzero unless every selected check passes."""
    args = parse_args()
    results: list[CheckResult] = []
    stages = selected_stages(args.stage)
    if stages == ["0"]:
        print("WARNING: Stage 1 (GPU-free) was not run -- the geometry/portfolio checks are the ones "
              "that catch silent math errors. Run with --stage 1 or use the default --stage 0,1.")
    for stage in stages:
        for check_id, fn in STAGE_CHECKS[stage]:
            run_check(results, check_id, stage, fn)
    write_report(results, stages)
    print_table(results)
    if any(result.status != "PASS" for result in results):
        print("\nDO NOT TRAIN -- fix the failing checks first.")
        raise SystemExit(1)
    print("\nAll selected checks passed.")


if __name__ == "__main__":
    main()
