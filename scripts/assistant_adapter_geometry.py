"""Geometry and floor certificate of the supervisor's assistant-task PPO experts (NB15).

The three experts (helpful, harmless, humor) are LoRA adapters on
``meta-llama/Llama-2-7b-hf``. This script

1. finds each expert directory by its folder name ``ppo_assistant_<goal>_s<seed>`` or
   ``<goal>_model`` (the name used in the shared Drive folder)
   (the individually downloaded files lost their names, the folder structure keeps them),
2. audits every adapter: configuration, hash, LoRA factor count, and any saved tensor that
   is **not** a LoRA factor (for example embeddings saved after a vocabulary resize, which
   the effective-update geometry would otherwise ignore silently),
3. computes ``R_gram``, ``R_cos`` and the update norms from the low-rank factors with the
   same code path as NB13/NB14 (``src/effective_lora_geometry.py``),
4. runs the floor certificate (``R^{-1} 1`` and the p-aware LP) for a fixed m = 3
   preference set, on both ``R_cos`` and ``R_gram``, and
5. reports the m = 3 redundancy diagnostic: the floor can open only if some entry of
   ``R^{-1} 1`` is non-positive, i.e. one expert is closer to both others than they are
   to each other.

No base model, GPU, reward model or Hugging Face token is needed.

Usage::

    python scripts/assistant_adapter_geometry.py --adapter_root <dir> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_portfolio import floor_lp, floor_lp_at_p  # noqa: E402
from src.effective_lora_geometry import (  # noqa: E402
    _parse_lora_factor_key,
    effective_lora_inner_product,
    effective_lora_update_norm,
    effective_lora_update_numel,
    load_adapter_config,
    load_effective_lora_geometry,
    validate_compatible_geometries,
)
from src.proxy_validation import build_search_set  # noqa: E402

GOALS = ("helpful", "harmless", "humor")
DEFAULT_SEED = 8888
EXPECTED_BASE = "meta-llama/Llama-2-7b-hf"
CONFIG_FIELDS = (
    "base_model_name_or_path", "revision", "peft_type", "task_type", "r", "lora_alpha",
    "lora_dropout", "target_modules", "bias", "use_rslora", "use_dora", "modules_to_save",
    "rank_pattern", "alpha_pattern", "fan_in_fan_out", "init_lora_weights",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_adapter_dirs(root: str | Path, goals: Sequence[str] = GOALS,
                      seed: int = DEFAULT_SEED) -> dict[str, Path]:
    """Map each goal to the unique adapter directory of its expert folder.

    Expert folders are named ``ppo_assistant_<goal>_s<seed>`` (training layout) or
    ``<goal>_model`` (shared Drive layout); the first name that matches is used. The adapter directory is the folder that contains ``adapter_config.json``. A folder
    named ``best_model`` is preferred when the expert folder holds several checkpoints.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Adapter root not found: {root}")
    found: dict[str, Path] = {}
    for goal in goals:
        names = (f"ppo_assistant_{goal}_s{seed}", f"{goal}_model")
        experts: list[Path] = []
        for expert_name in names:
            experts = [p for p in root.rglob(expert_name) if p.is_dir()]
            if experts:
                break
        if len(experts) != 1:
            raise FileNotFoundError(
                f"Expected exactly one folder named one of {names} below {root}, found {len(experts)}: "
                f"{[str(p) for p in experts]}"
            )
        configs = sorted(experts[0].rglob("adapter_config.json"))
        if not configs:
            raise FileNotFoundError(f"No adapter_config.json below {experts[0]}")
        preferred = [c for c in configs if c.parent.name == "best_model"]
        candidates = preferred or configs
        if len(candidates) != 1:
            raise RuntimeError(
                f"Ambiguous adapter for {goal}: {[str(c.parent) for c in candidates]}"
            )
        found[goal] = candidates[0].parent
    return found


def audit_adapter(adapter_dir: str | Path) -> dict[str, Any]:
    """Configuration, hashes and a list of saved tensors that are not LoRA factors."""
    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    config = load_adapter_config(adapter_dir)
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {adapter_dir}")
    lora_keys, other = 0, {}
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if _parse_lora_factor_key(key) is not None:
                lora_keys += 1
            else:
                other[key] = list(handle.get_slice(key).get_shape())
    tokenizer_files = sorted(p.name for p in adapter_dir.iterdir()
                             if p.name.startswith(("tokenizer", "special_tokens", "added_tokens")))
    added_tokens = {}
    added_path = adapter_dir / "added_tokens.json"
    if added_path.is_file():
        added_tokens = json.loads(added_path.read_text(encoding="utf-8"))
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_model_sha256": sha256_file(weights),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
        "config": {field: config.get(field) for field in CONFIG_FIELDS},
        "n_lora_factor_tensors": lora_keys,
        "non_lora_tensors": other,
        "tokenizer_files": tokenizer_files,
        "added_tokens": added_tokens,
    }


def check_audits(audits: dict[str, dict[str, Any]]) -> list[str]:
    """Return human-readable warnings; raise on problems that invalidate the geometry."""
    warnings: list[str] = []
    configs = {goal: audit["config"] for goal, audit in audits.items()}
    reference_goal = next(iter(configs))
    for goal, config in configs.items():
        if config.get("use_dora"):
            raise ValueError(f"{goal}: DoRA adapters need a different effective-update formula.")
        if config.get("modules_to_save"):
            raise ValueError(f"{goal}: modules_to_save={config['modules_to_save']} is not a pure LoRA update.")
        if config.get("base_model_name_or_path") != EXPECTED_BASE:
            warnings.append(f"{goal}: base_model_name_or_path is {config.get('base_model_name_or_path')!r}, "
                            f"expected {EXPECTED_BASE!r}. Ask the supervisor which base the adapter belongs to.")
        for field in ("r", "lora_alpha", "target_modules", "use_rslora", "base_model_name_or_path"):
            if config.get(field) != configs[reference_goal].get(field):
                warnings.append(f"{goal}: {field}={config.get(field)!r} differs from "
                                f"{reference_goal}={configs[reference_goal].get(field)!r}")
    for goal, audit in audits.items():
        if audit["non_lora_tensors"]:
            raise ValueError(
                f"{goal}: the adapter also stores non-LoRA tensors {audit['non_lora_tensors']}. "
                "They change theta_i but are not part of the LoRA geometry; do not report R "
                "without accounting for them."
            )
        if audit["added_tokens"]:
            warnings.append(f"{goal}: tokenizer adds tokens {audit['added_tokens']}; "
                            "the base embedding is resized before attaching the adapter.")
    hashes = [audit["adapter_model_sha256"] for audit in audits.values()]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Two goals point to byte-identical adapters; the folder assignment is wrong.")
    return warnings


def relationship_matrices(adapter_dirs: dict[str, Path]) -> dict[str, Any]:
    goals = list(adapter_dirs)
    geometries = {goal: load_effective_lora_geometry(adapter_dirs[goal]) for goal in goals}
    validate_compatible_geometries([geometries[g] for g in goals], goals)
    m = len(goals)
    gram = np.empty((m, m))
    for i, left in enumerate(goals):
        for j, right in enumerate(goals[: i + 1]):
            gram[i, j] = gram[j, i] = effective_lora_inner_product(geometries[left], geometries[right])
    norms = np.array([effective_lora_update_norm(geometries[g]) for g in goals])
    if not np.allclose(norms, np.sqrt(np.diag(gram)), rtol=1e-10):
        raise RuntimeError("Norms disagree with the Gram diagonal.")
    cos = gram / np.outer(norms, norms)
    cos = 0.5 * (cos + cos.T)
    first = geometries[goals[0]]
    return {
        "goals": goals, "R_gram": gram, "R_cos": cos, "norms": norms,
        "effective_update_numel": int(effective_lora_update_numel(first)),
        "n_modules": len(first),
        "module_suffixes": sorted({name.rsplit(".", 1)[-1] for name in first}),
    }


def preference_set(goals: Sequence[str], n_dirichlet: int = 64, seed: int = 137) -> list[tuple[str, np.ndarray]]:
    """Named m = 3 preferences plus Dirichlet draws, deduplicated."""
    m = len(goals)
    named: list[tuple[str, np.ndarray]] = [("balanced", np.full(m, 1.0 / m))]
    for k, goal in enumerate(goals):
        named.append((f"only_{goal}", np.eye(m)[k]))
    for k, goal in enumerate(goals):
        p = np.full(m, 0.4 / (m - 1))
        p[k] = 0.6
        named.append((f"dominant_{goal}", p))
    for i in range(m):
        for j in range(i + 1, m):
            p = np.zeros(m)
            p[i] = p[j] = 0.5
            named.append((f"pair_{goals[i]}_{goals[j]}", p))
    seen = {tuple(np.round(p, 9)) for _, p in named}
    prefs = list(named)
    draws = build_search_set(m, n_dirichlet=n_dirichlet, dirichlet_alpha=1.0,
                             preferences=[p for _, p in named], seed=seed)
    for index, vector in enumerate(draws):
        key = tuple(np.round(vector, 9))
        if key not in seen:
            seen.add(key)
            prefs.append((f"dirichlet_{index:02d}", np.asarray(vector, float)))
    for name, p in prefs:
        assert abs(p.sum() - 1) < 1e-9 and (p >= -1e-12).all(), name
    return prefs


def floor_report(R: np.ndarray, prefs: Sequence[tuple[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for name, p in prefs:
        t, collapsed = floor_lp_at_p(R, p)
        rows.append({"p_name": name, **{f"p_{k}": float(v) for k, v in enumerate(p)},
                     "t_lp": float(t), "collapsed": bool(collapsed)})
    return pd.DataFrame(rows)


def matrix_summary(R: np.ndarray, goals: Sequence[str]) -> dict[str, Any]:
    m = len(goals)
    eig = np.linalg.eigvalsh(R)
    off = R[np.triu_indices(m, 1)]
    inv_ones = np.linalg.solve(R, np.ones(m))
    global_t, global_collapsed = floor_lp(R)
    pairs = {f"{goals[i]}~{goals[j]}": float(R[i, j]) for i in range(m) for j in range(i + 1, m)}
    redundant = [goals[k] for k in range(m) if inv_ones[k] <= 1e-12]
    middle = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        if m == 3 and min(R[k, others[0]], R[k, others[1]]) > R[others[0], others[1]]:
            middle.append(goals[k])
    return {
        "eigenvalues": eig.tolist(),
        "condition_number": float(eig.max() / eig.min()) if eig.min() > 0 else float("inf"),
        "near_singular": bool(eig.min() <= 1e-8 * eig.max()),
        "dominant_eigenvalue_share": float(eig.max() / eig.sum()),
        "offdiag_min": float(off.min()), "offdiag_max": float(off.max()), "offdiag_mean": float(off.mean()),
        "n_negative_offdiag": int((off < 0).sum()),
        "pairs": pairs,
        "inverse_R_ones": inv_ones.tolist(),
        "inverse_R_ones_strictly_positive": bool(np.all(inv_ones > 1e-12)),
        "global_floor_lp_value": float(global_t), "global_floor_collapsed": bool(global_collapsed),
        "redundant_axes_inverse_R_ones_nonpositive": redundant,
        "middle_axes_closer_to_both_others": middle,
    }


def run(adapter_root: str | Path, out_dir: str | Path, goals: Sequence[str] = GOALS,
        seed: int = DEFAULT_SEED, n_dirichlet: int = 64, pref_seed: int = 137) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dirs = find_adapter_dirs(adapter_root, goals, seed)
    audits = {goal: audit_adapter(path) for goal, path in adapter_dirs.items()}
    warnings = check_audits(audits)
    geometry = relationship_matrices(adapter_dirs)
    prefs = preference_set(goals, n_dirichlet, pref_seed)

    report: dict[str, Any] = {
        "schema_version": 1,
        "task": "assistant (HH-RLHF), supervisor PPO experts",
        "goals": list(goals),
        "expert_seed": seed,
        "adapter_dirs": {g: str(p) for g, p in adapter_dirs.items()},
        "audits": audits,
        "warnings": warnings,
        "effective_update_numel": geometry["effective_update_numel"],
        "n_adapted_modules": geometry["n_modules"],
        "adapted_module_types": geometry["module_suffixes"],
        "update_norms": dict(zip(goals, geometry["norms"].tolist())),
        "n_preferences": len(prefs),
        "preference_seed": pref_seed,
        "primary_matrix": "R_cos",
    }
    for label in ("R_cos", "R_gram"):
        R = geometry[label]
        pd.DataFrame(R, index=goals, columns=goals).to_csv(out_dir / f"{label}_assistant.csv")
        floor = floor_report(R, prefs)
        floor.to_csv(out_dir / f"floor_lp_{label}.csv", index=False)
        report[label] = {
            **matrix_summary(R, goals),
            "floor_lp": {"n_preferences": len(floor), "n_collapsed": int(floor["collapsed"].sum()),
                         "open_preferences": floor.loc[~floor["collapsed"], "p_name"].tolist()},
        }
    pd.DataFrame({"goal": goals, "effective_update_norm": geometry["norms"]}).to_csv(
        out_dir / "update_norms_assistant.csv", index=False)
    pd.DataFrame([{"goal": g, **{k: v for k, v in a.items() if k not in ("config", "non_lora_tensors", "added_tokens", "tokenizer_files")},
                   **{f"config_{k}": json.dumps(v) for k, v in a["config"].items()}} for g, a in audits.items()]
                 ).to_csv(out_dir / "adapter_inventory_assistant.csv", index=False)
    (out_dir / "assistant_geometry_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                                            encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_dirichlet", type=int, default=64)
    args = parser.parse_args()
    report = run(args.adapter_root, args.out_dir, seed=args.seed, n_dirichlet=args.n_dirichlet)
    for label in ("R_cos", "R_gram"):
        print(label, json.dumps({k: report[label][k] for k in
                                 ("pairs", "inverse_R_ones", "floor_lp")}, indent=2))
    for warning in report["warnings"]:
        print("[warning]", warning)


if __name__ == "__main__":
    main()
