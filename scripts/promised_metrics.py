"""Compute the metrics that main.tex promises but no run has reported yet (NB17).

Review v3, section E: the thesis text defines and promises several quantities that NB10,
NB13 and NB13.1 never computed. Everything needed is already in the stored reward caches
(one line per evaluated merge point with its coefficient vector and the per-prompt scores
of every head), so this script needs no GPU, no model and no reward-model call.

Computed per run and per reward model:

E1  per-axis linearity: R2_vert (vertex-anchored affine fit, evaluated on the non-vertex
    points) and R2_OLS (best affine fit over all evaluated points), Def. linearity-indicator;
E2  Delta m% against the specialists (vertices), Def. delta-m, uniform and p-weighted;
E3  win rate against lambda = p per method configuration, and Delta U_p / SE;
E4  normalized weighted Tchebycheff shortfall T_p^norm (Sec. tchebycheff), method vs. baseline;
E5  selection-regret ratio reg_p(lambda*) / reg_p^random (Def. regret);
E6  H-floor check on the vertex matrix M_ij = r_i(theta_j) (Rem. h-limits, Prop.
    h-floor-collapse), global and p-aware, with the R-floor for comparison; plus the empirical
    counterpart: does any evaluated merge point beat lambda = p on every head at once?

Uncertainty: paired bootstrap over prompts (the same prompt draw is applied to every merge
point), percentile intervals. Important scope note for the thesis: the evaluated set B here
is the Phase-B set of each run (the 11 named preferences' baselines plus every moved method
output), not the 77-point search set the text describes. Its points cluster around the 11
preferences, which matters for E1 and E5.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
P_COLS = [f"p_{a}" for a in ATTRIBUTES]
LAM_COLS = [f"lam_{a}" for a in ATTRIBUTES]

RUNS: dict[str, dict[str, Any]] = {
    "nb10_rs_ppo": {
        "dir": "results/nb10_run1_outputs", "lambda_table": "lambda_table.csv",
        "caches": {"armorm": "reward_cache.jsonl"},
        "note": "RS-PPO; ArmoRM was the PPO training reward (circular)",
    },
    "nb13_dpo_seed137": {
        "dir": "results/nb13_nb11_dpo_run1_outputs", "lambda_table": "lambda_table.csv",
        "caches": {"armorm": "reward_cache.jsonl"},
        "tensor_check": {"armorm": "nb13_reward_tensor.npy"},
        "note": "DPO, TinyLlama, seed 137",
    },
    "nb13_1_mean_dpo": {
        "dir": "results/nb13_1_nb11_1_mean_dpo_three_rms_run1_two_rms_provisional_outputs",
        "lambda_table": "lambda_table_evaluated.csv",
        "caches": {"armorm": "reward_cache_armorm.jsonl", "urm": "reward_cache_urm.jsonl"},
        "tensor_check": {"armorm": "reward_tensor_armorm.npy", "urm": "reward_tensor_urm.npy"},
        "note": "mean-DPO over 5 seeds, TinyLlama; URM is an unbounded sensitivity scorer",
    },
}


def point_key(vector: Sequence[float], decimals: int = 8) -> str:
    return "|".join(f"{x:.{decimals}f}" for x in (np.round(np.asarray(vector, float), decimals) + 0.0))


class RewardData:
    """Merge points (n_points, m), per-prompt scores (n_points, n_prompts, m), key -> row."""

    def __init__(self, points: np.ndarray, tensor: np.ndarray, index: dict[str, int]):
        self.points, self.tensor, self.index = points, tensor, index

    def row(self, vector: Sequence[float]) -> int:
        return self.index[point_key(vector)]


def load_cache(path: Path) -> RewardData:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header, records = lines[0], lines[1:]
    records.sort(key=lambda r: int(r["index"]))
    points = np.asarray([r["lambda"] for r in records], dtype=float)
    tensor = np.asarray([r["scores"] for r in records], dtype=float)
    n_heads = header.get("num_objectives", header.get("num_attributes"))
    expected = (len(records), header["num_prompts"], n_heads)
    if tensor.shape != expected:
        raise ValueError(f"{path}: tensor shape {tensor.shape} != {expected}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{path}: non-finite scores")
    index = {point_key(p): i for i, p in enumerate(points)}
    if len(index) != len(points):
        raise ValueError(f"{path}: duplicate merge points")
    return RewardData(points, tensor, index)


def vertex_rows(data: RewardData) -> list[int]:
    m = data.points.shape[1]
    return [data.row(np.eye(m)[k]) for k in range(m)]


def evaluated_rows(table: pd.DataFrame, data: RewardData) -> pd.DataFrame:
    """Usable, moved method rows whose own point and baseline point were both evaluated."""
    rows = table[table["usable"].astype(bool) & table["moved"].astype(bool)].copy()
    rows["i_lam"] = [data.index.get(point_key(v)) for v in rows[LAM_COLS].to_numpy(float)]
    rows["i_p"] = [data.index.get(point_key(v)) for v in rows[P_COLS].to_numpy(float)]
    rows = rows[rows["i_lam"].notna() & rows["i_p"].notna()].copy()
    rows[["i_lam", "i_p"]] = rows[["i_lam", "i_p"]].astype(int)
    rows["label"] = [f"{m}({p})" if isinstance(p, str) and p else str(m)
                     for m, p in zip(rows["method"], rows["params"])]
    return rows.reset_index(drop=True)


def bootstrap_draws(n_prompts: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_prompts, size=(n_boot, n_prompts))


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


# ------------------------------------------------------------------------------- E1

def linearity(means: np.ndarray, points: np.ndarray, vertices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """R2_vert and R2_OLS for every head; means has shape (n_points, m)."""
    m = points.shape[1]
    held_out = np.setdiff1d(np.arange(len(points)), np.asarray(vertices))
    vertex_values = means[np.asarray(vertices)]                  # (m vertices, m heads): row k = r(e_k)
    pred_vert = points @ vertex_values                           # affine on the simplex through the vertices
    coef, *_ = np.linalg.lstsq(points, means, rcond=None)        # sum(lambda)=1 makes this the affine OLS fit
    pred_ols = points @ coef
    r2_vert, r2_ols = np.empty(m), np.empty(m)
    for i in range(m):
        y = means[:, i]
        y_h = y[held_out]
        r2_vert[i] = 1 - np.sum((y_h - pred_vert[held_out, i]) ** 2) / np.sum((y_h - y_h.mean()) ** 2)
        r2_ols[i] = 1 - np.sum((y - pred_ols[:, i]) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2_vert, r2_ols


# ------------------------------------------------------------------------------- E2 to E5

def delta_m(means: np.ndarray, vertices: Sequence[int], weights: np.ndarray | None = None) -> np.ndarray:
    """Delta m% of every point against the specialists; weights default to uniform."""
    spec = np.diag(means[np.asarray(vertices)])                  # r_k(theta_k)
    rel = (means - spec) / spec * 100.0
    if weights is None:
        return rel.mean(axis=1)
    return rel @ weights


def tchebycheff(means: np.ndarray, p: np.ndarray, rows: Sequence[int], eps_t: float = 1e-8) -> np.ndarray:
    best, worst = means.max(axis=0), means.min(axis=0)
    active = p > 0
    short = (best - means[np.asarray(rows)]) / (best - worst + eps_t)
    return np.max((p * short)[:, active], axis=1)


def per_row_metrics(means: np.ndarray, rows: pd.DataFrame, vertices: Sequence[int],
                    positive_scale: bool) -> pd.DataFrame:
    out = []
    if positive_scale:
        spec = np.diag(means[np.asarray(vertices)])
        rel = (means - spec) / spec * 100.0                      # (n_points, m) relative change vs. specialists
        dm = rel.mean(axis=1)
    for _, r in rows.iterrows():
        p = r[P_COLS].to_numpy(float)
        u_all = means @ p
        u_lam, u_p = u_all[r["i_lam"]], u_all[r["i_p"]]
        reg_random = u_all.max() - u_all.mean()
        t_lam, t_p = tchebycheff(means, p, [r["i_lam"], r["i_p"]])
        record = {
            "p_name": r["p_name"], "method": r["method"], "label": r["label"],
            "delta_u_p": u_lam - u_p,
            "regret_ratio_method": (u_all.max() - u_lam) / reg_random,
            "regret_ratio_baseline": (u_all.max() - u_p) / reg_random,
            "t_norm_method": t_lam, "t_norm_baseline": t_p,
        }
        if positive_scale:
            record.update({"delta_m_method": dm[r["i_lam"]], "delta_m_baseline": dm[r["i_p"]],
                           "delta_m_p_method": float(rel[r["i_lam"]] @ p),
                           "delta_m_p_baseline": float(rel[r["i_p"]] @ p)})
        out.append(record)
    return pd.DataFrame(out)


def standard_errors(tensor: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """SE of Delta U_p over prompts, from the paired per-prompt differences."""
    ses = []
    for _, r in rows.iterrows():
        p = r[P_COLS].to_numpy(float)
        diff = (tensor[r["i_lam"]] - tensor[r["i_p"]]) @ p
        ses.append(diff.std(ddof=1) / np.sqrt(len(diff)))
    return np.asarray(ses)


# ------------------------------------------------------------------------------- E6

def floor_checks(M: np.ndarray, prefs: np.ndarray) -> dict[str, Any]:
    from src.coefficient_portfolio import floor_lp, floor_lp_at_p
    t_global, collapsed = floor_lp(M)
    per_p = [floor_lp_at_p(M, p) for p in prefs]
    return {"global_t": float(t_global), "global_collapsed": bool(collapsed),
            "n_preferences": len(per_p), "n_open": int(sum(not c for _, c in per_p))}


def empirical_dominance(data: RewardData, rows: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Per evaluated preference: points that beat lambda = p on all heads (point estimate and in
    at least 95 % of the bootstrap draws), and the largest worst-head gain among them."""
    means = data.tensor.mean(axis=1)
    boot_means = [data.tensor[:, idx, :].mean(axis=1) for idx in draws]
    out = []
    for p_name, group in rows.groupby("p_name"):
        i_p = int(group["i_p"].iloc[0])
        gain = means - means[i_p]
        dominating = np.all(gain > 0, axis=1)
        dominating[i_p] = False
        share = np.mean([np.all(b - b[i_p] > 0, axis=1) for b in boot_means], axis=0)
        share[i_p] = 0.0
        out.append({"p_name": p_name, "n_points": int(len(means) - 1),
                    "n_dominating": int(dominating.sum()), "n_dominating_95": int((share >= 0.95).sum()),
                    "best_worst_head_gain": float(gain[dominating].min(axis=1).max()) if dominating.any() else 0.0})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------- driver

def analyse(run: str, rm: str, data: RewardData, table: pd.DataFrame, R_cos: np.ndarray | None,
            n_boot: int, seed: int) -> dict[str, Any]:
    means = data.tensor.mean(axis=1)
    vertices = vertex_rows(data)
    rows = evaluated_rows(table, data)
    positive_scale = bool(np.all(np.diag(means[np.asarray(vertices)]) > 0)) and rm != "urm"
    draws = bootstrap_draws(data.tensor.shape[1], n_boot, seed)
    phase_a_prefs = np.unique(table[P_COLS].to_numpy(float).round(12), axis=0)

    # E1
    r2_vert, r2_ols = linearity(means, data.points, vertices)
    boot_vert, boot_ols = [], []
    for idx in draws:
        bv, bo = linearity(data.tensor[:, idx, :].mean(axis=1), data.points, vertices)
        boot_vert.append(bv)
        boot_ols.append(bo)
    boot_vert, boot_ols = np.asarray(boot_vert), np.asarray(boot_ols)
    e1 = pd.DataFrame({
        "head": ATTRIBUTES, "R2_vert": r2_vert,
        "R2_vert_ci_low": [percentile_ci(boot_vert[:, i])[0] for i in range(5)],
        "R2_vert_ci_high": [percentile_ci(boot_vert[:, i])[1] for i in range(5)],
        "R2_OLS": r2_ols,
        "R2_OLS_ci_low": [percentile_ci(boot_ols[:, i])[0] for i in range(5)],
        "R2_OLS_ci_high": [percentile_ci(boot_ols[:, i])[1] for i in range(5)],
        "max_abs_pearson_any_affine_score": np.sqrt(np.clip(r2_ols, 0, None)),
    })

    # E2 to E5, point estimates per row
    per_row = per_row_metrics(means, rows, vertices, positive_scale)
    per_row["se_delta_u_p"] = standard_errors(data.tensor, rows)
    per_row["z_delta_u_p"] = per_row["delta_u_p"] / per_row["se_delta_u_p"]
    per_row["win"] = per_row["delta_u_p"] > 0
    per_row["t_norm_change"] = per_row["t_norm_method"] - per_row["t_norm_baseline"]

    # bootstrap of the per-method aggregates (same prompt draw for every point)
    agg_cols = ["delta_u_p", "regret_ratio_method", "t_norm_change"] + (
        ["delta_m_change"] if positive_scale else [])
    boot_by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for idx in draws[: min(n_boot, 1000)]:
        b = per_row_metrics(data.tensor[:, idx, :].mean(axis=1), rows, vertices, positive_scale)
        b["t_norm_change"] = b["t_norm_method"] - b["t_norm_baseline"]
        if positive_scale:
            b["delta_m_change"] = b["delta_m_method"] - b["delta_m_baseline"]
        for method, group in b.groupby("method"):
            for col in agg_cols:
                boot_by_method[method][col].append(float(group[col].mean()))
    if positive_scale:
        per_row["delta_m_change"] = per_row["delta_m_method"] - per_row["delta_m_baseline"]

    summary_rows = []
    for method, group in per_row.groupby("method"):
        rec = {"method": method, "n_rows": len(group), "n_configs": group["label"].nunique(),
               "win_rate": float(group["win"].mean()),
               "median_z": float(group["z_delta_u_p"].median()),
               "max_abs_z": float(group["z_delta_u_p"].abs().max()),
               "share_t_norm_worse": float((group["t_norm_change"] > 0).mean()),
               "median_regret_ratio_baseline": float(group["regret_ratio_baseline"].median())}
        for col in agg_cols:
            rec[f"mean_{col}"] = float(group[col].mean())
            lo, hi = percentile_ci(np.asarray(boot_by_method[method][col]))
            rec[f"mean_{col}_ci_low"], rec[f"mean_{col}_ci_high"] = lo, hi
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)

    # win rate per configuration over preferences, with a bootstrap over preferences
    rng = np.random.default_rng(seed + 1)
    win_rows = []
    for label, group in per_row.groupby("label"):
        wins = group["win"].to_numpy(bool)
        boots = [wins[rng.integers(0, len(wins), len(wins))].mean() for _ in range(2000)]
        lo, hi = percentile_ci(np.asarray(boots))
        win_rows.append({"label": label, "method": group["method"].iloc[0], "n_preferences": len(wins),
                         "win_rate": float(wins.mean()), "ci_low": lo, "ci_high": hi})
    win_df = pd.DataFrame(win_rows).sort_values(["method", "label"])

    # E6
    M = means[np.asarray(vertices)].T                            # M_ij = r_i(theta_j)
    h_floor = floor_checks(M, phase_a_prefs)
    boot_open = []
    for idx in draws[: min(n_boot, 500)]:
        Mb = data.tensor[np.asarray(vertices)][:, idx, :].mean(axis=1).T
        boot_open.append(not floor_checks(Mb, phase_a_prefs[:1])["global_collapsed"])
    h_floor["bootstrap_share_global_open"] = float(np.mean(boot_open))
    h_floor["n_bootstrap"] = len(boot_open)
    if R_cos is not None:
        r_floor = floor_checks(R_cos, phase_a_prefs)
        h_floor["R_cos_global_collapsed"] = r_floor["global_collapsed"]
        h_floor["R_cos_n_open"] = r_floor["n_open"]
    dominance = empirical_dominance(data, rows, draws[: min(n_boot, 500)])
    h_floor["empirical_prefs_with_dominating_point"] = int((dominance["n_dominating"] > 0).sum())
    h_floor["empirical_prefs_with_dominating_point_95"] = int((dominance["n_dominating_95"] > 0).sum())
    h_floor["empirical_n_prefs"] = int(len(dominance))
    h_floor["vertex_matrix"] = M.tolist()
    h_floor["column_sums"] = M.sum(axis=0).tolist()

    return {"run": run, "rm": rm, "n_points": int(len(data.points)), "n_prompts": int(data.tensor.shape[1]),
            "n_method_rows": int(len(rows)), "positive_scale": positive_scale,
            "e1": e1, "per_row": per_row, "dominance": dominance, "summary": summary, "win_rates": win_df, "h_floor": h_floor}


def load_R_cos(run_dir: Path) -> np.ndarray | None:
    for name in ("R_dpo_cos.csv", "R_mean_dpo_cos.csv", "R_cos.csv"):
        path = run_dir / name
        if path.is_file():
            return pd.read_csv(path, index_col=0).loc[list(ATTRIBUTES), list(ATTRIBUTES)].to_numpy(float)
    return None


def run_all(project_root: Path, out_dir: Path, n_boot: int = 2000, seed: int = 1717,
            runs: Sequence[str] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results, overview = {}, []
    for run in runs or RUNS:
        spec = RUNS[run]
        run_dir = project_root / spec["dir"]
        table = pd.read_csv(run_dir / spec["lambda_table"])
        R_cos = load_R_cos(run_dir)
        for rm, cache_name in spec["caches"].items():
            data = load_cache(run_dir / cache_name)
            check = spec.get("tensor_check", {}).get(rm)
            if check and (run_dir / check).is_file():
                stored = np.load(run_dir / check)
                if stored.shape != data.tensor.shape or not np.allclose(stored, data.tensor):
                    raise ValueError(f"{run}/{rm}: cache and stored tensor disagree")
            res = analyse(run, rm, data, table, R_cos, n_boot, seed)
            tag = f"{run}__{rm}"
            target = out_dir / tag
            target.mkdir(parents=True, exist_ok=True)
            res["e1"].to_csv(target / "E1_linearity.csv", index=False)
            res["per_row"].to_csv(target / "E2_E5_per_row.csv", index=False)
            res["summary"].to_csv(target / "E2_E5_summary_by_method.csv", index=False)
            res["win_rates"].to_csv(target / "E3_win_rate_by_config.csv", index=False)
            res["dominance"].to_csv(target / "E6_empirical_dominance.csv", index=False)
            (target / "E6_h_floor.json").write_text(json.dumps(res["h_floor"], indent=2) + "\n")
            results[tag] = res
            overview.append({
                "run": run, "rm": rm, "note": spec["note"], "n_points": res["n_points"],
                "n_method_rows": res["n_method_rows"],
                "R2_OLS_min": float(res["e1"]["R2_OLS"].min()), "R2_OLS_max": float(res["e1"]["R2_OLS"].max()),
                "R2_vert_min": float(res["e1"]["R2_vert"].min()), "R2_vert_max": float(res["e1"]["R2_vert"].max()),
                "win_rate_all_rows": float(res["per_row"]["win"].mean()),
                "max_abs_z": float(res["per_row"]["z_delta_u_p"].abs().max()),
                "median_regret_ratio_methods": float(res["per_row"]["regret_ratio_method"].median()),
                "median_regret_ratio_baseline": float(res["per_row"]["regret_ratio_baseline"].median()),
                "share_t_norm_worse": float((res["per_row"]["t_norm_change"] > 0).mean()),
                "H_floor_global_collapsed": res["h_floor"]["global_collapsed"],
                "H_floor_open_prefs": f'{res["h_floor"]["n_open"]}/{res["h_floor"]["n_preferences"]}',
                "H_floor_bootstrap_share_open": res["h_floor"]["bootstrap_share_global_open"],
                "R_floor_open_prefs": res["h_floor"].get("R_cos_n_open"),
                "empirical_dominating_point": f'{res["h_floor"]["empirical_prefs_with_dominating_point"]}/{res["h_floor"]["empirical_n_prefs"]}',
                "empirical_dominating_point_95": f'{res["h_floor"]["empirical_prefs_with_dominating_point_95"]}/{res["h_floor"]["empirical_n_prefs"]}',
            })
    overview_df = pd.DataFrame(overview)
    overview_df.to_csv(out_dir / "overview.csv", index=False)
    (out_dir / "nb17_report.json").write_text(json.dumps({
        "schema_version": 1, "n_boot": n_boot, "seed": seed,
        "bootstrap": "paired over prompts; the same prompt draw for every merge point; percentile 95% CI",
        "evaluated_set_note": "B = Phase-B evaluated points of each run, not the 77-point search set",
        "runs": {k: {kk: vv for kk, vv in v.items() if kk in ("run", "rm", "n_points", "n_prompts",
                                                               "n_method_rows", "positive_scale", "h_floor")}
                 for k, v in results.items()},
    }, indent=2, default=float) + "\n")
    return {"overview": overview_df, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--out_dir", default="results/nb17_promised_metrics")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    out = run_all(Path(args.project_root), Path(args.out_dir), args.n_boot, args.seed)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out["overview"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
