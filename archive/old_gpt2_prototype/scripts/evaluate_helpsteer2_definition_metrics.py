"""Compute Definition-3.17-style metrics for HelpSteer2 prototype results.

The input rewards are the existing lightweight HelpSteer2 proxy scores. They
are treated as normalized rewards ``r_tilde`` for this prototype evaluation.
They are not HelpSteer2 human labels and not reward-model scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_utils import load_labeled_relationship_matrix
from src.helpsteer2_scoring_utils import OBJECTIVES


P_COLUMNS = tuple(f"p_{objective}" for objective in OBJECTIVES)
LAMBDA_COLUMNS = tuple(f"lambda_{objective}" for objective in OBJECTIVES)
REWARD_COLUMNS = tuple(f"mean_{objective}_proxy" for objective in OBJECTIVES)
OUTPUT_REWARD_COLUMNS = tuple(f"reward_{objective}" for objective in OBJECTIVES)
SHORTFALL_COLUMNS = tuple(
    f"normalized_shortfall_{objective}" for objective in OBJECTIVES
)
Z_BEST_COLUMNS = tuple(f"z_best_{objective}" for objective in OBJECTIVES)
Z_WORST_COLUMNS = tuple(f"z_worst_{objective}" for objective in OBJECTIVES)

REQUIRED_COLUMNS = {
    "preference_name",
    "method",
    "hyperparameter_name",
    "hyperparameter_value",
    *P_COLUMNS,
    *LAMBDA_COLUMNS,
    *REWARD_COLUMNS,
}


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_comparison(path: Path) -> pd.DataFrame:
    """Load and validate the aggregated HelpSteer2 comparison CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Comparison file not found: {path}. "
            "Run scripts/evaluate_helpsteer2_m1_c1_merges.py first."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Comparison CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError(f"Comparison CSV contains no rows: {path}")

    frame = frame.copy()
    frame["preference_name"] = frame["preference_name"].astype(str).str.strip()
    frame["method"] = frame["method"].astype(str).str.strip()
    frame["hyperparameter_name"] = (
        frame["hyperparameter_name"].fillna("").astype(str).str.strip()
    )
    frame["hyperparameter_value"] = pd.to_numeric(
        frame["hyperparameter_value"],
        errors="coerce",
    )

    numeric_columns = [
        *P_COLUMNS,
        *LAMBDA_COLUMNS,
        *REWARD_COLUMNS,
    ]
    optional_numeric_columns = [
        "l1_distance_to_p",
        "l2_distance_to_p",
        "best_fixed_sweep_utility",
    ]
    for column in optional_numeric_columns:
        if column in frame.columns:
            numeric_columns.append(column)

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must contain only finite values.")

    for prefix, columns in [("Preference", P_COLUMNS), ("Lambda", LAMBDA_COLUMNS)]:
        sums = frame[list(columns)].sum(axis=1).to_numpy(dtype=float)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{prefix} vectors must sum to 1.")
        values = frame[list(columns)].to_numpy(dtype=float)
        if np.any(values < -1e-12):
            raise ValueError(f"{prefix} vectors must be non-negative.")

    return frame


def load_optional_relationship_matrix(path: Path) -> np.ndarray | None:
    """Load a labeled relationship matrix if it is available."""
    if not path.is_file():
        print(
            f"Relationship matrix not found at {path}. "
            "R-geometric distances will be left empty."
        )
        return None
    matrix = load_labeled_relationship_matrix(path, OBJECTIVES)
    return 0.5 * (matrix + matrix.T)


def vector_from_row(row: pd.Series, columns: tuple[str, ...]) -> np.ndarray:
    """Read a row vector from ordered DataFrame columns."""
    return row.loc[list(columns)].to_numpy(dtype=float)


def get_direct_preference_rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the direct-preference baseline row for each preference."""
    direct_rows: dict[str, pd.Series] = {}
    for preference_name, group in frame.groupby("preference_name", sort=False):
        direct = group.loc[group["method"] == "direct_preference"]
        if direct.empty:
            raise ValueError(
                f"Missing direct_preference baseline for {preference_name!r}."
            )
        direct_rows[preference_name] = direct.iloc[0]
    return direct_rows


def utility_best_for_preference(
    group: pd.DataFrame,
    direct_row: pd.Series,
) -> tuple[float, str]:
    """Return the finite-sweep best utility when available, otherwise local best."""
    if "best_fixed_sweep_utility" in group.columns:
        values = pd.to_numeric(
            group["best_fixed_sweep_utility"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            return float(values.iloc[0]), "fixed_lambda_sweep"

    rewards = group[list(REWARD_COLUMNS)].to_numpy(dtype=float)
    preference = vector_from_row(direct_row, P_COLUMNS)
    utilities = rewards @ preference
    return float(np.max(utilities)), "input_comparison_set"


def compute_rows(
    frame: pd.DataFrame,
    relationship_matrix: np.ndarray | None,
    eps: float,
) -> pd.DataFrame:
    """Compute all requested definition-style metrics."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive value.")

    direct_rows = get_direct_preference_rows(frame)
    output_rows: list[dict[str, object]] = []

    for preference_name, group in frame.groupby("preference_name", sort=False):
        direct_row = direct_rows[preference_name]
        direct_rewards = vector_from_row(direct_row, REWARD_COLUMNS)
        preference = vector_from_row(direct_row, P_COLUMNS)
        direct_utility = float(preference @ direct_rewards)
        lambda_best_utility, lambda_best_source = utility_best_for_preference(
            group,
            direct_row,
        )

        benchmark_rewards = group[list(REWARD_COLUMNS)].to_numpy(dtype=float)
        z_best = benchmark_rewards.max(axis=0)
        z_worst = benchmark_rewards.min(axis=0)
        denominator = z_best - z_worst + eps
        direct_shortfalls = preference * (
            (z_best - direct_rewards) / denominator
        )
        direct_tchebychev = float(np.max(direct_shortfalls))

        for _, row in group.iterrows():
            rewards = vector_from_row(row, REWARD_COLUMNS)
            lambdas = vector_from_row(row, LAMBDA_COLUMNS)
            difference = lambdas - preference
            utility = float(preference @ rewards)
            avg_reward = float(np.mean(rewards))
            shortfalls = preference * ((z_best - rewards) / denominator)
            tchebychev = float(np.max(shortfalls))

            if "l1_distance_to_p" in row and pd.notna(row["l1_distance_to_p"]):
                l1_distance = float(row["l1_distance_to_p"])
            else:
                l1_distance = float(np.linalg.norm(difference, ord=1))

            if "l2_distance_to_p" in row and pd.notna(row["l2_distance_to_p"]):
                l2_distance = float(row["l2_distance_to_p"])
            else:
                l2_distance = float(np.linalg.norm(difference, ord=2))

            r_geometric_distance = (
                float(difference @ relationship_matrix @ difference)
                if relationship_matrix is not None
                else np.nan
            )

            output_row: dict[str, object] = {
                "preference_name": preference_name,
                "method": row["method"],
                "hyperparameter_name": row["hyperparameter_name"],
                "hyperparameter_value": row["hyperparameter_value"],
                **dict(zip(P_COLUMNS, preference)),
                **dict(zip(LAMBDA_COLUMNS, lambdas)),
                **dict(zip(OUTPUT_REWARD_COLUMNS, rewards)),
                "avg_reward": avg_reward,
                "preference_weighted_utility": utility,
                "direct_preference_utility": direct_utility,
                "delta_utility_over_direct": utility - direct_utility,
                "lambda_best_utility": lambda_best_utility,
                "lambda_best_source": lambda_best_source,
                "utility_gap_to_lambda_best": lambda_best_utility - utility,
                "l1_distance_to_p": l1_distance,
                "l2_distance_to_p": l2_distance,
                "r_geometric_distance_to_p": r_geometric_distance,
                **dict(zip(Z_BEST_COLUMNS, z_best)),
                **dict(zip(Z_WORST_COLUMNS, z_worst)),
                **dict(zip(SHORTFALL_COLUMNS, shortfalls)),
                "tchebychev_norm": tchebychev,
                "direct_preference_tchebychev_norm": direct_tchebychev,
                "delta_tchebychev_norm_over_direct": (
                    direct_tchebychev - tchebychev
                ),
            }
            output_rows.append(output_row)

    return pd.DataFrame(output_rows)


def write_markdown_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    """Write a short human-readable summary of the definition metrics."""
    best_utility = (
        metrics.sort_values(
            ["preference_name", "preference_weighted_utility"],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates("preference_name", keep="first")
        .reset_index(drop=True)
    )
    best_tchebychev = (
        metrics.sort_values(
            ["preference_name", "tchebychev_norm"],
            ascending=[True, True],
            kind="stable",
        )
        .drop_duplicates("preference_name", keep="first")
        .reset_index(drop=True)
    )

    lines = [
        "# HelpSteer2 Definition-Style Evaluation Metrics",
        "",
        "This file summarizes the metrics corresponding to the evaluation "
        "definition used in the thesis draft. The existing HelpSteer2 proxy "
        "scores are treated as normalized rewards $\\tilde r_i$.",
        "",
        "The scores are lightweight heuristic proxy scores. They are not "
        "HelpSteer2 human labels and not reward-model scores.",
        "",
        "## Best Utility by Preference",
        "",
        "| Preference | Best method | Setting | $U_p$ | "
        "$\\Delta U_p$ vs direct | Gap to $\\lambda_{\\mathrm{best}}$ |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in best_utility.iterrows():
        setting = format_setting(row)
        lines.append(
            f"| {pretty_name(row['preference_name'])} "
            f"| {row['method']} "
            f"| {setting} "
            f"| {row['preference_weighted_utility']:.6f} "
            f"| {row['delta_utility_over_direct']:+.6f} "
            f"| {row['utility_gap_to_lambda_best']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Normalized Tchebychev Score by Preference",
            "",
            "Smaller $T_p^{\\mathrm{norm}}$ values are better. A positive "
            "$\\Delta T_p^{\\mathrm{norm}}$ means the method reduces the "
            "worst preference-weighted objective shortfall relative to "
            "$\\lambda=p$.",
            "",
            "| Preference | Best method | Setting | $T_p^{\\mathrm{norm}}$ | "
            "$\\Delta T_p^{\\mathrm{norm}}$ vs direct |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for _, row in best_tchebychev.iterrows():
        setting = format_setting(row)
        lines.append(
            f"| {pretty_name(row['preference_name'])} "
            f"| {row['method']} "
            f"| {setting} "
            f"| {row['tchebychev_norm']:.6f} "
            f"| {row['delta_tchebychev_norm_over_direct']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `avg_reward` is the arithmetic mean over the five objective "
            "proxy rewards.",
            "- `preference_weighted_utility` is $U_p=\\sum_i p_i\\tilde r_i$.",
            "- `delta_utility_over_direct` compares against the "
            "`direct_preference` row for the same preference vector.",
            "- `utility_gap_to_lambda_best` uses the fixed finite-sweep "
            "reference when available.",
            "- `r_geometric_distance_to_p` is "
            "$(\\lambda-p)^TR(\\lambda-p)$ when the relationship matrix is "
            "available.",
            "- `z_best_*` and `z_worst_*` are the per-objective benchmark "
            "values used to normalize the Tchebychev shortfalls.",
            "- Tchebychev best and worst values are computed over the rows in "
            "the input comparison table.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def pretty_name(value: object) -> str:
    """Return a readable preference name."""
    return str(value).replace("_", " ").title()


def format_setting(row: pd.Series) -> str:
    """Format a hyperparameter setting for Markdown output."""
    name = str(row.get("hyperparameter_name", "")).strip()
    value = row.get("hyperparameter_value", np.nan)
    if not name or pd.isna(value):
        return "-"
    return f"{name}={float(value):g}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute Definition-3.17-style metrics for the HelpSteer2 "
            "M1/C1 prototype comparison."
        )
    )
    parser.add_argument(
        "--comparison_path",
        default="results/helpsteer2_m1_c1_comparison.csv",
    )
    parser.add_argument(
        "--relationship_matrix_path",
        default="results/helpsteer2_relationship_matrix.csv",
    )
    parser.add_argument(
        "--output_path",
        default="results/helpsteer2_definition_metrics.csv",
    )
    parser.add_argument(
        "--summary_output_path",
        default="results/helpsteer2_definition_metrics_summary.md",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Small positive denominator stabilizer for Tchebychev scores.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the metric computation."""
    args = parse_args()
    comparison_path = resolve_project_path(args.comparison_path)
    relationship_matrix_path = resolve_project_path(
        args.relationship_matrix_path
    )
    output_path = resolve_project_path(args.output_path)
    summary_output_path = resolve_project_path(args.summary_output_path)

    comparison = load_comparison(comparison_path)
    relationship_matrix = load_optional_relationship_matrix(
        relationship_matrix_path
    )
    metrics = compute_rows(comparison, relationship_matrix, eps=args.eps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False, float_format="%.10f")
    write_markdown_summary(metrics, summary_output_path)

    print(f"Definition-style metrics written to: {output_path}")
    print(f"Metric summary written to: {summary_output_path}")


if __name__ == "__main__":
    main()
