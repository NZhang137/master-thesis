"""Create compact tables and plots for the HelpSteer2 M1/C1 comparison.

The input scores are lightweight heuristic proxy scores. They are not
HelpSteer2 human labels or reward-model scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = ("uniform", "direct_preference", "M1", "C1")
METHOD_LABELS = {
    "uniform": "Uniform",
    "direct_preference": "Direct preference",
    "M1": "M1",
    "C1": "C1",
}
METHOD_COLORS = {
    "uniform": "#7A7A7A",
    "direct_preference": "#2F6B9A",
    "M1": "#2A9D8F",
    "C1": "#D97706",
}

SUMMARY_COLUMNS = [
    "preference_name",
    "method",
    "best_hyperparameter_name",
    "best_hyperparameter_value",
    "utility_for_preference",
    "mean_helpfulness_proxy",
    "mean_correctness_proxy",
    "mean_coherence_proxy",
    "mean_complexity_proxy",
    "mean_verbosity_proxy",
    "l1_distance_to_p",
    "l2_distance_to_p",
    "min_relationship_score",
]

BEST_METHOD_COLUMNS = [
    "preference_name",
    "best_method",
    "best_hyperparameter_name",
    "best_hyperparameter_value",
    "best_utility_for_preference",
    "direct_preference_utility",
    "improvement_over_direct_preference",
    "uniform_utility",
    "improvement_over_uniform",
]

REQUIRED_COLUMNS = {
    "preference_name",
    "method",
    "hyperparameter_name",
    "hyperparameter_value",
    "utility_for_preference",
    "mean_helpfulness_proxy",
    "mean_correctness_proxy",
    "mean_coherence_proxy",
    "mean_complexity_proxy",
    "mean_verbosity_proxy",
    "l1_distance_to_p",
    "l2_distance_to_p",
    "min_relationship_score",
}


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_comparison(path: Path) -> pd.DataFrame:
    """Load and validate the response-level comparison table."""
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
    if (frame["preference_name"] == "").any():
        raise ValueError("preference_name values must be non-empty.")

    unsupported = sorted(set(frame["method"]).difference(METHOD_ORDER))
    if unsupported:
        raise ValueError(
            "Unsupported methods in comparison CSV: " + ", ".join(unsupported)
        )

    numeric_columns = [
        "utility_for_preference",
        "mean_helpfulness_proxy",
        "mean_correctness_proxy",
        "mean_coherence_proxy",
        "mean_complexity_proxy",
        "mean_verbosity_proxy",
        "l1_distance_to_p",
        "l2_distance_to_p",
        "min_relationship_score",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must contain only finite values.")

    frame["hyperparameter_value"] = pd.to_numeric(
        frame["hyperparameter_value"],
        errors="coerce",
    )
    return frame


def select_best_setting_per_method(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-utility setting for each preference and method."""
    preference_order = list(dict.fromkeys(frame["preference_name"]))
    working = frame.reset_index(names="_source_order").copy()
    working["_hyperparameter_sort"] = working["hyperparameter_value"].fillna(
        np.inf
    )
    working = working.sort_values(
        [
            "preference_name",
            "method",
            "utility_for_preference",
            "_hyperparameter_sort",
            "_source_order",
        ],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    selected = working.drop_duplicates(
        subset=["preference_name", "method"],
        keep="first",
    ).copy()

    selected["preference_name"] = pd.Categorical(
        selected["preference_name"],
        categories=preference_order,
        ordered=True,
    )
    selected["method"] = pd.Categorical(
        selected["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    selected = selected.sort_values(
        ["preference_name", "method"],
        kind="stable",
    )

    selected = selected.rename(
        columns={
            "hyperparameter_name": "best_hyperparameter_name",
            "hyperparameter_value": "best_hyperparameter_value",
        }
    )
    selected["best_hyperparameter_name"] = (
        selected["best_hyperparameter_name"].fillna("").astype(str)
    )
    return selected[SUMMARY_COLUMNS].reset_index(drop=True)


def build_best_method_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Identify the highest-utility selected method for each preference."""
    rows: list[dict[str, object]] = []
    for preference_name, group in summary.groupby(
        "preference_name",
        sort=False,
        observed=True,
    ):
        methods = set(group["method"].astype(str))
        missing = {"direct_preference", "uniform"}.difference(methods)
        if missing:
            raise ValueError(
                f"{preference_name!r} is missing baseline methods: "
                + ", ".join(sorted(missing))
            )

        ordered = group.copy()
        ordered["_method_order"] = ordered["method"].astype(str).map(
            {method: index for index, method in enumerate(METHOD_ORDER)}
        )
        ordered = ordered.sort_values(
            ["utility_for_preference", "_method_order"],
            ascending=[False, True],
            kind="stable",
        )
        best = ordered.iloc[0]
        direct_utility = float(
            group.loc[
                group["method"].astype(str) == "direct_preference",
                "utility_for_preference",
            ].iloc[0]
        )
        uniform_utility = float(
            group.loc[
                group["method"].astype(str) == "uniform",
                "utility_for_preference",
            ].iloc[0]
        )
        best_utility = float(best["utility_for_preference"])
        rows.append(
            {
                "preference_name": str(preference_name),
                "best_method": str(best["method"]),
                "best_hyperparameter_name": best[
                    "best_hyperparameter_name"
                ],
                "best_hyperparameter_value": best[
                    "best_hyperparameter_value"
                ],
                "best_utility_for_preference": best_utility,
                "direct_preference_utility": direct_utility,
                "improvement_over_direct_preference": (
                    best_utility - direct_utility
                ),
                "uniform_utility": uniform_utility,
                "improvement_over_uniform": best_utility - uniform_utility,
            }
        )

    return pd.DataFrame(rows, columns=BEST_METHOD_COLUMNS)


def pretty_preference(name: str) -> str:
    """Convert an internal preference name to a readable chart label."""
    return name.replace("_", " ").title()


def create_grouped_bar_plot(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    y_label: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Create one grouped bar chart from the selected method settings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preferences = list(
        dict.fromkeys(summary["preference_name"].astype(str).tolist())
    )
    methods = [
        method
        for method in METHOD_ORDER
        if method in set(summary["method"].astype(str))
    ]
    x_positions = np.arange(len(preferences), dtype=float)
    total_width = 0.8
    bar_width = total_width / max(len(methods), 1)

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for method_index, method in enumerate(methods):
        method_rows = (
            summary.loc[summary["method"].astype(str) == method]
            .set_index(summary.loc[
                summary["method"].astype(str) == method,
                "preference_name",
            ].astype(str))
            .reindex(preferences)
        )
        values = method_rows[value_column].to_numpy(dtype=float)
        offset = (method_index - (len(methods) - 1) / 2) * bar_width
        axis.bar(
            x_positions + offset,
            values,
            width=bar_width * 0.9,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.6,
        )

    axis.set_title(title, fontsize=14, pad=12)
    axis.set_ylabel(y_label)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [pretty_preference(name) for name in preferences],
        rotation=0,
    )
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    else:
        maximum = float(summary[value_column].max())
        axis.set_ylim(0.0, max(maximum * 1.15, 0.05))
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(methods),
        frameon=False,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_plots(summary: pd.DataFrame, plots_dir: Path) -> None:
    """Create the three requested thesis-oriented comparison plots."""
    create_grouped_bar_plot(
        summary,
        value_column="utility_for_preference",
        title="HelpSteer2 Proxy Utility by Method",
        y_label="Preference-weighted proxy utility",
        output_path=plots_dir / "helpsteer2_utility_by_method.png",
        y_limits=(0.0, 1.0),
    )
    create_grouped_bar_plot(
        summary,
        value_column="l1_distance_to_p",
        title="Distance from the Original Preference Vector",
        y_label="L1 distance to p",
        output_path=plots_dir / "helpsteer2_distance_to_preference.png",
    )
    create_grouped_bar_plot(
        summary,
        value_column="min_relationship_score",
        title="Minimum Relationship Score by Method",
        y_label="Minimum component of R lambda",
        output_path=plots_dir / "helpsteer2_min_relationship_score.png",
        y_limits=(0.0, 1.0),
    )


def format_hyperparameter(row: pd.Series) -> str:
    """Format the selected hyperparameter for a Markdown table."""
    name = str(row["best_hyperparameter_name"]).strip()
    value = row["best_hyperparameter_value"]
    if not name or pd.isna(value):
        return "-"
    return f"{name}={float(value):g}"


def write_markdown_summary(
    summary: pd.DataFrame,
    best_methods: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a concise, data-backed interpretation of the comparison."""
    lines = [
        "# HelpSteer2 M1/C1 Result Summary",
        "",
        "## Comparison",
        "",
        "The comparison evaluates uniform coefficients, direct preference "
        "(\\(\\lambda=p\\)), M1 relationship-softmax correction, and the C1 "
        "CAGrad-inspired one-shot mapping. For M1 and C1, the table selects "
        "the hyperparameter setting with the highest "
        "`utility_for_preference` for each preference vector.",
        "",
        "## Best Method by Preference",
        "",
        "| Preference | Best method | Setting | Proxy utility | "
        "Improvement over direct | Improvement over uniform |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in best_methods.iterrows():
        lines.append(
            f"| {pretty_preference(row['preference_name'])} "
            f"| {METHOD_LABELS[row['best_method']]} "
            f"| {format_hyperparameter(row)} "
            f"| {row['best_utility_for_preference']:.6f} "
            f"| {row['improvement_over_direct_preference']:+.6f} "
            f"| {row['improvement_over_uniform']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## M1 and C1 Relative to Direct Preference",
            "",
        ]
    )
    for preference_name, group in summary.groupby(
        "preference_name",
        sort=False,
        observed=True,
    ):
        indexed = group.assign(
            _method=group["method"].astype(str)
        ).set_index("_method")
        direct = float(
            indexed.loc["direct_preference", "utility_for_preference"]
        )
        m1 = indexed.loc["M1"]
        c1 = indexed.loc["C1"]
        m1_delta = float(m1["utility_for_preference"]) - direct
        c1_delta = float(c1["utility_for_preference"]) - direct
        lines.append(
            f"- **{pretty_preference(str(preference_name))}:** "
            f"best M1 ({format_hyperparameter(m1)}) changes proxy utility "
            f"by {m1_delta:+.6f}; best C1 ({format_hyperparameter(c1)}) "
            f"changes it by {c1_delta:+.6f} relative to direct preference."
        )

    m1_rows = summary.loc[summary["method"].astype(str) == "M1"]
    c1_rows = summary.loc[summary["method"].astype(str) == "C1"]
    lines.extend(
        [
            "",
            "## Distance from the Original Preference",
            "",
            "The selected M1 settings remain close to the original preference "
            f"vectors (L1 distance {m1_rows['l1_distance_to_p'].min():.4f}"
            f"-{m1_rows['l1_distance_to_p'].max():.4f}; L2 distance "
            f"{m1_rows['l2_distance_to_p'].min():.4f}"
            f"-{m1_rows['l2_distance_to_p'].max():.4f}). The selected C1 "
            "settings move farther "
            f"(L1 distance {c1_rows['l1_distance_to_p'].min():.4f}"
            f"-{c1_rows['l1_distance_to_p'].max():.4f}; L2 distance "
            f"{c1_rows['l2_distance_to_p'].min():.4f}"
            f"-{c1_rows['l2_distance_to_p'].max():.4f}). This describes "
            "coefficient displacement only; it is not evidence of general "
            "model quality.",
            "",
            "## Limitations",
            "",
            "- These are lightweight heuristic proxy scores.",
            "- They are not HelpSteer2 human labels.",
            "- They are not reward-model scores.",
            "- Generated responses do not automatically have HelpSteer2 labels.",
            "- The comparison uses a small prompt set and a single recorded run.",
            "- Hyperparameters are selected on the same proxy evaluation used "
            "for reporting.",
            "- Any finite-sweep reference should be described as "
            "\\(\\lambda_{\\mathrm{best}}\\), not as an oracle.",
            "- The results do not establish improvement of the global Pareto "
            "front.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Summarize the HelpSteer2 M1/C1 proxy comparison."
    )
    parser.add_argument(
        "--input_path",
        default="results/helpsteer2_m1_c1_comparison.csv",
    )
    parser.add_argument(
        "--summary_table_path",
        default="results/helpsteer2_method_summary_table.csv",
    )
    parser.add_argument(
        "--best_methods_path",
        default="results/helpsteer2_best_methods_by_preference.csv",
    )
    parser.add_argument(
        "--markdown_path",
        default="results/helpsteer2_m1_c1_result_summary.md",
    )
    parser.add_argument(
        "--plots_dir",
        default="results/plots",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Create tables and Markdown without rendering PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Create summary tables, plots, and the Markdown interpretation."""
    args = parse_args()
    input_path = resolve_project_path(args.input_path)
    summary_path = resolve_project_path(args.summary_table_path)
    best_methods_path = resolve_project_path(args.best_methods_path)
    markdown_path = resolve_project_path(args.markdown_path)
    plots_dir = resolve_project_path(args.plots_dir)

    comparison = load_comparison(input_path)
    summary = select_best_setting_per_method(comparison)
    best_methods = build_best_method_table(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    best_methods_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    best_methods.to_csv(
        best_methods_path,
        index=False,
        float_format="%.10f",
    )
    write_markdown_summary(summary, best_methods, markdown_path)

    if not args.skip_plots:
        create_plots(summary, plots_dir)

    print(f"Method summary: {summary_path}")
    print(f"Best methods: {best_methods_path}")
    print(f"Markdown summary: {markdown_path}")
    if args.skip_plots:
        print("Plots skipped by request.")
    else:
        print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
