"""Evaluate fixed HelpSteer2 adapter merges with lightweight proxy scores.

The five scores in this script are deterministic placeholder heuristics. They
are not reward-model scores, factual correctness measurements, human
HelpSteer2 labels, or final thesis evaluation results. The script only checks
that the many-objective evaluation pipeline works on an existing fixed set of
adapter merges.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helpsteer2_scoring_utils import PROXY_COLUMNS, score_response

ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

LAMBDA_COLUMNS = tuple(f"lambda_{attribute}" for attribute in ATTRIBUTES)
REQUIRED_COLUMNS = {
    "merge_name",
    *LAMBDA_COLUMNS,
    "prompt",
    "generated_response",
}

PREFERENCES = {
    "utility_balanced": (0.2, 0.2, 0.2, 0.2, 0.2),
    "utility_quality_focused": (0.3, 0.3, 0.3, 0.05, 0.05),
    "utility_detailed_answer": (0.25, 0.25, 0.2, 0.15, 0.15),
    "utility_helpfulness_focused": (0.6, 0.1, 0.1, 0.1, 0.1),
}

def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_generations(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the HelpSteer2 merge-generation CSV."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}. "
            "Run scripts/evaluate_helpsteer2_adapter_merges.py first."
        )

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV contains no rows: {input_path}")
    return fieldnames, rows


def parse_lambda_values(row: dict[str, str]) -> tuple[float, ...]:
    """Read and validate a five-dimensional lambda vector from one row."""
    try:
        values = tuple(float(row[column]) for column in LAMBDA_COLUMNS)
    except (TypeError, ValueError) as error:
        raise ValueError("All lambda columns must contain numeric values.") from error

    if any(value < 0 for value in values):
        raise ValueError("Lambda values must be non-negative.")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(
            f"Lambda values must sum to 1.0, received {sum(values):.8f}."
        )
    return values


def score_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Add the five heuristic proxy scores to every generated response."""
    scored_rows: list[dict[str, object]] = []
    for row in rows:
        lambda_values = parse_lambda_values(row)
        response = row.get("generated_response") or ""
        scored_rows.append(
            {
                **row,
                **dict(zip(LAMBDA_COLUMNS, lambda_values)),
                **score_response(response),
            }
        )
    return scored_rows


def summarize_merges(
    scored_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate proxy scores by merge name and full lambda vector."""
    grouped: dict[
        tuple[str, tuple[float, ...]],
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in scored_rows:
        lambda_values = tuple(float(row[column]) for column in LAMBDA_COLUMNS)
        grouped[(str(row["merge_name"]), lambda_values)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (merge_name, lambda_values), rows_for_merge in sorted(grouped.items()):
        num_responses = len(rows_for_merge)
        mean_proxies = {
            column: sum(float(row[column]) for row in rows_for_merge)
            / num_responses
            for column in PROXY_COLUMNS
        }

        summary: dict[str, object] = {
            "merge_name": merge_name,
            **dict(zip(LAMBDA_COLUMNS, lambda_values)),
            **{
                f"mean_{column}": round(value, 6)
                for column, value in mean_proxies.items()
            },
            "mean_response_length": round(
                sum(int(row["response_length"]) for row in rows_for_merge)
                / num_responses,
                3,
            ),
            "num_responses": num_responses,
        }

        attribute_means = tuple(mean_proxies[column] for column in PROXY_COLUMNS)
        for utility_name, preference in PREFERENCES.items():
            utility = sum(
                weight * score
                for weight, score in zip(preference, attribute_means)
            )
            summary[utility_name] = round(utility, 6)

        summary_rows.append(summary)

    return summary_rows


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write a small UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_best_merges(summary_rows: list[dict[str, object]]) -> None:
    """Print the best tested fixed merge under each placeholder utility."""
    print("\nBest tested merges under heuristic proxy utilities:")
    for utility_name, preference in PREFERENCES.items():
        best = max(summary_rows, key=lambda row: float(row[utility_name]))
        preference_text = ", ".join(f"{value:.2f}" for value in preference)
        print(
            f"  {utility_name} p=[{preference_text}]: "
            f"{best['merge_name']} "
            f"(utility={float(best[utility_name]):.4f})"
        )


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Score fixed HelpSteer2 adapter merges with placeholder heuristics."
        )
    )
    parser.add_argument(
        "--input_path",
        "--input-path",
        dest="input_path",
        default="results/helpsteer2_adapter_merge_generations.csv",
    )
    parser.add_argument(
        "--scored_output_path",
        "--scored-output-path",
        dest="scored_output_path",
        default="results/helpsteer2_adapter_merge_scored_generations.csv",
    )
    parser.add_argument(
        "--summary_output_path",
        "--summary-output-path",
        dest="summary_output_path",
        default="results/helpsteer2_lambda_sweep_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Score generations, summarize fixed merges, and save both CSV files."""
    args = parse_args()
    input_path = resolve_project_path(args.input_path)
    scored_output_path = resolve_project_path(args.scored_output_path)
    summary_output_path = resolve_project_path(args.summary_output_path)

    input_fieldnames, rows = read_generations(input_path)
    scored_rows = score_rows(rows)
    summary_rows = summarize_merges(scored_rows)

    scored_fieldnames = input_fieldnames + [
        *PROXY_COLUMNS,
        "response_length",
        "empty_response",
    ]
    summary_fieldnames = [
        "merge_name",
        *LAMBDA_COLUMNS,
        *(f"mean_{column}" for column in PROXY_COLUMNS),
        "mean_response_length",
        "num_responses",
        *PREFERENCES.keys(),
    ]

    write_csv(scored_output_path, scored_rows, scored_fieldnames)
    write_csv(summary_output_path, summary_rows, summary_fieldnames)

    print(
        "These scores are lightweight placeholder heuristics, not reward-model "
        "scores or human HelpSteer2 labels."
    )
    print(f"Scored {len(scored_rows)} generated responses.")
    print(f"Saved response scores to {scored_output_path}")
    print(f"Saved merge summary to {summary_output_path}")
    print_best_merges(summary_rows)


if __name__ == "__main__":
    main()
