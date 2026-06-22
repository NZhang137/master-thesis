"""Score and summarize a fixed lambda sweep with lightweight heuristics.

The scores in this script are deterministic prototype proxies, not learned
reward-model scores and not a final RLHF evaluation. This step evaluates only
the existing fixed merge grid; it does not compute R or lambda = f(p, R).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring_utils import (
    count_phrase_hits,
    count_words,
    harmlessness_proxy,
    helpfulness_proxy,
    reasonable_length_score,
    score_response,
)


REQUIRED_COLUMNS = {
    "lambda_helpful",
    "lambda_harmless",
    "prompt",
    "generated_response",
}

PREFERENCES = {
    "utility_p_50_50": (0.5, 0.5),
    "utility_p_80_20": (0.8, 0.2),
    "utility_p_20_80": (0.2, 0.8),
}


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_generations(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the merge-generation CSV."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}. "
            "Run scripts/evaluate_adapter_merges.py first."
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
        raise ValueError(f"Input CSV contains no generated responses: {input_path}")
    return fieldnames, rows


def score_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Add heuristic proxy scores to each generated response."""
    scored_rows = []

    for row in rows:
        response = row.get("generated_response") or ""
        scores = score_response(response)
        try:
            lambda_helpful = float(row["lambda_helpful"])
            lambda_harmless = float(row["lambda_harmless"])
        except (TypeError, ValueError) as error:
            raise ValueError("Lambda columns must contain numeric values.") from error

        scored_rows.append(
            {
                **row,
                "lambda_helpful": lambda_helpful,
                "lambda_harmless": lambda_harmless,
                "helpfulness_proxy": scores["helpfulness_proxy"],
                "harmlessness_proxy": scores["harmlessness_proxy"],
                "length": scores["response_length"],
                "empty_response": scores["empty_response"],
            }
        )

    return scored_rows


def summarize_by_lambda(
    scored_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate response-level proxies and utilities by lambda pair."""
    grouped: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        key = (float(row["lambda_helpful"]), float(row["lambda_harmless"]))
        grouped[key].append(row)

    summary_rows = []
    for (lambda_helpful, lambda_harmless), rows in sorted(
        grouped.items(),
        reverse=True,
    ):
        num_responses = len(rows)
        mean_helpfulness = sum(
            float(row["helpfulness_proxy"]) for row in rows
        ) / num_responses
        mean_harmlessness = sum(
            float(row["harmlessness_proxy"]) for row in rows
        ) / num_responses
        mean_length = sum(int(row["length"]) for row in rows) / num_responses

        summary = {
            "lambda_helpful": lambda_helpful,
            "lambda_harmless": lambda_harmless,
            "mean_helpfulness_proxy": round(mean_helpfulness, 6),
            "mean_harmlessness_proxy": round(mean_harmlessness, 6),
            "mean_length": round(mean_length, 3),
            "num_responses": num_responses,
        }
        for column, (p_helpful, p_harmless) in PREFERENCES.items():
            utility = (
                p_helpful * mean_helpfulness
                + p_harmless * mean_harmlessness
            )
            summary[column] = round(utility, 6)

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


def print_best_lambdas(summary_rows: list[dict[str, object]]) -> None:
    """Print the best fixed lambda pair for each example preference."""
    print("\nBest fixed lambda pairs under prototype heuristic utilities:")
    labels = {
        "utility_p_50_50": "[0.5, 0.5]",
        "utility_p_80_20": "[0.8, 0.2]",
        "utility_p_20_80": "[0.2, 0.8]",
    }

    for utility_column, preference_label in labels.items():
        best = max(summary_rows, key=lambda row: float(row[utility_column]))
        print(
            f"  p = {preference_label}: "
            f"lambda = [{best['lambda_helpful']}, {best['lambda_harmless']}], "
            f"utility = {float(best[utility_column]):.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Heuristically score the fixed LoRA lambda sweep."
    )
    parser.add_argument(
        "--input_path",
        "--input-path",
        dest="input_path",
        default="results/adapter_merge_generations.csv",
    )
    parser.add_argument(
        "--scored_output_path",
        "--scored-output-path",
        dest="scored_output_path",
        default="results/adapter_merge_scored_generations.csv",
    )
    parser.add_argument(
        "--summary_output_path",
        "--summary-output-path",
        dest="summary_output_path",
        default="results/lambda_sweep_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Score generated responses, aggregate by lambda, and save both tables."""
    args = parse_args()
    input_path = resolve_project_path(args.input_path)
    scored_output_path = resolve_project_path(args.scored_output_path)
    summary_output_path = resolve_project_path(args.summary_output_path)

    input_fieldnames, rows = read_generations(input_path)
    scored_rows = score_rows(rows)
    summary_rows = summarize_by_lambda(scored_rows)

    scored_fieldnames = input_fieldnames + [
        "helpfulness_proxy",
        "harmlessness_proxy",
        "length",
        "empty_response",
    ]
    summary_fieldnames = [
        "lambda_helpful",
        "lambda_harmless",
        "mean_helpfulness_proxy",
        "mean_harmlessness_proxy",
        "mean_length",
        "num_responses",
        "utility_p_50_50",
        "utility_p_80_20",
        "utility_p_20_80",
    ]

    write_csv(scored_output_path, scored_rows, scored_fieldnames)
    write_csv(summary_output_path, summary_rows, summary_fieldnames)

    print(f"Scored {len(scored_rows)} generated responses.")
    print(f"Saved response scores to {scored_output_path}")
    print(f"Saved lambda summary to {summary_output_path}")
    print_best_lambdas(summary_rows)


if __name__ == "__main__":
    main()
