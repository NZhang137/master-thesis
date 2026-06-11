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
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

LAMBDA_COLUMNS = tuple(f"lambda_{attribute}" for attribute in ATTRIBUTES)
PROXY_COLUMNS = tuple(f"{attribute}_proxy" for attribute in ATTRIBUTES)

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

HELPFUL_PHRASES = (
    "you can",
    "try",
    "consider",
    "recommend",
    "help",
    "practice",
    "support",
    "important",
)

EXPLANATION_PHRASES = (
    "because",
    "therefore",
    "for example",
    "this means",
    "the reason",
    "evidence",
    "depends on",
)

CALIBRATED_PHRASES = (
    "may",
    "might",
    "can",
    "often",
    "generally",
    "in some cases",
)

OVERCONFIDENT_PHRASES = (
    "always",
    "never",
    "guaranteed",
    "100 percent",
    "without exception",
)

COHERENCE_PHRASES = (
    "first",
    "second",
    "then",
    "however",
    "also",
    "finally",
    "because",
    "therefore",
)

WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
SENTENCE_END_PATTERN = re.compile(r"[.!?]+")


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value to a closed interval."""
    return min(max(value, lower), upper)


def normalize_text(text: str) -> str:
    """Return lowercase text with collapsed whitespace."""
    return " ".join(text.lower().split())


def words(text: str) -> list[str]:
    """Extract simple lowercase word-like tokens."""
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def count_phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    """Count how many distinct phrases occur in normalized text."""
    normalized = normalize_text(text)
    return sum(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None
        for phrase in phrases
    )


def reasonable_length_score(length: int) -> float:
    """Return a bounded score favoring responses between 20 and 120 words."""
    if length <= 0:
        return 0.0
    if length < 20:
        return length / 20
    if length <= 120:
        return 1.0
    return max(0.0, 1.0 - (length - 120) / 180)


def helpfulness_proxy(response: str) -> float:
    """Estimate helpful surface wording, not actual task success."""
    response_words = words(response)
    if not response_words:
        return 0.0

    phrase_score = min(count_phrase_hits(response, HELPFUL_PHRASES) / 3, 1.0)
    score = (
        0.20
        + 0.45 * reasonable_length_score(len(response_words))
        + 0.35 * phrase_score
    )
    return round(clamp(score), 6)


def correctness_proxy(response: str) -> float:
    """Estimate explanatory and calibrated wording, not factual correctness."""
    if not response.strip():
        return 0.0

    explanation_score = min(
        count_phrase_hits(response, EXPLANATION_PHRASES) / 3,
        1.0,
    )
    calibration_score = min(
        count_phrase_hits(response, CALIBRATED_PHRASES) / 3,
        1.0,
    )
    overconfidence_score = min(
        count_phrase_hits(response, OVERCONFIDENT_PHRASES) / 2,
        1.0,
    )
    score = (
        0.35
        + 0.35 * explanation_score
        + 0.20 * calibration_score
        + 0.10 * reasonable_length_score(len(words(response)))
        - 0.30 * overconfidence_score
    )
    return round(clamp(score), 6)


def adjacent_repetition_rate(tokens: list[str]) -> float:
    """Return the fraction of adjacent token pairs that repeat a token."""
    if len(tokens) < 2:
        return 0.0
    repeated = sum(a == b for a, b in zip(tokens, tokens[1:]))
    return repeated / (len(tokens) - 1)


def coherence_proxy(response: str) -> float:
    """Estimate basic organization and continuity from textual surface cues."""
    response_words = words(response)
    if not response_words:
        return 0.0

    sentence_score = min(len(SENTENCE_END_PATTERN.findall(response)) / 3, 1.0)
    connector_score = min(
        count_phrase_hits(response, COHERENCE_PHRASES) / 3,
        1.0,
    )
    repetition_penalty = min(adjacent_repetition_rate(response_words) * 5, 1.0)
    score = (
        0.30
        + 0.30 * sentence_score
        + 0.30 * connector_score
        + 0.10 * reasonable_length_score(len(response_words))
        - 0.35 * repetition_penalty
    )
    return round(clamp(score), 6)


def complexity_proxy(response: str) -> float:
    """Estimate lexical and sentence complexity without judging its quality."""
    response_words = words(response)
    if not response_words:
        return 0.0

    average_word_length = sum(map(len, response_words)) / len(response_words)
    long_word_ratio = sum(len(word) >= 8 for word in response_words) / len(
        response_words
    )
    sentence_count = max(len(SENTENCE_END_PATTERN.findall(response)), 1)
    average_sentence_length = len(response_words) / sentence_count

    word_length_score = clamp((average_word_length - 3.5) / 3.0)
    long_word_score = clamp(long_word_ratio / 0.25)
    sentence_length_score = clamp((average_sentence_length - 6) / 24)
    score = (
        0.15
        + 0.35 * word_length_score
        + 0.30 * long_word_score
        + 0.20 * sentence_length_score
    )
    return round(clamp(score), 6)


def verbosity_proxy(response: str) -> float:
    """Use normalized word count as a simple proxy for response verbosity."""
    response_length = len(words(response))
    return round(clamp(response_length / 120), 6)


def score_response(response: str) -> dict[str, float | int | bool]:
    """Compute all response-level placeholder proxy fields."""
    response_length = len(words(response))
    return {
        "helpfulness_proxy": helpfulness_proxy(response),
        "correctness_proxy": correctness_proxy(response),
        "coherence_proxy": coherence_proxy(response),
        "complexity_proxy": complexity_proxy(response),
        "verbosity_proxy": verbosity_proxy(response),
        "response_length": response_length,
        "empty_response": response_length == 0,
    }


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
