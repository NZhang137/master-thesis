"""Evaluate HelpSteer2 uniform, direct-preference, M1, and C1 merges.

The response scores are lightweight deterministic proxy heuristics. They are
not reward-model scores or human HelpSteer2 labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_methods import (
    l1_distance,
    l2_distance,
    validate_preference_vector,
)
from src.coefficient_utils import load_labeled_relationship_matrix
from src.helpsteer2_scoring_utils import (
    OBJECTIVES,
    PROXY_COLUMNS,
    score_response,
)


P_COLUMNS = tuple(f"p_{objective}" for objective in OBJECTIVES)
LAMBDA_COLUMNS = tuple(f"lambda_{objective}" for objective in OBJECTIVES)

TEST_PROMPTS = (
    "Human: What is a good way to stay motivated?\n\nAssistant:",
    "Human: How can I improve my study habits?\n\nAssistant:",
    "Human: Explain why sleep is important.\n\nAssistant:",
    "Human: How should I handle a disagreement with a friend?\n\nAssistant:",
)

GENERATION_COLUMNS = [
    "preference_name",
    "method",
    "hyperparameter_name",
    "hyperparameter_value",
    *LAMBDA_COLUMNS,
    "prompt",
    "generated_response",
]

SCORED_COLUMNS = [
    *GENERATION_COLUMNS,
    *PROXY_COLUMNS,
    "response_length",
    "empty_response",
]

COMPARISON_COLUMNS = [
    "preference_name",
    "method",
    "hyperparameter_name",
    "hyperparameter_value",
    *P_COLUMNS,
    *LAMBDA_COLUMNS,
    *(f"mean_{column}" for column in PROXY_COLUMNS),
    "mean_length",
    "num_responses",
    "utility_for_preference",
    "l1_distance_to_p",
    "l2_distance_to_p",
    "min_relationship_score",
    "best_fixed_sweep_utility",
    "gap_to_best_fixed_sweep",
]

REQUIRED_COEFFICIENT_COLUMNS = {
    "preference_name",
    "method",
    "hyperparameter_name",
    "hyperparameter_value",
    *P_COLUMNS,
    *LAMBDA_COLUMNS,
    "min_relationship_score",
    "l1_distance_to_p",
    "l2_distance_to_p",
}

FIXED_SWEEP_UTILITY_COLUMNS = {
    "balanced": "utility_balanced",
    "quality_focused": "utility_quality_focused",
    "detailed_answer": "utility_detailed_answer",
    "helpfulness_focused": "utility_helpfulness_focused",
}

ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


@dataclass(frozen=True)
class Candidate:
    """One preference, method, hyperparameter, and lambda setting."""

    preference_name: str
    method: str
    hyperparameter_name: str
    hyperparameter_value: float | None
    preference: np.ndarray
    lambdas: np.ndarray
    l1_distance_to_p: float
    l2_distance_to_p: float
    min_relationship_score: float | None


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_numeric_vector(
    row: dict[str, str],
    columns: tuple[str, ...],
    label: str,
) -> np.ndarray:
    """Read a finite non-negative simplex vector from CSV columns."""
    try:
        values = np.array([float(row[column]) for column in columns])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} columns must contain numeric values.") from error
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{label} values must be finite and non-negative.")
    if abs(float(values.sum()) - 1.0) > 1e-6:
        raise ValueError(f"{label} values must sum to 1.0.")
    return values / float(values.sum())


def parse_optional_float(value: str, label: str) -> float | None:
    """Parse an optional finite numeric CSV field."""
    if value is None or not str(value).strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric when provided.") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be finite when provided.")
    return parsed


def read_coefficient_candidates(
    coefficient_path: Path,
    relationships: np.ndarray | None,
) -> list[Candidate]:
    """Load coefficient rows and add one uniform baseline per preference."""
    if not coefficient_path.is_file():
        raise FileNotFoundError(
            f"Coefficient file not found: {coefficient_path}. "
            "Run scripts/compute_helpsteer2_m1_c1_coefficients.py first."
        )

    with coefficient_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COEFFICIENT_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(
                "Coefficient CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        coefficient_rows = list(reader)

    if not coefficient_rows:
        raise ValueError(f"Coefficient CSV contains no rows: {coefficient_path}")

    preferences: dict[str, np.ndarray] = {}
    parsed_rows: list[Candidate] = []
    for row in coefficient_rows:
        preference_name = row["preference_name"].strip()
        method = row["method"].strip()
        if not preference_name:
            raise ValueError("preference_name must be non-empty.")
        if method not in {"direct_preference", "M1", "C1"}:
            raise ValueError(f"Unsupported coefficient method: {method!r}")

        preference = validate_preference_vector(
            parse_numeric_vector(row, P_COLUMNS, "Preference")
        )
        previous = preferences.setdefault(preference_name, preference)
        if not np.allclose(previous, preference, atol=1e-9):
            raise ValueError(
                f"Preference rows disagree for {preference_name!r}."
            )

        lambdas = parse_numeric_vector(row, LAMBDA_COLUMNS, "Lambda")
        hyperparameter_name = row["hyperparameter_name"].strip()
        hyperparameter_value = parse_optional_float(
            row["hyperparameter_value"],
            "hyperparameter_value",
        )
        if method == "direct_preference":
            hyperparameter_name = ""
            hyperparameter_value = None
        elif not hyperparameter_name or hyperparameter_value is None:
            raise ValueError(
                f"{method} rows require a hyperparameter name and value."
            )

        parsed_rows.append(
            Candidate(
                preference_name=preference_name,
                method=method,
                hyperparameter_name=hyperparameter_name,
                hyperparameter_value=hyperparameter_value,
                preference=preference,
                lambdas=lambdas,
                l1_distance_to_p=float(
                    parse_optional_float(
                        row["l1_distance_to_p"],
                        "l1_distance_to_p",
                    )
                ),
                l2_distance_to_p=float(
                    parse_optional_float(
                        row["l2_distance_to_p"],
                        "l2_distance_to_p",
                    )
                ),
                min_relationship_score=parse_optional_float(
                    row["min_relationship_score"],
                    "min_relationship_score",
                ),
            )
        )

    uniform = np.full(len(OBJECTIVES), 1.0 / len(OBJECTIVES))
    uniform_candidates = []
    for preference_name, preference in preferences.items():
        min_score = (
            float(np.min(relationships @ uniform))
            if relationships is not None
            else None
        )
        uniform_candidates.append(
            Candidate(
                preference_name=preference_name,
                method="uniform",
                hyperparameter_name="",
                hyperparameter_value=None,
                preference=preference,
                lambdas=uniform.copy(),
                l1_distance_to_p=l1_distance(uniform, preference),
                l2_distance_to_p=l2_distance(uniform, preference),
                min_relationship_score=min_score,
            )
        )

    return uniform_candidates + parsed_rows


def load_optional_relationship_matrix(matrix_path: Path) -> np.ndarray | None:
    """Load R for uniform relationship scores when the file is available."""
    if not matrix_path.is_file():
        print(
            f"Relationship matrix not found at {matrix_path}. "
            "Uniform min_relationship_score will be left empty."
        )
        return None
    return load_labeled_relationship_matrix(matrix_path, OBJECTIVES)


def validate_adapter_paths(adapter_paths: list[Path]) -> None:
    """Validate all local PEFT adapter folders before model loading."""
    for objective, path in zip(OBJECTIVES, adapter_paths):
        if not path.is_dir():
            raise FileNotFoundError(
                f"{objective} adapter directory not found: {path}"
            )
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"{objective} adapter is missing adapter_config.json: {path}"
            )
        if not any((path / name).is_file() for name in ADAPTER_WEIGHT_FILENAMES):
            expected = " or ".join(ADAPTER_WEIGHT_FILENAMES)
            raise FileNotFoundError(
                f"{objective} adapter is missing {expected}: {path}"
            )


def lambda_cache_key(lambdas: np.ndarray) -> tuple[float, ...]:
    """Return a stable cache key for a saved ten-decimal lambda vector."""
    return tuple(float(round(value, 10)) for value in lambdas)


def generate_responses(
    candidates: list[Candidate],
    model_name: str,
    adapter_paths: list[Path],
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, object]]:
    """Generate prompt responses for all settings, caching duplicate lambdas."""
    import torch

    from src.evaluation_utils import generate_response
    from src.merge_utils import load_model_with_weighted_lora_adapters_multi

    validate_adapter_paths(adapter_paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    response_cache: dict[tuple[float, ...], list[str]] = {}
    generation_rows: list[dict[str, object]] = []

    for index, candidate in enumerate(candidates, start=1):
        cache_key = lambda_cache_key(candidate.lambdas)
        if cache_key not in response_cache:
            print(
                f"\nGenerating unique merge {len(response_cache) + 1}: "
                f"lambda={list(cache_key)}"
            )
            model, tokenizer = load_model_with_weighted_lora_adapters_multi(
                model_name=model_name,
                adapter_paths=adapter_paths,
                adapter_names=OBJECTIVES,
                weights=candidate.lambdas,
                device=device,
            )
            responses = []
            for prompt_index, prompt in enumerate(TEST_PROMPTS):
                prompt_seed = seed + prompt_index
                torch.manual_seed(prompt_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(prompt_seed)
                responses.append(
                    generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                )
            response_cache[cache_key] = responses

            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(
                f"Reusing cached responses for setting "
                f"{index}/{len(candidates)}: {candidate.method}"
            )

        for prompt, response in zip(TEST_PROMPTS, response_cache[cache_key]):
            generation_rows.append(
                {
                    "preference_name": candidate.preference_name,
                    "method": candidate.method,
                    "hyperparameter_name": candidate.hyperparameter_name,
                    "hyperparameter_value": (
                        candidate.hyperparameter_value
                        if candidate.hyperparameter_value is not None
                        else ""
                    ),
                    **dict(zip(LAMBDA_COLUMNS, candidate.lambdas)),
                    "prompt": prompt,
                    "generated_response": response,
                }
            )

    print(
        f"\nEvaluated {len(candidates)} settings using "
        f"{len(response_cache)} unique lambda vectors."
    )
    return generation_rows


def score_generation_rows(
    generation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add the shared HelpSteer2 placeholder proxy fields to each response."""
    return [
        {
            **row,
            **score_response(str(row.get("generated_response") or "")),
        }
        for row in generation_rows
    ]


def candidate_key(
    preference_name: str,
    method: str,
    hyperparameter_name: str,
    hyperparameter_value: object,
) -> tuple[str, str, str, float | None]:
    """Build a stable key for one method and hyperparameter setting."""
    value = (
        None
        if hyperparameter_value is None or str(hyperparameter_value) == ""
        else float(hyperparameter_value)
    )
    return preference_name, method, hyperparameter_name, value


def read_lambda_best_utilities(summary_path: Path) -> dict[str, float]:
    """Read the best fixed-sweep utility for each example preference."""
    if not summary_path.is_file():
        print(
            f"Fixed-sweep summary not found at {summary_path}. "
            "lambda_best comparison columns will be left empty."
        )
        return {}

    with summary_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        available = {
            preference: column
            for preference, column in FIXED_SWEEP_UTILITY_COLUMNS.items()
            if column in fieldnames
        }
        rows = list(reader)

    if not rows:
        raise ValueError(f"Fixed-sweep summary contains no rows: {summary_path}")

    lambda_best_utilities = {}
    for preference_name, column in available.items():
        try:
            values = [float(row[column]) for row in rows]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Fixed-sweep column {column!r} must be numeric."
            ) from error
        if not all(np.isfinite(value) for value in values):
            raise ValueError(
                f"Fixed-sweep column {column!r} must contain finite values."
            )
        lambda_best_utilities[preference_name] = max(values)

    return lambda_best_utilities


def aggregate_comparison(
    candidates: list[Candidate],
    scored_rows: list[dict[str, object]],
    lambda_best_utilities: dict[str, float],
) -> list[dict[str, object]]:
    """Aggregate proxy means and preference utility for every setting."""
    grouped: dict[
        tuple[str, str, str, float | None],
        list[dict[str, object]],
    ] = {}
    for row in scored_rows:
        key = candidate_key(
            str(row["preference_name"]),
            str(row["method"]),
            str(row["hyperparameter_name"]),
            row["hyperparameter_value"],
        )
        grouped.setdefault(key, []).append(row)

    comparison_rows = []
    for candidate in candidates:
        key = candidate_key(
            candidate.preference_name,
            candidate.method,
            candidate.hyperparameter_name,
            candidate.hyperparameter_value,
        )
        rows = grouped.get(key, [])
        if len(rows) != len(TEST_PROMPTS):
            raise ValueError(
                f"Expected {len(TEST_PROMPTS)} responses for {key}, "
                f"received {len(rows)}."
            )

        mean_proxies = np.array(
            [
                sum(float(row[column]) for row in rows) / len(rows)
                for column in PROXY_COLUMNS
            ],
            dtype=np.float64,
        )
        utility = float(candidate.preference @ mean_proxies)
        lambda_best = lambda_best_utilities.get(candidate.preference_name)

        comparison_rows.append(
            {
                "preference_name": candidate.preference_name,
                "method": candidate.method,
                "hyperparameter_name": candidate.hyperparameter_name,
                "hyperparameter_value": (
                    candidate.hyperparameter_value
                    if candidate.hyperparameter_value is not None
                    else ""
                ),
                **dict(zip(P_COLUMNS, candidate.preference)),
                **dict(zip(LAMBDA_COLUMNS, candidate.lambdas)),
                **{
                    f"mean_{column}": value
                    for column, value in zip(PROXY_COLUMNS, mean_proxies)
                },
                "mean_length": (
                    sum(int(row["response_length"]) for row in rows) / len(rows)
                ),
                "num_responses": len(rows),
                "utility_for_preference": utility,
                "l1_distance_to_p": candidate.l1_distance_to_p,
                "l2_distance_to_p": candidate.l2_distance_to_p,
                "min_relationship_score": (
                    candidate.min_relationship_score
                    if candidate.min_relationship_score is not None
                    else ""
                ),
                "best_fixed_sweep_utility": (
                    lambda_best if lambda_best is not None else ""
                ),
                "gap_to_best_fixed_sweep": (
                    utility - lambda_best if lambda_best is not None else ""
                ),
            }
        )

    return comparison_rows


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


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    candidate_count: int,
    generation_count: int,
    fixed_sweep_available: bool,
) -> None:
    """Write compact provenance and evaluation metadata."""
    metadata = {
        "input_coefficient_file": args.coefficient_path,
        "relationship_matrix_file": args.relationship_matrix_path,
        "fixed_sweep_summary_file": args.lambda_sweep_summary_path,
        "output_files": {
            "generations": args.output_generations_path,
            "scored_generations": args.output_scored_path,
            "comparison": args.output_comparison_path,
            "metadata": args.output_metadata_path,
        },
        "objectives": list(OBJECTIVES),
        "methods_evaluated": ["uniform", "direct_preference", "M1", "C1"],
        "prompt_count": len(TEST_PROMPTS),
        "candidate_count": candidate_count,
        "generation_count": generation_count,
        "fixed_sweep_lambda_best_comparison_available": fixed_sweep_available,
        "generation_settings": {
            "model_name": args.model_name,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "note": (
            "Proxy scores are lightweight prototype scores, not HelpSteer2 "
            "human labels or reward-model scores."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def print_summary(comparison_rows: list[dict[str, object]]) -> None:
    """Print the best evaluated method setting for each preference."""
    print("\nBest evaluated settings under lightweight proxy utility:")
    preference_names = list(
        dict.fromkeys(str(row["preference_name"]) for row in comparison_rows)
    )
    for preference_name in preference_names:
        rows = [
            row
            for row in comparison_rows
            if row["preference_name"] == preference_name
        ]
        best = max(rows, key=lambda row: float(row["utility_for_preference"]))
        hyperparameter = ""
        if best["hyperparameter_name"]:
            hyperparameter = (
                f", {best['hyperparameter_name']}="
                f"{float(best['hyperparameter_value']):g}"
            )
        print(
            f"  {preference_name}: {best['method']}{hyperparameter}, "
            f"utility={float(best['utility_for_preference']):.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse model, adapter, input, output, and generation settings."""
    parser = argparse.ArgumentParser(
        description="Evaluate HelpSteer2 uniform, direct, M1, and C1 merges."
    )
    parser.add_argument(
        "--coefficient_path",
        "--coefficient-path",
        dest="coefficient_path",
        default="results/helpsteer2_m1_c1_coefficients.csv",
    )
    parser.add_argument(
        "--relationship_matrix_path",
        "--relationship-matrix-path",
        dest="relationship_matrix_path",
        default="results/helpsteer2_relationship_matrix.csv",
    )
    parser.add_argument(
        "--lambda_sweep_summary_path",
        "--lambda-sweep-summary-path",
        dest="lambda_sweep_summary_path",
        default="results/helpsteer2_lambda_sweep_summary.csv",
    )
    parser.add_argument(
        "--output_generations_path",
        "--output-generations-path",
        dest="output_generations_path",
        default="results/helpsteer2_m1_c1_merge_generations.csv",
    )
    parser.add_argument(
        "--output_scored_path",
        "--output-scored-path",
        dest="output_scored_path",
        default="results/helpsteer2_m1_c1_scored_generations.csv",
    )
    parser.add_argument(
        "--output_comparison_path",
        "--output-comparison-path",
        dest="output_comparison_path",
        default="results/helpsteer2_m1_c1_comparison.csv",
    )
    parser.add_argument(
        "--output_metadata_path",
        "--output-metadata-path",
        dest="output_metadata_path",
        default="results/helpsteer2_m1_c1_comparison_metadata.json",
    )
    parser.add_argument("--model_name", "--model-name", default="gpt2")
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        type=int,
        default=100,
    )
    parser.add_argument("--seed", type=int, default=42)
    for objective in OBJECTIVES:
        parser.add_argument(
            f"--{objective}_adapter_path",
            f"--{objective}-adapter-path",
            dest=f"{objective}_adapter_path",
            default=f"adapters/helpsteer2-gpt2-{objective}-adapter",
        )
    return parser.parse_args()


def main() -> None:
    """Generate, score, aggregate, and save the M1/C1 comparison."""
    args = parse_args()
    if not args.model_name.strip():
        raise ValueError("model_name must be non-empty.")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    coefficient_path = resolve_project_path(args.coefficient_path)
    relationship_matrix_path = resolve_project_path(
        args.relationship_matrix_path
    )
    lambda_sweep_summary_path = resolve_project_path(
        args.lambda_sweep_summary_path
    )
    output_generations_path = resolve_project_path(
        args.output_generations_path
    )
    output_scored_path = resolve_project_path(args.output_scored_path)
    output_comparison_path = resolve_project_path(
        args.output_comparison_path
    )
    output_metadata_path = resolve_project_path(args.output_metadata_path)
    adapter_paths = [
        resolve_project_path(getattr(args, f"{objective}_adapter_path"))
        for objective in OBJECTIVES
    ]

    relationships = load_optional_relationship_matrix(
        relationship_matrix_path
    )
    candidates = read_coefficient_candidates(
        coefficient_path,
        relationships,
    )
    generation_rows = generate_responses(
        candidates,
        model_name=args.model_name,
        adapter_paths=adapter_paths,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    scored_rows = score_generation_rows(generation_rows)
    lambda_best_utilities = read_lambda_best_utilities(
        lambda_sweep_summary_path
    )
    comparison_rows = aggregate_comparison(
        candidates,
        scored_rows,
        lambda_best_utilities,
    )

    write_csv(
        output_generations_path,
        generation_rows,
        GENERATION_COLUMNS,
    )
    write_csv(output_scored_path, scored_rows, SCORED_COLUMNS)
    write_csv(
        output_comparison_path,
        comparison_rows,
        COMPARISON_COLUMNS,
    )
    write_metadata(
        output_metadata_path,
        args,
        candidate_count=len(candidates),
        generation_count=len(generation_rows),
        fixed_sweep_available=bool(lambda_best_utilities),
    )

    print(
        "\nProxy scores are lightweight prototype scores, not HelpSteer2 "
        "human labels or reward-model scores."
    )
    print(f"Saved {len(generation_rows)} generations to {output_generations_path}")
    print(f"Saved scored generations to {output_scored_path}")
    print(f"Saved comparison table to {output_comparison_path}")
    print(f"Saved metadata to {output_metadata_path}")
    print_summary(comparison_rows)


if __name__ == "__main__":
    main()
