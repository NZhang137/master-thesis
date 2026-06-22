"""Compare M1 with simple coefficient-selection baselines.

Responses are evaluated with deterministic heuristic proxy scores. These
scores are placeholders for prototype comparisons, not final reward-model
evaluation. The script performs no coefficient-space optimization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coefficient_utils import (
    l1_distance,
    l2_distance,
    load_labeled_relationship_matrix,
    normalize_simplex,
    relationship_softmax_mapping,
)
from src.scoring_utils import score_response


MODEL_NAME = "gpt2"
OBJECTIVE_NAMES = ("helpful", "harmless")
PREFERENCES = (
    ("p_50_50", np.array([0.5, 0.5], dtype=np.float64)),
    ("p_80_20", np.array([0.8, 0.2], dtype=np.float64)),
    ("p_20_80", np.array([0.2, 0.8], dtype=np.float64)),
)
GRID_UTILITY_COLUMNS = {
    "p_50_50": "utility_p_50_50",
    "p_80_20": "utility_p_80_20",
    "p_20_80": "utility_p_20_80",
}

GENERATION_FIELDNAMES = [
    "preference_name",
    "method",
    "tau",
    "p_helpful",
    "p_harmless",
    "lambda_helpful",
    "lambda_harmless",
    "prompt",
    "generated_response",
    "helpfulness_proxy",
    "harmlessness_proxy",
    "response_length",
    "empty_response",
]

COMPARISON_FIELDNAMES = [
    "preference_name",
    "method",
    "tau",
    "p_helpful",
    "p_harmless",
    "lambda_helpful",
    "lambda_harmless",
    "mean_helpfulness_proxy",
    "mean_harmlessness_proxy",
    "mean_length",
    "num_responses",
    "utility",
    "l1_distance_to_p",
    "l2_distance_to_p",
    "best_grid_utility_if_available",
    "gap_to_best_grid_if_available",
]


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_candidates(R: np.ndarray, tau: float) -> list[dict[str, object]]:
    """Build uniform, direct-preference, and M1 candidates."""
    candidates = []

    for preference_name, preference in PREFERENCES:
        p = normalize_simplex(preference)
        method_lambdas = (
            ("uniform", "", np.array([0.5, 0.5], dtype=np.float64)),
            ("direct_preference", "", p.copy()),
            ("M1", tau, relationship_softmax_mapping(p, R, tau=tau)),
        )

        for method, method_tau, lambdas in method_lambdas:
            candidates.append(
                {
                    "preference_name": preference_name,
                    "method": method,
                    "tau": method_tau,
                    "p": p,
                    "lambdas": normalize_simplex(lambdas),
                }
            )

    return candidates


def read_best_grid_utilities(summary_path: Path) -> dict[str, float]:
    """Read the best fixed-grid utility for each example preference."""
    if not summary_path.is_file():
        print(
            f"Fixed-grid summary not found at {summary_path}. "
            "Grid comparison columns will be left empty."
        )
        return {}

    with summary_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        missing = set(GRID_UTILITY_COLUMNS.values()).difference(fieldnames)
        if missing:
            raise ValueError(
                "Lambda-sweep summary is missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Lambda-sweep summary contains no rows: {summary_path}")

    best_utilities = {}
    for preference_name, utility_column in GRID_UTILITY_COLUMNS.items():
        try:
            values = np.array(
                [float(row[utility_column]) for row in rows],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Column '{utility_column}' must contain numeric values."
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Column '{utility_column}' must contain finite values."
            )
        best_utilities[preference_name] = float(np.max(values))

    return best_utilities


def generate_and_score(
    helpful_adapter_path: Path,
    harmless_adapter_path: Path,
    lambdas: np.ndarray,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    """Generate and heuristically score the shared prompt set for one merge."""
    import torch

    from src.evaluation_utils import PROTOTYPE_TEST_PROMPTS, generate_response
    from src.merge_utils import load_model_with_weighted_lora_adapters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tokenizer = None

    try:
        model, tokenizer = load_model_with_weighted_lora_adapters(
            model_name=MODEL_NAME,
            helpful_adapter_path=helpful_adapter_path,
            harmless_adapter_path=harmless_adapter_path,
            weights=lambdas,
            device=device,
        )

        rows = []
        for prompt_index, prompt in enumerate(PROTOTYPE_TEST_PROMPTS):
            seed = 42 + prompt_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            rows.append(
                {
                    "prompt": prompt,
                    "generated_response": response,
                    **score_response(response),
                }
            )
        return rows
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def lambda_cache_key(lambdas: np.ndarray) -> tuple[float, float]:
    """Create a stable cache key for one normalized two-objective lambda."""
    return tuple(float(value) for value in np.round(lambdas, decimals=12))


def evaluate_candidates(
    candidates: list[dict[str, object]],
    helpful_adapter_path: Path,
    harmless_adapter_path: Path,
    max_new_tokens: int,
    best_grid_utilities: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Generate responses and aggregate utilities for all candidates."""
    generation_cache: dict[tuple[float, float], list[dict[str, object]]] = {}
    generation_rows = []
    comparison_rows = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        p = np.asarray(candidate["p"], dtype=np.float64)
        lambdas = np.asarray(candidate["lambdas"], dtype=np.float64)
        key = lambda_cache_key(lambdas)

        print(
            f"\nCandidate {candidate_index}/{len(candidates)}: "
            f"{candidate['preference_name']} / {candidate['method']} / "
            f"lambda=[{lambdas[0]:.4f}, {lambdas[1]:.4f}]"
        )

        if key not in generation_cache:
            generation_cache[key] = generate_and_score(
                helpful_adapter_path=helpful_adapter_path,
                harmless_adapter_path=harmless_adapter_path,
                lambdas=lambdas,
                max_new_tokens=max_new_tokens,
            )
        else:
            print("Reusing generations for an identical lambda pair.")

        scored_responses = generation_cache[key]
        for response_row in scored_responses:
            generation_rows.append(
                {
                    "preference_name": candidate["preference_name"],
                    "method": candidate["method"],
                    "tau": candidate["tau"],
                    "p_helpful": p[0],
                    "p_harmless": p[1],
                    "lambda_helpful": lambdas[0],
                    "lambda_harmless": lambdas[1],
                    **response_row,
                }
            )

        num_responses = len(scored_responses)
        mean_helpfulness = sum(
            float(row["helpfulness_proxy"]) for row in scored_responses
        ) / num_responses
        mean_harmlessness = sum(
            float(row["harmlessness_proxy"]) for row in scored_responses
        ) / num_responses
        mean_length = sum(
            int(row["response_length"]) for row in scored_responses
        ) / num_responses
        utility = p[0] * mean_helpfulness + p[1] * mean_harmlessness

        best_grid_utility = best_grid_utilities.get(
            str(candidate["preference_name"])
        )
        gap_to_grid = (
            utility - best_grid_utility
            if best_grid_utility is not None
            else ""
        )

        comparison_rows.append(
            {
                "preference_name": candidate["preference_name"],
                "method": candidate["method"],
                "tau": candidate["tau"],
                "p_helpful": p[0],
                "p_harmless": p[1],
                "lambda_helpful": lambdas[0],
                "lambda_harmless": lambdas[1],
                "mean_helpfulness_proxy": mean_helpfulness,
                "mean_harmlessness_proxy": mean_harmlessness,
                "mean_length": mean_length,
                "num_responses": num_responses,
                "utility": utility,
                "l1_distance_to_p": l1_distance(lambdas, p),
                "l2_distance_to_p": l2_distance(lambdas, p),
                "best_grid_utility_if_available": (
                    best_grid_utility
                    if best_grid_utility is not None
                    else ""
                ),
                # Positive values mean the candidate exceeded the grid benchmark.
                "gap_to_best_grid_if_available": gap_to_grid,
            }
        )

    return generation_rows, comparison_rows


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write a compact UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(comparison_rows: list[dict[str, object]]) -> None:
    """Print candidate utilities grouped by preference."""
    print("\nM1 baseline comparison:")
    for preference_name, _ in PREFERENCES:
        print(f"\n  {preference_name}:")
        preference_rows = [
            row
            for row in comparison_rows
            if row["preference_name"] == preference_name
        ]
        for row in preference_rows:
            gap = row["gap_to_best_grid_if_available"]
            gap_text = (
                f", grid gap={float(gap):+.4f}"
                if gap != ""
                else ", grid gap=n/a"
            )
            print(
                f"    {row['method']}: "
                f"lambda=[{float(row['lambda_helpful']):.4f}, "
                f"{float(row['lambda_harmless']):.4f}], "
                f"utility={float(row['utility']):.4f}{gap_text}"
            )


def parse_args() -> argparse.Namespace:
    """Parse input, adapter, output, and generation settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare M1 with uniform and direct-preference coefficients "
            "using heuristic prototype scores."
        )
    )
    parser.add_argument(
        "--relationship_matrix_path",
        "--relationship-matrix-path",
        dest="relationship_matrix_path",
        default="results/relationship_matrix.csv",
    )
    parser.add_argument(
        "--lambda_sweep_summary_path",
        "--lambda-sweep-summary-path",
        dest="lambda_sweep_summary_path",
        default="results/lambda_sweep_summary.csv",
    )
    parser.add_argument(
        "--helpful_adapter_path",
        "--helpful-adapter-path",
        dest="helpful_adapter_path",
        default="adapters/gpt2-helpful-adapter",
    )
    parser.add_argument(
        "--harmless_adapter_path",
        "--harmless-adapter-path",
        dest="harmless_adapter_path",
        default="adapters/gpt2-harmless-adapter",
    )
    parser.add_argument(
        "--output_generations_path",
        "--output-generations-path",
        dest="output_generations_path",
        default="results/m1_baseline_generations.csv",
    )
    parser.add_argument(
        "--output_comparison_path",
        "--output-comparison-path",
        dest="output_comparison_path",
        default="results/m1_baseline_comparison.csv",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=80,
    )
    return parser.parse_args()


def main() -> None:
    """Load inputs, evaluate all candidates, and save both result tables."""
    args = parse_args()
    if not np.isfinite(args.tau) or args.tau < 0:
        raise ValueError("tau must be a finite, non-negative value.")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    relationship_matrix_path = resolve_project_path(
        args.relationship_matrix_path
    )
    lambda_sweep_summary_path = resolve_project_path(
        args.lambda_sweep_summary_path
    )
    helpful_adapter_path = resolve_project_path(args.helpful_adapter_path)
    harmless_adapter_path = resolve_project_path(args.harmless_adapter_path)
    output_generations_path = resolve_project_path(
        args.output_generations_path
    )
    output_comparison_path = resolve_project_path(args.output_comparison_path)

    R = load_labeled_relationship_matrix(
        relationship_matrix_path,
        OBJECTIVE_NAMES,
    )
    best_grid_utilities = read_best_grid_utilities(
        lambda_sweep_summary_path
    )
    candidates = build_candidates(R, tau=args.tau)

    print(f"Relationship matrix: {relationship_matrix_path}")
    print(f"Helpful adapter: {helpful_adapter_path}")
    print(f"Harmless adapter: {harmless_adapter_path}")
    print(f"M1 tau: {args.tau}")
    print(
        "Heuristic proxy scores are used; this is not final reward-model "
        "evaluation."
    )

    generation_rows, comparison_rows = evaluate_candidates(
        candidates=candidates,
        helpful_adapter_path=helpful_adapter_path,
        harmless_adapter_path=harmless_adapter_path,
        max_new_tokens=args.max_new_tokens,
        best_grid_utilities=best_grid_utilities,
    )

    write_csv(
        output_generations_path,
        generation_rows,
        GENERATION_FIELDNAMES,
    )
    write_csv(
        output_comparison_path,
        comparison_rows,
        COMPARISON_FIELDNAMES,
    )

    print_summary(comparison_rows)
    print(f"\nSaved {len(generation_rows)} generations to {output_generations_path}")
    print(f"Saved {len(comparison_rows)} comparisons to {output_comparison_path}")


if __name__ == "__main__":
    main()
