"""Evaluate all HelpSteer2 coefficient rows with fixed prompts.

This script evaluates coefficient vectors inside the fixed
Rewarded-Soups-style interpolation family. It merges the five local
HelpSteer2 LoRA adapters according to each lambda row, generates responses on
the same fixed prompt set, computes lightweight proxy scores, and summarizes
preference-weighted utility.

The proxy scores are deterministic surface-level heuristics. They are not
HelpSteer2 human labels, not reward-model scores, and not final thesis
evidence.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import generate_response
from src.helpsteer2_scoring_utils import PROXY_COLUMNS, score_response
from src.merge_utils import load_model_with_weighted_lora_adapters_multi


ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
LAMBDA_COLUMNS = tuple(f"lambda_{attribute}" for attribute in ATTRIBUTES)
PREFERENCE_COLUMNS = tuple(f"p_{attribute}" for attribute in ATTRIBUTES)
REQUIRED_COEFFICIENT_COLUMNS = {
    "preference_name",
    "method",
    "method_family",
    "hyperparameter_id",
    "hyperparameters_json",
    *PREFERENCE_COLUMNS,
    *LAMBDA_COLUMNS,
    "l1_distance_to_p",
    "l2_distance_to_p",
}
REQUIRED_PROMPT_FIELDS = {"prompt_id", "category", "prompt", "notes"}
COST_KEY_COLUMNS = ("method", "preference_name", "hyperparameter_id")
COST_COLUMNS = (
    "runtime_seconds",
    "peak_memory_mb",
    "solver_iterations",
    "solver_success",
)
TOLERANCE = 1e-6


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def adapter_paths_from_dir(adapter_dir: Path) -> list[Path]:
    """Return the default HelpSteer2 adapter paths in objective order."""
    return [
        adapter_dir / f"helpsteer2-gpt2-{attribute}-adapter"
        for attribute in ATTRIBUTES
    ]


def validate_adapter_paths(adapter_paths: list[Path]) -> None:
    """Check that every expected PEFT adapter folder is present."""
    missing: list[str] = []
    for attribute, path in zip(ATTRIBUTES, adapter_paths):
        if not path.is_dir():
            missing.append(f"{attribute}: missing directory {path}")
            continue
        if not (path / "adapter_config.json").is_file():
            missing.append(f"{attribute}: missing adapter_config.json in {path}")
        if not any(
            (path / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            missing.append(
                f"{attribute}: missing adapter_model.safetensors or "
                f"adapter_model.bin in {path}"
            )
    if missing:
        raise FileNotFoundError(
            "Missing HelpSteer2 adapter files:\n" + "\n".join(missing)
        )


def read_csv_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read a CSV file and validate that required columns exist."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns.difference(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file contains no rows: {path}")
    return rows


def read_prompt_jsonl(path: Path) -> list[dict[str, str]]:
    """Read and validate the fixed prompt JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt JSONL not found: {path}")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as prompt_file:
        for line_number, line in enumerate(prompt_file, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Empty prompt line at {path}:{line_number}")
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Prompt line {line_number} must be an object.")
            missing = REQUIRED_PROMPT_FIELDS.difference(row)
            if missing:
                raise ValueError(
                    f"Prompt line {line_number} is missing fields: "
                    + ", ".join(sorted(missing))
                )
            prompt_id = str(row["prompt_id"]).strip()
            prompt = str(row["prompt"]).strip()
            category = str(row["category"]).strip()
            if not prompt_id or not prompt or not category:
                raise ValueError(
                    f"Prompt line {line_number} has empty prompt_id, category, "
                    "or prompt."
                )
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate prompt_id found: {prompt_id}")
            seen_ids.add(prompt_id)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": str(row["prompt"]),
                    "notes": str(row["notes"]),
                }
            )
    if not rows:
        raise ValueError(f"Prompt file contains no prompts: {path}")
    return rows


def parse_vector(row: dict[str, str], columns: tuple[str, ...], label: str) -> list[float]:
    """Parse a finite numeric vector from CSV string columns."""
    values: list[float] = []
    for column in columns:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} column {column} must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} column {column} must be finite.")
        values.append(value)
    return values


def validate_simplex(values: list[float], label: str) -> None:
    """Validate that a vector lies on the simplex within tolerance."""
    if len(values) != len(ATTRIBUTES):
        raise ValueError(f"{label} must have {len(ATTRIBUTES)} values.")
    if any(value < -TOLERANCE for value in values):
        raise ValueError(f"{label} contains negative values: {values}")
    total = sum(values)
    if abs(total - 1.0) > TOLERANCE:
        raise ValueError(f"{label} must sum to 1.0, received {total:.10f}.")


def validate_coefficient_rows(rows: list[dict[str, str]]) -> None:
    """Validate coefficient rows before model generation starts."""
    for row_index, row in enumerate(rows, start=1):
        lambdas = parse_vector(row, LAMBDA_COLUMNS, f"row {row_index} lambda")
        preferences = parse_vector(
            row,
            PREFERENCE_COLUMNS,
            f"row {row_index} preference",
        )
        validate_simplex(lambdas, f"row {row_index} lambda")
        validate_simplex(preferences, f"row {row_index} preference")


def read_cost_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Read optional coefficient-computation cost rows."""
    if not path.is_file():
        print(f"Cost file not found, continuing without cost join: {path}")
        return {}

    required = set(COST_KEY_COLUMNS).union(COST_COLUMNS)
    rows = read_csv_rows(path, required)
    return {
        (
            row["method"],
            row["preference_name"],
            row["hyperparameter_id"],
        ): row
        for row in rows
    }


def cost_for_row(
    row: dict[str, str],
    cost_lookup: dict[tuple[str, str, str], dict[str, str]],
) -> dict[str, str]:
    """Return cost fields for a coefficient row when available."""
    key = (row["method"], row["preference_name"], row["hyperparameter_id"])
    cost_row = cost_lookup.get(key, {})
    return {column: cost_row.get(column, "") for column in COST_COLUMNS}


def set_generation_seed(seed: int, coefficient_index: int, prompt_index: int) -> None:
    """Set deterministic generation seeds for one row/prompt pair."""
    effective_seed = seed + coefficient_index * 1000 + prompt_index
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def coefficient_metadata(row: dict[str, str]) -> dict[str, Any]:
    """Extract shared method, preference, lambda, and distance fields."""
    metadata: dict[str, Any] = {
        "preference_name": row["preference_name"],
        "method": row["method"],
        "method_family": row["method_family"],
        "hyperparameter_id": row["hyperparameter_id"],
        "hyperparameters_json": row["hyperparameters_json"],
        "l1_distance_to_p": float(row["l1_distance_to_p"]),
        "l2_distance_to_p": float(row["l2_distance_to_p"]),
    }
    metadata.update(
        {column: float(row[column]) for column in PREFERENCE_COLUMNS}
    )
    metadata.update({column: float(row[column]) for column in LAMBDA_COLUMNS})
    return metadata


def compute_utility(score_fields: dict[str, Any], preferences: list[float]) -> float:
    """Compute preference-weighted utility from proxy scores."""
    return sum(
        preference * float(score_fields[f"{attribute}_proxy"])
        for preference, attribute in zip(preferences, ATTRIBUTES)
    )


def generate_all_rows(
    coefficient_rows: list[dict[str, str]],
    prompt_rows: list[dict[str, str]],
    adapter_paths: list[Path],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Merge adapters for every coefficient row and generate all responses."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Coefficient rows: {len(coefficient_rows)}")
    print(f"Prompts: {len(prompt_rows)}")
    print(
        "Total generations: "
        f"{len(coefficient_rows) * len(prompt_rows)}"
    )

    generation_rows: list[dict[str, Any]] = []
    for coefficient_index, coefficient_row in enumerate(coefficient_rows):
        lambdas = parse_vector(
            coefficient_row,
            LAMBDA_COLUMNS,
            f"coefficient row {coefficient_index + 1}",
        )
        metadata = coefficient_metadata(coefficient_row)
        print(
            f"\n[{coefficient_index + 1}/{len(coefficient_rows)}] "
            f"{metadata['preference_name']} / {metadata['method']} / "
            f"{metadata['hyperparameter_id']} lambda={lambdas}"
        )

        model, tokenizer = load_model_with_weighted_lora_adapters_multi(
            model_name=model_name,
            adapter_paths=adapter_paths,
            adapter_names=ATTRIBUTES,
            weights=lambdas,
            device=device,
        )

        for prompt_index, prompt_row in enumerate(prompt_rows):
            set_generation_seed(seed, coefficient_index, prompt_index)
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_row["prompt"],
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            generation_rows.append(
                {
                    **metadata,
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt_category": prompt_row["category"],
                    "prompt": prompt_row["prompt"],
                    "generated_response": response,
                }
            )

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generation_rows


def score_generation_rows(
    generation_rows: list[dict[str, Any]],
    cost_lookup: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    """Score every generation and add preference-weighted utility."""
    scored_rows: list[dict[str, Any]] = []
    for row in generation_rows:
        score_fields = score_response(str(row["generated_response"]))
        preferences = [float(row[column]) for column in PREFERENCE_COLUMNS]
        scored_rows.append(
            {
                **row,
                **score_fields,
                "utility": round(compute_utility(score_fields, preferences), 6),
                **cost_for_row(row, cost_lookup),
            }
        )
    return scored_rows


def mean(values: list[float]) -> float:
    """Return the arithmetic mean for a non-empty list."""
    return sum(values) / len(values)


def summarize_scores(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate scored generations by coefficient setting."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[
            (
                str(row["preference_name"]),
                str(row["method"]),
                str(row["hyperparameter_id"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        first = rows[0]
        summary: dict[str, Any] = {
            "preference_name": first["preference_name"],
            "method": first["method"],
            "method_family": first["method_family"],
            "hyperparameter_id": first["hyperparameter_id"],
            "hyperparameters_json": first["hyperparameters_json"],
            "num_prompts": len(rows),
            "mean_utility": round(mean([float(row["utility"]) for row in rows]), 6),
            "mean_response_length": round(
                mean([float(row["response_length"]) for row in rows]),
                3,
            ),
            "empty_response_count": sum(bool(row["empty_response"]) for row in rows),
            "l1_distance_to_p": first["l1_distance_to_p"],
            "l2_distance_to_p": first["l2_distance_to_p"],
        }
        summary.update({column: first[column] for column in PREFERENCE_COLUMNS})
        summary.update({column: first[column] for column in LAMBDA_COLUMNS})
        for proxy_column in PROXY_COLUMNS:
            summary[f"mean_{proxy_column}"] = round(
                mean([float(row[proxy_column]) for row in rows]),
                6,
            )
        for cost_column in COST_COLUMNS:
            summary[cost_column] = first.get(cost_column, "")
        summary_rows.append(summary)

    add_baseline_improvements(summary_rows)
    return summary_rows


def add_baseline_improvements(summary_rows: list[dict[str, Any]]) -> None:
    """Add improvement over direct preference and uniform baselines."""
    direct_by_preference = {
        str(row["preference_name"]): float(row["mean_utility"])
        for row in summary_rows
        if row["method"] == "direct_preference"
    }
    uniform_by_preference = {
        str(row["preference_name"]): float(row["mean_utility"])
        for row in summary_rows
        if row["method"] == "uniform"
    }
    for row in summary_rows:
        preference_name = str(row["preference_name"])
        direct_utility = direct_by_preference.get(preference_name)
        uniform_utility = uniform_by_preference.get(preference_name)
        row["direct_preference_utility"] = (
            round(direct_utility, 6) if direct_utility is not None else ""
        )
        row["improvement_over_direct_preference"] = (
            round(float(row["mean_utility"]) - direct_utility, 6)
            if direct_utility is not None
            else ""
        )
        row["uniform_utility"] = (
            round(uniform_utility, 6) if uniform_utility is not None else ""
        )
        row["improvement_over_uniform"] = (
            round(float(row["mean_utility"]) - uniform_utility, 6)
            if uniform_utility is not None
            else ""
        )


def best_methods_by_preference(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the best tested setting for each preference vector."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["preference_name"])].append(row)

    best_rows: list[dict[str, Any]] = []
    for preference_name, rows in sorted(grouped.items()):
        best = max(rows, key=lambda row: float(row["mean_utility"]))
        best_rows.append(
            {
                "preference_name": preference_name,
                "method": best["method"],
                "hyperparameter_id": best["hyperparameter_id"],
                "hyperparameters_json": best["hyperparameters_json"],
                "mean_utility": best["mean_utility"],
                "improvement_over_direct_preference": best[
                    "improvement_over_direct_preference"
                ],
                "improvement_over_uniform": best["improvement_over_uniform"],
            }
        )
    return best_rows


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write rows to a UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_summary(
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write machine-readable summary metadata and best settings."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "HelpSteer2 all-method merge evaluation with lightweight proxy "
            "scores inside the fixed Rewarded-Soups-style interpolation family."
        ),
        "proxy_score_caveat": (
            "Scores are deterministic proxy scores, not HelpSteer2 human labels "
            "or reward-model scores."
        ),
        "inputs": {
            "coefficients_path": args.coefficients_path,
            "prompts_path": args.prompts_path,
            "method_costs_path": str(
                Path(args.coefficients_path).parent
                / "helpsteer2_method_costs.csv"
            ),
            "adapter_dir": args.adapter_dir,
        },
        "generation_settings": {
            "model_name": args.model_name,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "objectives": list(ATTRIBUTES),
        "num_summary_rows": len(summary_rows),
        "best_methods_by_preference": best_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_markdown_summary(
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> None:
    """Write a short human-readable evaluation summary."""
    lines = [
        "# HelpSteer2 All-Method Result Summary",
        "",
        "This report summarizes the full HelpSteer2 all-method adapter-merge "
        "evaluation. All coefficient rows were evaluated on the same fixed "
        "prompt set inside the fixed Rewarded-Soups-style interpolation family.",
        "",
        "The scores are lightweight proxy scores. They are not HelpSteer2 human "
        "labels, not reward-model scores, and should not be interpreted as final "
        "thesis evidence.",
        "",
        "## Best Tested Setting by Preference",
        "",
        "| Preference | Best method | Hyperparameter | Mean utility | Improvement over direct | Improvement over uniform |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in best_rows:
        lines.append(
            "| {preference_name} | {method} | `{hyperparameter_id}` | "
            "{mean_utility} | {improvement_over_direct_preference} | "
            "{improvement_over_uniform} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Evaluation Files",
            "",
            "- `results/helpsteer2_all_method_generations.csv`: raw generated responses.",
            "- `results/helpsteer2_all_method_scores.csv`: generated responses with proxy scores and utility.",
            "- `results/helpsteer2_all_method_result_summary.csv`: aggregate utility and method comparisons.",
            "- `results/helpsteer2_all_method_result_summary.json`: machine-readable run metadata and best settings.",
            "",
            "## Limitations",
            "",
            "- Proxy scores are surface-level heuristics.",
            "- Generated responses do not automatically have HelpSteer2 labels.",
            "- The evaluation does not establish global Pareto-front improvement.",
            "- Stronger reward-model or human evaluation remains future work.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all output paths for this evaluation run."""
    return {
        "generations": output_dir / "helpsteer2_all_method_generations.csv",
        "scores": output_dir / "helpsteer2_all_method_scores.csv",
        "summary_csv": output_dir / "helpsteer2_all_method_result_summary.csv",
        "summary_md": output_dir / "helpsteer2_all_method_result_summary.md",
        "summary_json": output_dir / "helpsteer2_all_method_result_summary.json",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Evaluate all HelpSteer2 coefficient-method adapter merges."
    )
    parser.add_argument(
        "--coefficients_path",
        "--coefficients-path",
        dest="coefficients_path",
        default="results/helpsteer2_all_method_coefficients.csv",
    )
    parser.add_argument(
        "--prompts_path",
        "--prompts-path",
        dest="prompts_path",
        default="data/evaluation_prompts/helpsteer2_fixed_prompts.jsonl",
    )
    parser.add_argument(
        "--adapter_dir",
        "--adapter-dir",
        dest="adapter_dir",
        default="adapters",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="results",
    )
    parser.add_argument("--model_name", "--model-name", dest="model_name", default="gpt2")
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=80,
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", "--top-p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the full all-method HelpSteer2 merge evaluation."""
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")

    coefficients_path = resolve_project_path(args.coefficients_path)
    prompts_path = resolve_project_path(args.prompts_path)
    adapter_dir = resolve_project_path(args.adapter_dir)
    output_dir = resolve_project_path(args.output_dir)
    cost_path = coefficients_path.parent / "helpsteer2_method_costs.csv"
    paths = output_paths(output_dir)

    adapter_paths = adapter_paths_from_dir(adapter_dir)
    validate_adapter_paths(adapter_paths)

    coefficient_rows = read_csv_rows(
        coefficients_path,
        REQUIRED_COEFFICIENT_COLUMNS,
    )
    validate_coefficient_rows(coefficient_rows)
    prompt_rows = read_prompt_jsonl(prompts_path)
    cost_lookup = read_cost_rows(cost_path)

    generation_rows = generate_all_rows(
        coefficient_rows=coefficient_rows,
        prompt_rows=prompt_rows,
        adapter_paths=adapter_paths,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    scored_rows = score_generation_rows(generation_rows, cost_lookup)
    summary_rows = summarize_scores(scored_rows)
    best_rows = best_methods_by_preference(summary_rows)

    generation_columns = [
        "preference_name",
        "method",
        "method_family",
        "hyperparameter_id",
        "hyperparameters_json",
        *PREFERENCE_COLUMNS,
        *LAMBDA_COLUMNS,
        "l1_distance_to_p",
        "l2_distance_to_p",
        "prompt_id",
        "prompt_category",
        "prompt",
        "generated_response",
    ]
    score_columns = generation_columns + [
        *PROXY_COLUMNS,
        "response_length",
        "empty_response",
        "utility",
        *COST_COLUMNS,
    ]
    summary_columns = [
        "preference_name",
        "method",
        "method_family",
        "hyperparameter_id",
        "hyperparameters_json",
        *PREFERENCE_COLUMNS,
        *LAMBDA_COLUMNS,
        *(f"mean_{column}" for column in PROXY_COLUMNS),
        "mean_utility",
        "mean_response_length",
        "empty_response_count",
        "num_prompts",
        "l1_distance_to_p",
        "l2_distance_to_p",
        "direct_preference_utility",
        "improvement_over_direct_preference",
        "uniform_utility",
        "improvement_over_uniform",
        *COST_COLUMNS,
    ]

    write_csv(paths["generations"], generation_rows, generation_columns)
    write_csv(paths["scores"], scored_rows, score_columns)
    write_csv(paths["summary_csv"], summary_rows, summary_columns)
    write_markdown_summary(paths["summary_md"], summary_rows, best_rows)
    write_json_summary(paths["summary_json"], summary_rows, best_rows, args)

    print("\nSaved outputs:")
    for path in paths.values():
        print(f"  - {path}")
    print("\nBest tested setting by preference:")
    for row in best_rows:
        print(
            f"  {row['preference_name']}: {row['method']} "
            f"({row['hyperparameter_id']}), "
            f"mean_utility={float(row['mean_utility']):.4f}"
        )
    print(
        "\nProxy-score caveat: these scores are lightweight heuristics, not "
        "HelpSteer2 human labels or reward-model scores."
    )


if __name__ == "__main__":
    main()
