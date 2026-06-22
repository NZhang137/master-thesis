"""Generate responses from fixed five-objective HelpSteer2 LoRA merges.

This script tests many-objective adapter interpolation only. It does not
compute a relationship matrix or apply the M1 coefficient mapping.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import generate_response
from src.merge_utils import load_model_with_weighted_lora_adapters_multi


ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

MERGE_CANDIDATES = (
    ("uniform", (0.2, 0.2, 0.2, 0.2, 0.2)),
    ("one_hot_helpfulness", (1.0, 0.0, 0.0, 0.0, 0.0)),
    ("one_hot_correctness", (0.0, 1.0, 0.0, 0.0, 0.0)),
    ("one_hot_coherence", (0.0, 0.0, 1.0, 0.0, 0.0)),
    ("one_hot_complexity", (0.0, 0.0, 0.0, 1.0, 0.0)),
    ("one_hot_verbosity", (0.0, 0.0, 0.0, 0.0, 1.0)),
    ("helpful_correct_coherent", (0.4, 0.3, 0.2, 0.05, 0.05)),
    ("concise_quality", (0.3, 0.3, 0.3, 0.05, 0.05)),
    ("detailed_answer", (0.25, 0.25, 0.2, 0.15, 0.15)),
)

TEST_PROMPTS = (
    "Human: What is a good way to stay motivated?\n\nAssistant:",
    "Human: How can I improve my study habits?\n\nAssistant:",
    "Human: Explain why sleep is important.\n\nAssistant:",
    "Human: How should I handle a disagreement with a friend?\n\nAssistant:",
)

OUTPUT_COLUMNS = [
    "merge_name",
    "lambda_helpfulness",
    "lambda_correctness",
    "lambda_coherence",
    "lambda_complexity",
    "lambda_verbosity",
    "prompt",
    "generated_response",
]


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse model, adapter, generation, and output settings."""
    parser = argparse.ArgumentParser(
        description="Evaluate fixed five-objective HelpSteer2 LoRA merges."
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default="gpt2",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        dest="output_path",
        default="results/helpsteer2_adapter_merge_generations.csv",
    )
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=100,
    )
    for attribute in ATTRIBUTES:
        parser.add_argument(
            f"--{attribute}_adapter_path",
            f"--{attribute}-adapter-path",
            dest=f"{attribute}_adapter_path",
            default=(
                f"adapters/helpsteer2-gpt2-{attribute}-adapter"
            ),
        )
    return parser.parse_args()


def main() -> None:
    """Evaluate all fixed merges and save their generated responses."""
    args = parse_args()
    if not args.model_name.strip():
        raise ValueError("model_name must be non-empty.")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    adapter_paths = [
        resolve_project_path(getattr(args, f"{attribute}_adapter_path"))
        for attribute in ATTRIBUTES
    ]
    output_path = resolve_project_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for attribute, path in zip(ATTRIBUTES, adapter_paths):
        print(f"{attribute}: {path}")

    rows = []
    for candidate_index, (merge_name, weights) in enumerate(
        MERGE_CANDIDATES,
        start=1,
    ):
        print(
            f"\nMerge {candidate_index}/{len(MERGE_CANDIDATES)}: "
            f"{merge_name} with lambda={list(weights)}"
        )
        model, tokenizer = load_model_with_weighted_lora_adapters_multi(
            model_name=args.model_name,
            adapter_paths=adapter_paths,
            adapter_names=ATTRIBUTES,
            weights=weights,
            device=device,
        )

        for prompt_index, prompt in enumerate(TEST_PROMPTS):
            seed = 42 + prompt_index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append(
                {
                    "merge_name": merge_name,
                    **{
                        f"lambda_{attribute}": weight
                        for attribute, weight in zip(ATTRIBUTES, weights)
                    },
                    "prompt": prompt,
                    "generated_response": response,
                }
            )
            print(
                f"Generated response "
                f"{prompt_index + 1}/{len(TEST_PROMPTS)}"
            )

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} generations to {output_path}")


if __name__ == "__main__":
    main()
