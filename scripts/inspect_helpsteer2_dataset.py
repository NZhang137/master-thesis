"""Inspect HelpSteer2 and summarize objective-specific training selections."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import (
    get_attribute_min_ratings,
    get_attribute_order,
    get_max_training_examples_per_attribute,
    load_experiment_config,
    validate_preference_vectors,
)
from src.helpsteer2_utils import (
    HELPSTEER2_ATTRIBUTES,
    HELPSTEER2_DATASET_NAME,
    HELPSTEER2_TEXT_COLUMNS,
    LOW_OVERLAP_SELECTION_ORDER,
    inspect_helpsteer2_columns,
    load_helpsteer2_split,
    make_low_overlap_attribute_training_texts,
    summarize_attribute_counts,
)


DEFAULT_CONFIG_PATH = "configs/tinyllama_helpsteer2_armorm.yaml"
DEFAULT_OUTPUT_PATH = "results/helpsteer2_dataset_summary.json"
EXAMPLE_TEXT_COUNT = 3
EXAMPLE_TEXT_MAX_CHARS = 500


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse config, dataset split, selection limit, and output path."""
    parser = argparse.ArgumentParser(
        description="Inspect HelpSteer2 ratings and training-text selections."
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--split", default="train[:1000]")
    parser.add_argument(
        "--max_examples",
        "--max-examples",
        dest="max_examples",
        type=int,
        default=None,
        help="Optional maximum number of selected texts per attribute.",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        dest="output_path",
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def required_text(config: dict[str, object], key: str) -> str:
    """Return one required non-empty text field from the config."""
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty string.")
    return value.strip()


def preview_text(text: str, max_chars: int = EXAMPLE_TEXT_MAX_CHARS) -> str:
    """Return a compact deterministic training-text preview."""
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def main() -> None:
    """Validate the split, print counts, and save a JSON inspection summary."""
    args = parse_args()
    if not isinstance(args.split, str) or not args.split.strip():
        raise ValueError("split must be a non-empty string.")
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("max_examples must be at least 1 when provided.")

    config_path = resolve_project_path(args.config_path)
    output_path = resolve_project_path(args.output_path)
    config = load_experiment_config(config_path)
    attributes = get_attribute_order(config)
    min_ratings = get_attribute_min_ratings(config)
    configured_max_examples = get_max_training_examples_per_attribute(config)
    max_examples = args.max_examples
    if max_examples is None:
        max_examples = configured_max_examples
    preferences = validate_preference_vectors(config)
    if attributes != HELPSTEER2_ATTRIBUTES:
        raise ValueError(
            "Configured attributes must exactly match the fixed HelpSteer2 "
            "order: " + ", ".join(HELPSTEER2_ATTRIBUTES)
        )
    dataset_name = required_text(config, "dataset_name")
    if dataset_name != HELPSTEER2_DATASET_NAME:
        raise ValueError(
            f"Configured dataset must be {HELPSTEER2_DATASET_NAME!r}, "
            f"got {dataset_name!r}."
        )

    dataset = load_helpsteer2_split(args.split)
    columns = inspect_helpsteer2_columns(args.split)
    required_columns = (*HELPSTEER2_TEXT_COLUMNS, *attributes)
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise ValueError(
            "HelpSteer2 split is missing configured columns: "
            + ", ".join(missing_columns)
        )
    if len(dataset) == 0:
        raise ValueError(f"HelpSteer2 split contains no rows: {args.split!r}.")

    count_summary = summarize_attribute_counts(
        args.split,
        attribute_min_ratings=min_ratings,
    )
    selected_texts, selection_summaries = make_low_overlap_attribute_training_texts(
        attributes=list(attributes),
        split=args.split,
        max_examples=max_examples,
        attribute_min_ratings=min_ratings,
        selection_order=LOW_OVERLAP_SELECTION_ORDER,
    )
    attribute_summaries: dict[str, dict[str, object]] = {}
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {', '.join(columns)}")
    print(f"Attributes: {', '.join(attributes)}\n")

    print(
        "Low-overlap selection order: "
        + " -> ".join(LOW_OVERLAP_SELECTION_ORDER)
        + "\n"
    )

    for attribute in attributes:
        threshold = min_ratings[attribute]
        texts = selected_texts[attribute]
        if not texts or any(not text.strip() for text in texts):
            raise ValueError(
                f"Attribute {attribute!r} produced no non-empty training texts."
            )
        counts = count_summary[attribute]
        rating_counts = counts["rating_counts"]
        rating_text = ", ".join(
            f"{rating}={rating_counts[rating]}" for rating in range(5)
        )
        print(
            f"{attribute}: threshold >= {threshold}; "
            f"ratings {{{rating_text}}}; "
            f"selected={len(texts)}; "
            f"prior_use={selection_summaries[attribute]['prior_usage_counts']}"
        )
        print("  example: " + preview_text(texts[0], 180).replace("\n", " "))
        attribute_summaries[attribute] = {
            "rating_counts": {
                str(rating): int(count) for rating, count in rating_counts.items()
            },
            "min_rating": threshold,
            "available_at_threshold": int(
                counts["selected_count_at_threshold"]
            ),
            "selected_example_count": len(texts),
            "max_examples": max_examples,
            "low_overlap_selection": selection_summaries[attribute],
            "example_training_texts": [
                preview_text(text) for text in texts[:EXAMPLE_TEXT_COUNT]
            ],
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config_path,
        "experiment_name": required_text(config, "experiment_name"),
        "dataset_name": dataset_name,
        "split": args.split,
        "row_count": len(dataset),
        "columns": list(columns),
        "attributes": list(attributes),
        "preference_vectors": {
            name: list(vector) for name, vector in preferences.items()
        },
        "selection": {
            "attribute_min_ratings": min_ratings,
            "small_split_fallback": "highest observed rating",
            "max_examples": max_examples,
            "selection_order": list(LOW_OVERLAP_SELECTION_ORDER),
            "ordering": (
                "descending rating buckets, then lowest prior use within "
                "rating, then original row index"
            ),
        },
        "attribute_summaries": attribute_summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")
    print(f"\nSaved dataset summary to {output_path}")


if __name__ == "__main__":
    main()
