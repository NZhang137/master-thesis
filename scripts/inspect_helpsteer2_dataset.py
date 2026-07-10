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
    INDEPENDENT_SELECTION_OTHER_MAX_RATING,
    INDEPENDENT_SELECTION_RATINGS,
    INDEPENDENT_SELECTION_SEED,
    compute_prompt_overlap_report,
    inspect_helpsteer2_columns,
    load_helpsteer2_split,
    make_independent_attribute_training_texts,
    save_prompt_overlap_report,
    summarize_attribute_counts,
)


DEFAULT_CONFIG_PATH = "configs/tinyllama_helpsteer2_armorm.yaml"
DEFAULT_OUTPUT_PATH = "results/helpsteer2_dataset_summary.json"
DEFAULT_OVERLAP_CSV_PATH = "results/helpsteer2_selection_overlap_matrix.csv"
DEFAULT_OVERLAP_JSON_PATH = "results/helpsteer2_selection_overlap_matrix.json"
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
    parser.add_argument(
        "--selection_seed",
        "--selection-seed",
        dest="selection_seed",
        type=int,
        default=INDEPENDENT_SELECTION_SEED,
    )
    parser.add_argument(
        "--overlap_csv_path",
        "--overlap-csv-path",
        dest="overlap_csv_path",
        default=DEFAULT_OVERLAP_CSV_PATH,
    )
    parser.add_argument(
        "--overlap_json_path",
        "--overlap-json-path",
        dest="overlap_json_path",
        default=DEFAULT_OVERLAP_JSON_PATH,
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
    overlap_csv_path = resolve_project_path(args.overlap_csv_path)
    overlap_json_path = resolve_project_path(args.overlap_json_path)
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
    selected_texts, selection_summaries = make_independent_attribute_training_texts(
        attributes=list(attributes),
        split=args.split,
        max_examples=max_examples,
        seed=args.selection_seed,
        ratings=INDEPENDENT_SELECTION_RATINGS,
    )
    overlap_report = compute_prompt_overlap_report(selection_summaries, attributes)
    save_prompt_overlap_report(
        overlap_report,
        csv_path=overlap_csv_path,
        json_path=overlap_json_path,
    )
    attribute_summaries: dict[str, dict[str, object]] = {}
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {', '.join(columns)}")
    print(f"Attributes: {', '.join(attributes)}\n")

    print("Selection strategy: independent Top-N per attribute")
    print(
        "Selection ratings: "
        + " then ".join(str(rating) for rating in INDEPENDENT_SELECTION_RATINGS)
    )
    print(
        "Selection non-target rating cap: "
        f"<= {INDEPENDENT_SELECTION_OTHER_MAX_RATING}"
    )
    print(f"Selection seed: {args.selection_seed}\n")

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
            f"{attribute}: eval threshold >= {threshold}; "
            f"ratings {{{rating_text}}}; "
            f"selected={len(texts)}; "
            f"selected_rating_counts="
            f"{selection_summaries[attribute]['selected_rating_counts']}"
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
            "training_selection": selection_summaries[attribute],
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
            "evaluation_attribute_min_ratings": min_ratings,
            "max_examples": max_examples,
            "selection_seed": args.selection_seed,
            "ratings_considered": list(INDEPENDENT_SELECTION_RATINGS),
            "non_target_max_rating": INDEPENDENT_SELECTION_OTHER_MAX_RATING,
            "ordering": (
                "independent per attribute: sort each rating bucket by row "
                "index, shuffle with the same seed, concatenate rating 4 then "
                "rating 3, require all non-target ratings <= 3, take Top-N; "
                "overlap allowed"
            ),
            "overlap_csv_path": str(overlap_csv_path),
            "overlap_json_path": str(overlap_json_path),
        },
        "attribute_summaries": attribute_summaries,
        "prompt_overlap": overlap_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")
    print(f"\nSaved dataset summary to {output_path}")
    print(f"Saved prompt overlap CSV to {overlap_csv_path}")
    print(f"Saved prompt overlap JSON to {overlap_json_path}")


if __name__ == "__main__":
    main()
