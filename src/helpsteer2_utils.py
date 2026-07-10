"""Reusable data utilities for the final HelpSteer2 experiment.

HelpSteer2 ratings select objective-specific supervised prompt/response texts
for independent TinyLlama specialists. Dataset iteration and sorting preserve
deterministic source order for equal ratings.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path


HELPSTEER2_DATASET_NAME = "nvidia/HelpSteer2"
HELPSTEER2_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
INDEPENDENT_SELECTION_SEED = 1
INDEPENDENT_SELECTION_RATINGS = (4, 3)
INDEPENDENT_SELECTION_OTHER_MAX_RATING = 3
HELPSTEER2_TEXT_COLUMNS = ("prompt", "response")
DEFAULT_MIN_RATING = 3
MIN_RATING = 0
MAX_RATING = 4
DEFAULT_ATTRIBUTE_MIN_RATINGS = {
    "helpfulness": 3,
    "correctness": 3,
    "coherence": 3,
    "complexity": 2,
    "verbosity": 3,
}


@dataclass(frozen=True)
class HelpSteer2ScoredRow:
    """One validated HelpSteer2 row prepared for deterministic selection."""

    index: int
    prompt: str
    text: str
    ratings: dict[str, int]


def _validate_split(split: str) -> str:
    """Return a stripped non-empty Hugging Face split expression."""
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string.")
    return split.strip()


@lru_cache(maxsize=8)
def load_helpsteer2_split(split: str = "train[:100]"):
    """Load and cache one split or split slice from NVIDIA HelpSteer2."""
    normalized_split = _validate_split(split)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Loading HelpSteer2 requires the datasets package. Install it "
            "with `pip install datasets`."
        ) from error
    return load_dataset(HELPSTEER2_DATASET_NAME, split=normalized_split)


def load_helpsteer2_dataset(split: str = "train[:100]"):
    """Backward-compatible alias for :func:`load_helpsteer2_split`."""
    return load_helpsteer2_split(split)


def inspect_helpsteer2_columns(split: str = "train[:100]") -> tuple[str, ...]:
    """Return dataset columns in their stored order."""
    dataset = load_helpsteer2_split(split)
    columns = tuple(str(column) for column in dataset.column_names)
    if not columns:
        raise ValueError(f"HelpSteer2 split has no columns: {split!r}.")
    return columns


def _normalize_attribute(attribute: str) -> str:
    """Validate and normalize one HelpSteer2 attribute name."""
    if not isinstance(attribute, str):
        raise ValueError("attribute must be a string.")
    normalized = attribute.strip().lower()
    if normalized not in HELPSTEER2_ATTRIBUTES:
        raise ValueError(
            f"Unsupported HelpSteer2 attribute {attribute!r}. Choose one of: "
            + ", ".join(HELPSTEER2_ATTRIBUTES)
            + "."
        )
    return normalized


def _parse_rating(value: object, attribute: str) -> int:
    """Return one validated integer rating in the HelpSteer2 range."""
    if isinstance(value, bool):
        raise ValueError(f"HelpSteer2 {attribute!r} ratings must be integers.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"HelpSteer2 {attribute!r} ratings must be integers."
        ) from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"HelpSteer2 {attribute!r} ratings must be integers.")
    rating = int(numeric)
    if rating < MIN_RATING or rating > MAX_RATING:
        raise ValueError(
            f"HelpSteer2 {attribute!r} rating must be in "
            f"[{MIN_RATING}, {MAX_RATING}]."
        )
    return rating


def _validate_row_fields(example: Mapping[str, object]) -> None:
    """Check that one row contains text fields and all fixed attributes."""
    required = (*HELPSTEER2_TEXT_COLUMNS, *HELPSTEER2_ATTRIBUTES)
    missing = [field for field in required if field not in example]
    if missing:
        raise ValueError(
            "HelpSteer2 row is missing required fields: " + ", ".join(missing)
        )


def format_helpsteer2_text(example: Mapping[str, object]) -> str:
    """Format one HelpSteer2 prompt/response pair for causal-LM training."""
    if not isinstance(example, Mapping):
        raise TypeError("Each HelpSteer2 example must be a mapping.")
    _validate_row_fields(example)
    prompt = str(example["prompt"] or "").strip()
    response = str(example["response"] or "").strip()
    if not prompt or not response:
        raise ValueError("HelpSteer2 prompt and response must be non-empty.")
    return f"Human: {prompt}\n\nAssistant: {response}"


def _resolve_selection_options(
    attribute: str,
    min_rating: int | None,
    max_examples: int | None,
    attribute_min_ratings: Mapping[str, int] | None,
) -> tuple[int, int | None]:
    """Resolve an explicit, configured, or built-in attribute threshold."""
    if min_rating is not None:
        threshold = min_rating
    elif attribute_min_ratings is not None:
        if attribute not in attribute_min_ratings:
            raise ValueError(
                f"attribute_min_ratings is missing {attribute!r}."
            )
        threshold = attribute_min_ratings[attribute]
    else:
        threshold = DEFAULT_ATTRIBUTE_MIN_RATINGS[attribute]

    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise ValueError(
            f"Minimum rating for {attribute!r} must be an integer."
        )
    if threshold < MIN_RATING or threshold > MAX_RATING:
        raise ValueError(
            f"min_rating must be in [{MIN_RATING}, {MAX_RATING}]."
        )
    if max_examples is not None:
        if isinstance(max_examples, bool) or not isinstance(max_examples, int):
            raise ValueError("max_examples must be a positive integer or None.")
        if max_examples < 1:
            raise ValueError("max_examples must be at least 1.")
    return threshold, max_examples


def make_attribute_training_texts(
    attribute: str,
    split: str = "train[:100]",
    min_rating: int | None = None,
    max_examples: int | None = None,
    attribute_min_ratings: Mapping[str, int] | None = None,
) -> list[str]:
    """Create deterministic supervised texts for one objective specialist.

    An explicit ``min_rating`` overrides ``attribute_min_ratings`` and the
    built-in defaults. The built-in threshold is 2 for complexity and 3 for
    the other attributes. If a small split has no row at the resolved
    threshold, rows with its highest observed rating are selected instead.
    Results are sorted by descending rating and then original dataset index.
    """
    normalized_attribute = _normalize_attribute(attribute)
    threshold, limit = _resolve_selection_options(
        normalized_attribute,
        min_rating,
        max_examples,
        attribute_min_ratings,
    )
    dataset = load_helpsteer2_split(split)
    scored_texts: list[tuple[int, int, str]] = []

    for index, example in enumerate(dataset):
        if not isinstance(example, Mapping):
            raise TypeError("Every HelpSteer2 row must be a mapping.")
        _validate_row_fields(example)
        rating = _parse_rating(example[normalized_attribute], normalized_attribute)
        text = format_helpsteer2_text(example)
        if not text.strip():
            raise ValueError("Selected HelpSteer2 training text is empty.")
        scored_texts.append((rating, index, text))

    if not scored_texts:
        raise ValueError(f"HelpSteer2 split contains no examples: {split!r}.")

    selected = [item for item in scored_texts if item[0] >= threshold]
    if not selected:
        highest_rating = max(item[0] for item in scored_texts)
        selected = [item for item in scored_texts if item[0] == highest_rating]

    selected.sort(key=lambda item: (-item[0], item[1]))
    if limit is not None:
        selected = selected[:limit]
    texts = [text for _, _, text in selected]
    if not texts or any(not text.strip() for text in texts):
        raise ValueError(
            f"No non-empty training texts were selected for {normalized_attribute}."
        )
    return texts


def _collect_scored_rows(split: str) -> list[HelpSteer2ScoredRow]:
    """Load one split and validate text plus all HelpSteer2 ratings once."""
    dataset = load_helpsteer2_split(split)
    scored_rows: list[HelpSteer2ScoredRow] = []
    for index, example in enumerate(dataset):
        if not isinstance(example, Mapping):
            raise TypeError("Every HelpSteer2 row must be a mapping.")
        _validate_row_fields(example)
        ratings = {
            attribute: _parse_rating(example[attribute], attribute)
            for attribute in HELPSTEER2_ATTRIBUTES
        }
        prompt = str(example["prompt"] or "").strip()
        text = format_helpsteer2_text(example)
        if not text.strip():
            raise ValueError("Selected HelpSteer2 training text is empty.")
        scored_rows.append(
            HelpSteer2ScoredRow(
                index=index,
                prompt=prompt,
                text=text,
                ratings=ratings,
            )
        )
    if not scored_rows:
        raise ValueError(f"HelpSteer2 split contains no examples: {split!r}.")
    return scored_rows


def _normalize_attribute_sequence(
    attributes: tuple[str, ...] | list[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """Validate and de-duplicate an attribute sequence while preserving order."""
    normalized: list[str] = []
    for attribute in attributes:
        value = _normalize_attribute(attribute)
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError(f"{label} must contain at least one attribute.")
    return tuple(normalized)


def _stable_prompt_hash(prompt: str) -> str:
    """Return a deterministic compact identifier for one prompt string."""
    return sha256(prompt.encode("utf-8")).hexdigest()


def _shuffled_rating_bucket(
    rows: list[HelpSteer2ScoredRow],
    *,
    attribute: str,
    rating: int,
    seed: int,
    other_max_rating: int = INDEPENDENT_SELECTION_OTHER_MAX_RATING,
) -> list[HelpSteer2ScoredRow]:
    """Return one rating bucket shuffled with a fixed seed."""
    bucket = sorted(
        (
            row
            for row in rows
            if row.ratings[attribute] == rating
            and all(
                row.ratings[other_attribute] <= other_max_rating
                for other_attribute in HELPSTEER2_ATTRIBUTES
                if other_attribute != attribute
            )
        ),
        key=lambda row: row.index,
    )
    shuffled = list(bucket)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def make_independent_attribute_training_texts(
    attributes: tuple[str, ...] | list[str] = HELPSTEER2_ATTRIBUTES,
    split: str = "train[:100]",
    max_examples: int | None = None,
    seed: int = INDEPENDENT_SELECTION_SEED,
    ratings: tuple[int, ...] | list[int] = INDEPENDENT_SELECTION_RATINGS,
) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
    """Select each attribute independently by its own rating buckets.

    For every requested attribute, rating buckets are considered in descending
    preference order (by default 4, then 3). Each bucket is shuffled with the
    same fixed seed, and examples are taken until ``max_examples`` is reached.
    Rows are eligible only when all non-target HelpSteer2 attributes have
    ratings <= 3. Attributes do not interact: no prior-use sorting and no
    cross-attribute deduplication are applied, so overlap between attribute
    selections is allowed and measured separately.
    """
    normalized_attributes = _normalize_attribute_sequence(
        list(attributes),
        label="attributes",
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    normalized_ratings: list[int] = []
    for rating in ratings:
        if isinstance(rating, bool) or not isinstance(rating, int):
            raise ValueError("ratings must contain only integers.")
        if rating < MIN_RATING or rating > MAX_RATING:
            raise ValueError(
                f"ratings must be in [{MIN_RATING}, {MAX_RATING}]."
            )
        if rating not in normalized_ratings:
            normalized_ratings.append(rating)
    if not normalized_ratings:
        raise ValueError("ratings must contain at least one rating.")
    limit = _resolve_selection_options(
        normalized_attributes[0],
        min_rating=None,
        max_examples=max_examples,
        attribute_min_ratings=None,
    )[1]

    scored_rows = _collect_scored_rows(split)
    texts_by_attribute: dict[str, list[str]] = {}
    summaries: dict[str, dict[str, object]] = {}
    selection_shortages: list[str] = []

    for attribute in normalized_attributes:
        selected: list[HelpSteer2ScoredRow] = []
        candidate_counts = {
            rating: len(
                _shuffled_rating_bucket(
                    scored_rows,
                    attribute=attribute,
                    rating=rating,
                    seed=seed,
                )
            )
            for rating in normalized_ratings
        }
        for rating in normalized_ratings:
            bucket = _shuffled_rating_bucket(
                scored_rows,
                attribute=attribute,
                rating=rating,
                seed=seed,
            )
            remaining = None if limit is None else limit - len(selected)
            if remaining is not None and remaining <= 0:
                break
            selected.extend(bucket if remaining is None else bucket[:remaining])

        rating_counts = Counter(row.ratings[attribute] for row in selected)
        mean_ratings = (
            {
                target_attribute: (
                    sum(row.ratings[target_attribute] for row in selected)
                    / len(selected)
                )
                for target_attribute in normalized_attributes
            }
            if selected
            else {}
        )

        texts_by_attribute[attribute] = [row.text for row in selected]
        summaries[attribute] = {
            "selection_strategy": "independent_top_n_by_attribute_rating",
            "seed": seed,
            "ratings_considered": list(normalized_ratings),
            "candidate_counts_by_rating": dict(sorted(candidate_counts.items())),
            "selected_example_count": len(selected),
            "max_examples": limit,
            "selected_rating_counts": dict(sorted(rating_counts.items())),
            "selected_mean_ratings": mean_ratings,
            "selected_row_indices": [row.index for row in selected],
            "selected_prompt_hashes": [
                _stable_prompt_hash(row.prompt) for row in selected
            ],
            "ordering": (
                "independent per attribute: shuffle rating buckets with fixed "
                "seed, take rating 4 before rating 3, require non-target "
                "ratings <= 3, allow overlap"
            ),
        }
        if limit is not None and len(selected) < limit:
            counts_text = ", ".join(
                f"rating {rating}={candidate_counts[rating]}"
                for rating in normalized_ratings
            )
            selection_shortages.append(
                f"{attribute}: selected {len(selected)} texts; "
                f"need {limit}; eligible candidates after non-target <= "
                f"{INDEPENDENT_SELECTION_OTHER_MAX_RATING}: {counts_text}"
            )

    if selection_shortages:
        considered = ", ".join(str(rating) for rating in normalized_ratings)
        raise ValueError(
            "Not enough independent Top-N training texts for all attributes. "
            f"Ratings considered: {considered}. Details:\n  "
            + "\n  ".join(selection_shortages)
        )

    return texts_by_attribute, summaries


def compute_prompt_overlap_report(
    selection_summaries: Mapping[str, Mapping[str, object]],
    attributes: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Compute absolute and percentage overlaps between selected prompt sets."""
    normalized_attributes = _normalize_attribute_sequence(
        list(attributes),
        label="attributes",
    )
    prompt_sets: dict[str, set[str]] = {}
    for attribute in normalized_attributes:
        summary = selection_summaries[attribute]
        prompt_hashes = summary.get("selected_prompt_hashes")
        if not isinstance(prompt_hashes, list) or not all(
            isinstance(value, str) for value in prompt_hashes
        ):
            raise ValueError(
                f"Selection summary for {attribute!r} is missing prompt hashes."
            )
        prompt_sets[attribute] = set(prompt_hashes)

    absolute_matrix: dict[str, dict[str, int]] = {}
    percent_matrix: dict[str, dict[str, float]] = {}
    pairs: list[dict[str, object]] = []
    for attribute_a in normalized_attributes:
        absolute_matrix[attribute_a] = {}
        percent_matrix[attribute_a] = {}
        prompts_a = prompt_sets[attribute_a]
        for attribute_b in normalized_attributes:
            prompts_b = prompt_sets[attribute_b]
            overlap_count = len(prompts_a.intersection(prompts_b))
            denominator = len(prompts_a)
            overlap_percent = (
                0.0 if denominator == 0 else 100.0 * overlap_count / denominator
            )
            absolute_matrix[attribute_a][attribute_b] = overlap_count
            percent_matrix[attribute_a][attribute_b] = overlap_percent
            pairs.append(
                {
                    "attribute_a": attribute_a,
                    "attribute_b": attribute_b,
                    "overlap_count": overlap_count,
                    "overlap_percent": overlap_percent,
                    "size_a": len(prompts_a),
                    "size_b": len(prompts_b),
                }
            )

    return {
        "attributes": list(normalized_attributes),
        "absolute_matrix": absolute_matrix,
        "percent_matrix": percent_matrix,
        "pairs": pairs,
        "percent_denominator": "row attribute selected prompt set size",
    }


def save_prompt_overlap_report(
    overlap_report: Mapping[str, object],
    *,
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    """Save pairwise prompt overlap counts and percentages as CSV and JSON."""
    csv_output_path = Path(csv_path)
    json_output_path = Path(json_path)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = overlap_report.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("overlap_report must contain a list under 'pairs'.")
    with csv_output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "attribute_a",
                "attribute_b",
                "overlap_count",
                "overlap_percent",
                "size_a",
                "size_b",
            ],
        )
        writer.writeheader()
        for row in pairs:
            if not isinstance(row, Mapping):
                raise ValueError("overlap_report pairs must be mappings.")
            writer.writerow(
                {
                    "attribute_a": row["attribute_a"],
                    "attribute_b": row["attribute_b"],
                    "overlap_count": row["overlap_count"],
                    "overlap_percent": f"{float(row['overlap_percent']):.6f}",
                    "size_a": row["size_a"],
                    "size_b": row["size_b"],
                }
            )

    with json_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(overlap_report, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")


def summarize_attribute_counts(
    split: str = "train[:100]",
    attribute_min_ratings: Mapping[str, int] | None = None,
) -> dict[str, dict[str, object]]:
    """Count ratings and threshold-selected rows for every attribute."""
    dataset = load_helpsteer2_split(split)
    if len(dataset) == 0:
        raise ValueError(f"HelpSteer2 split contains no examples: {split!r}.")

    counters = {attribute: Counter() for attribute in HELPSTEER2_ATTRIBUTES}
    for example in dataset:
        if not isinstance(example, Mapping):
            raise TypeError("Every HelpSteer2 row must be a mapping.")
        _validate_row_fields(example)
        for attribute in HELPSTEER2_ATTRIBUTES:
            counters[attribute][_parse_rating(example[attribute], attribute)] += 1

    summary: dict[str, dict[str, object]] = {}
    for attribute in HELPSTEER2_ATTRIBUTES:
        threshold, _ = _resolve_selection_options(
            attribute,
            min_rating=None,
            max_examples=None,
            attribute_min_ratings=attribute_min_ratings,
        )
        rating_counts = {
            rating: counters[attribute].get(rating, 0)
            for rating in range(MIN_RATING, MAX_RATING + 1)
        }
        selected_count = sum(
            count
            for rating, count in rating_counts.items()
            if rating >= threshold
        )
        if selected_count == 0:
            highest = max(
                rating for rating, count in rating_counts.items() if count > 0
            )
            selected_count = rating_counts[highest]
        summary[attribute] = {
            "rating_counts": rating_counts,
            "min_rating": threshold,
            "selected_count_at_threshold": selected_count,
        }
    return summary
