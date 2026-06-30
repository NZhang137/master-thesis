"""Reusable data utilities for the final HelpSteer2 experiment.

HelpSteer2 ratings select objective-specific supervised prompt/response texts
for independent TinyLlama specialists. Dataset iteration and sorting preserve
deterministic source order for equal ratings.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache


HELPSTEER2_DATASET_NAME = "nvidia/HelpSteer2"
HELPSTEER2_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
LOW_OVERLAP_SELECTION_ORDER = (
    "complexity",
    "verbosity",
    "helpfulness",
    "correctness",
    "coherence",
)
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
        text = format_helpsteer2_text(example)
        if not text.strip():
            raise ValueError("Selected HelpSteer2 training text is empty.")
        scored_rows.append(
            HelpSteer2ScoredRow(index=index, text=text, ratings=ratings)
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


def make_low_overlap_attribute_training_texts(
    attributes: tuple[str, ...] | list[str] = HELPSTEER2_ATTRIBUTES,
    split: str = "train[:100]",
    max_examples: int | None = None,
    attribute_min_ratings: Mapping[str, int] | None = None,
    selection_order: tuple[str, ...] | list[str] = LOW_OVERLAP_SELECTION_ORDER,
) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
    """Select training texts for all attributes with minimal row reuse.

    The first selected attribute gets its highest-rated examples. Later
    attributes prefer rows with fewer previous selections before considering
    rating and original dataset order. This keeps each specialist high-rating
    filtered while reducing overlap between specialists as much as possible.
    """
    normalized_attributes = _normalize_attribute_sequence(
        list(attributes),
        label="attributes",
    )
    normalized_order = _normalize_attribute_sequence(
        list(selection_order),
        label="selection_order",
    )
    unknown = [
        attribute
        for attribute in normalized_attributes
        if attribute not in normalized_order
    ]
    if unknown:
        raise ValueError(
            "selection_order is missing requested attributes: "
            + ", ".join(unknown)
        )
    thresholds = {
        attribute: _resolve_selection_options(
            attribute,
            min_rating=None,
            max_examples=max_examples,
            attribute_min_ratings=attribute_min_ratings,
        )[0]
        for attribute in normalized_attributes
    }
    limit = _resolve_selection_options(
        normalized_attributes[0],
        min_rating=None,
        max_examples=max_examples,
        attribute_min_ratings=attribute_min_ratings,
    )[1]

    scored_rows = _collect_scored_rows(split)
    usage_counts = {row.index: 0 for row in scored_rows}
    texts_by_attribute: dict[str, list[str]] = {}
    summaries: dict[str, dict[str, object]] = {}
    ordered_attributes = [
        attribute for attribute in normalized_order if attribute in normalized_attributes
    ]

    for selection_index, attribute in enumerate(ordered_attributes, start=1):
        threshold = thresholds[attribute]
        candidates = [
            row for row in scored_rows if row.ratings[attribute] >= threshold
        ]
        fallback_rating = None
        if not candidates:
            fallback_rating = max(row.ratings[attribute] for row in scored_rows)
            candidates = [
                row
                for row in scored_rows
                if row.ratings[attribute] == fallback_rating
            ]

        candidates.sort(
            key=lambda row: (
                usage_counts[row.index],
                -row.ratings[attribute],
                row.index,
            )
        )
        selected = candidates if limit is None else candidates[:limit]
        if not selected:
            raise ValueError(
                f"No non-empty training texts were selected for {attribute}."
            )

        prior_usage_counts = Counter(usage_counts[row.index] for row in selected)
        rating_counts = Counter(row.ratings[attribute] for row in selected)
        for row in selected:
            usage_counts[row.index] += 1

        texts_by_attribute[attribute] = [row.text for row in selected]
        summaries[attribute] = {
            "selection_order_index": selection_index,
            "min_rating": threshold,
            "fallback_rating": fallback_rating,
            "available_at_threshold": sum(
                1 for row in scored_rows if row.ratings[attribute] >= threshold
            ),
            "candidate_count": len(candidates),
            "selected_example_count": len(selected),
            "max_examples": limit,
            "prior_usage_counts": dict(sorted(prior_usage_counts.items())),
            "selected_rating_counts": dict(sorted(rating_counts.items())),
            "ordering": "lowest prior use, descending rating, original row index",
        }

    return texts_by_attribute, summaries


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
