"""Load and validate central experiment configuration files."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_SIMPLEX_TOLERANCE = 1e-8


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration as a dictionary.

    PyYAML is imported lazily so modules that only use other project utilities
    do not require it. The YAML root must be a mapping.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "Loading experiment YAML requires PyYAML. Install it with "
            "`pip install pyyaml`."
        ) from error

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, Mapping):
        raise ValueError("The experiment configuration root must be a mapping.")
    return dict(config)


def get_attribute_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the validated objective order from ``config``."""
    attributes = config.get("attributes")
    if (
        not isinstance(attributes, Sequence)
        or isinstance(attributes, (str, bytes))
        or not attributes
    ):
        raise ValueError("config['attributes'] must be a non-empty sequence.")

    normalized = []
    for attribute in attributes:
        if not isinstance(attribute, str) or not attribute.strip():
            raise ValueError("Every attribute must be a non-empty string.")
        normalized.append(attribute.strip())

    if len(set(normalized)) != len(normalized):
        raise ValueError("Attribute names must be unique.")
    return tuple(normalized)


def get_preference_vectors(
    config: Mapping[str, Any],
) -> dict[str, tuple[float, ...]]:
    """Return preference vectors as finite floating-point tuples."""
    configured_vectors = config.get("preference_vectors")
    if not isinstance(configured_vectors, Mapping) or not configured_vectors:
        raise ValueError(
            "config['preference_vectors'] must be a non-empty mapping."
        )

    vectors: dict[str, tuple[float, ...]] = {}
    for name, values in configured_vectors.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every preference vector needs a non-empty name.")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
        ):
            raise ValueError(
                f"Preference vector {name!r} must be a non-empty sequence."
            )

        numeric_values = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Preference vector {name!r} must contain only numbers."
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Preference vector {name!r} contains a non-finite value."
                )
            numeric_values.append(numeric_value)
        vectors[name.strip()] = tuple(numeric_values)

    return vectors


def get_attribute_min_ratings(
    config: Mapping[str, Any],
) -> dict[str, int]:
    """Return validated minimum ratings in the configured attribute order."""
    attributes = get_attribute_order(config)
    configured_ratings = config.get("attribute_min_ratings")
    if not isinstance(configured_ratings, Mapping):
        raise ValueError("config['attribute_min_ratings'] must be a mapping.")

    configured_names = set(configured_ratings)
    expected_names = set(attributes)
    if configured_names != expected_names:
        missing = sorted(expected_names.difference(configured_names))
        extra = sorted(configured_names.difference(expected_names))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ValueError(
            "attribute_min_ratings must match attributes exactly ("
            + "; ".join(details)
            + ")."
        )

    thresholds: dict[str, int] = {}
    for attribute in attributes:
        value = configured_ratings[attribute]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Minimum rating for {attribute!r} must be an integer."
            )
        if value < 0 or value > 4:
            raise ValueError(
                f"Minimum rating for {attribute!r} must be in [0, 4]."
            )
        thresholds[attribute] = value
    return thresholds


def get_max_training_examples_per_attribute(config: Mapping[str, Any]) -> int | None:
    """Return an optional shared cap for selected training texts per attribute."""
    value = config.get("max_training_examples_per_attribute")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "config['max_training_examples_per_attribute'] must be an integer or null."
        )
    if value < 1:
        raise ValueError(
            "config['max_training_examples_per_attribute'] must be at least 1."
        )
    return value


def validate_preference_vectors(
    config: Mapping[str, Any],
    tolerance: float = DEFAULT_SIMPLEX_TOLERANCE,
) -> dict[str, tuple[float, ...]]:
    """Validate that all configured preferences lie on the objective simplex.

    Returns the validated floating-point vectors for convenient reuse.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number.")

    attributes = get_attribute_order(config)
    vectors = get_preference_vectors(config)
    for name, vector in vectors.items():
        if len(vector) != len(attributes):
            raise ValueError(
                f"Preference vector {name!r} has length {len(vector)}; "
                f"expected {len(attributes)} for the configured attributes."
            )
        if any(value < 0.0 for value in vector):
            raise ValueError(
                f"Preference vector {name!r} must be non-negative."
            )
        total = math.fsum(vector)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"Preference vector {name!r} sums to {total:.12g}, not 1."
            )
    return vectors
