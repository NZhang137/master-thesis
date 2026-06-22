"""Validate and summarize the central TinyLlama experiment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import (
    get_attribute_min_ratings,
    get_attribute_order,
    load_experiment_config,
    validate_preference_vectors,
)


DEFAULT_CONFIG_PATH = "configs/tinyllama_helpsteer2_armorm.yaml"


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse the optional central-config path."""
    parser = argparse.ArgumentParser(
        description="Validate the TinyLlama + HelpSteer2 + ArmoRM config."
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def required_text(config: dict[str, object], key: str) -> str:
    """Return a required non-empty text setting."""
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty string.")
    return value.strip()


def main() -> None:
    """Load, validate, and print the central experiment settings."""
    args = parse_args()
    config_path = resolve_project_path(args.config_path)
    config = load_experiment_config(config_path)
    attributes = get_attribute_order(config)
    min_ratings = get_attribute_min_ratings(config)
    preferences = validate_preference_vectors(config)

    print("TinyLlama + HelpSteer2 + ArmoRM configuration is valid.\n")
    print(f"Experiment:   {required_text(config, 'experiment_name')}")
    print(f"Base model:   {required_text(config, 'base_model_name')}")
    print(f"Dataset:      {required_text(config, 'dataset_name')}")
    print(f"Reward model: {required_text(config, 'reward_model_name')}")
    print(f"Attributes:   {', '.join(attributes)}")
    print("Attribute minimum ratings:")
    for attribute, threshold in min_ratings.items():
        print(f"  - {attribute}: >= {threshold}")
    print("Preference vectors:")
    for name, vector in preferences.items():
        values = ", ".join(f"{value:.2f}" for value in vector)
        print(f"  - {name}: [{values}]")


if __name__ == "__main__":
    main()
