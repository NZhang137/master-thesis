"""Check locally generated HelpSteer2 PEFT adapter folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


def adapter_directory_name(model_name: str, attribute: str) -> str:
    """Return the adapter directory name used by the training script."""
    model_slug = model_name.strip().replace("/", "-")
    return f"helpsteer2-{model_slug}-{attribute}-adapter"


def resolve_output_dir(path_value: str) -> Path:
    """Resolve the adapter root relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_adapter(attribute: str, adapter_path: Path) -> bool:
    """Check for the PEFT configuration and safetensors weight file."""
    if not adapter_path.is_dir():
        print(f"[FAIL] {attribute}: directory not found: {adapter_path}")
        return False

    expected_files = (
        adapter_path / "adapter_config.json",
        adapter_path / "adapter_model.safetensors",
    )
    missing = [path.name for path in expected_files if not path.is_file()]
    if missing:
        print(
            f"[FAIL] {attribute}: missing {', '.join(missing)} "
            f"in {adapter_path}"
        )
        return False

    print(
        f"[OK] {attribute}: found adapter_config.json and "
        f"adapter_model.safetensors"
    )
    return True


def parse_args() -> argparse.Namespace:
    """Parse the model, attribute, and adapter-root settings."""
    parser = argparse.ArgumentParser(
        description="Check generated HelpSteer2 GPT-2 LoRA adapters."
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default="gpt2",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=list(DEFAULT_ATTRIBUTES),
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="adapters",
    )
    return parser.parse_args()


def main() -> int:
    """Check every requested attribute folder and return a shell status."""
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    attributes = list(
        dict.fromkeys(attribute.strip().lower() for attribute in args.attributes)
    )

    if not args.model_name.strip():
        raise ValueError("model_name must be non-empty.")
    if not attributes or any(not attribute for attribute in attributes):
        raise ValueError("At least one non-empty attribute is required.")
    unsupported = sorted(set(attributes).difference(DEFAULT_ATTRIBUTES))
    if unsupported:
        raise ValueError(
            "Unsupported HelpSteer2 attributes: " + ", ".join(unsupported)
        )

    print("Checking HelpSteer2 prototype LoRA adapters...")
    results = []
    for attribute in attributes:
        adapter_path = output_dir / adapter_directory_name(
            args.model_name,
            attribute,
        )
        results.append(check_adapter(attribute, adapter_path))

    if all(results):
        print(
            "Success: all requested HelpSteer2 adapters contain the expected "
            "PEFT files."
        )
        return 0

    print("Failure: one or more HelpSteer2 adapters are missing or incomplete.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
