"""Check locally generated TinyLlama HelpSteer2 PEFT adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


def adapter_directory_name(attribute: str) -> str:
    """Return the output name used by the TinyLlama training script."""
    return f"tinyllama-helpsteer2-{attribute}-adapter"


def parse_args() -> argparse.Namespace:
    """Parse the adapter root and selected attributes."""
    parser = argparse.ArgumentParser(
        description="Check TinyLlama HelpSteer2 PEFT adapter folders."
    )
    parser.add_argument(
        "--adapter_dir",
        "--adapter-dir",
        dest="adapter_dir",
        default="adapters",
    )
    parser.add_argument("--attributes", nargs="+", default=list(ATTRIBUTES))
    return parser.parse_args()


def main() -> int:
    """Validate requested adapter folders and return a shell status."""
    args = parse_args()
    root = Path(args.adapter_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    attributes = list(dict.fromkeys(value.strip().lower() for value in args.attributes))
    unsupported = sorted(set(attributes).difference(ATTRIBUTES))
    if not attributes or unsupported:
        raise ValueError(
            "Attributes must be selected from: " + ", ".join(ATTRIBUTES)
        )

    all_valid = True
    for attribute in attributes:
        adapter_path = root / adapter_directory_name(attribute)
        config_path = adapter_path / "adapter_config.json"
        safetensors_path = adapter_path / "adapter_model.safetensors"
        bin_path = adapter_path / "adapter_model.bin"
        missing = []
        if not adapter_path.is_dir():
            missing.append("adapter directory")
        if not config_path.is_file():
            missing.append("adapter_config.json")
        if not safetensors_path.is_file() and not bin_path.is_file():
            missing.append("adapter_model.safetensors or adapter_model.bin")

        if missing:
            all_valid = False
            print(f"[FAIL] {attribute}: missing {', '.join(missing)} at {adapter_path}")
        else:
            weight_name = (
                safetensors_path.name if safetensors_path.is_file() else bin_path.name
            )
            print(f"[OK] {attribute}: adapter_config.json and {weight_name}")

    if all_valid:
        print("Success: all requested TinyLlama HelpSteer2 adapters are complete.")
        return 0
    print("Failure: one or more TinyLlama HelpSteer2 adapters are incomplete.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
