"""Verify that both prototype PEFT adapter directories are complete."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATHS = {
    "Helpful": PROJECT_ROOT / "adapters" / "gpt2-helpful-adapter",
    "Harmless": PROJECT_ROOT / "adapters" / "gpt2-harmless-adapter",
}
WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


def check_adapter(name: str, adapter_path: Path) -> bool:
    """Check for the PEFT configuration and one supported weight file."""
    if not adapter_path.is_dir():
        print(f"[FAIL] {name}: directory not found: {adapter_path}")
        return False

    config_path = adapter_path / "adapter_config.json"
    weight_paths = [adapter_path / filename for filename in WEIGHT_FILENAMES]
    existing_weights = [path for path in weight_paths if path.is_file()]

    missing = []
    if not config_path.is_file():
        missing.append("adapter_config.json")
    if not existing_weights:
        missing.append("adapter_model.safetensors or adapter_model.bin")

    if missing:
        print(f"[FAIL] {name}: missing {', '.join(missing)} in {adapter_path}")
        return False

    print(
        f"[OK] {name}: found {config_path.name} and "
        f"{existing_weights[0].name}"
    )
    return True


def main() -> int:
    """Check both expected adapter directories and return a shell status."""
    print("Checking prototype LoRA adapters...")
    results = [
        check_adapter(name, adapter_path)
        for name, adapter_path in ADAPTER_PATHS.items()
    ]

    if all(results):
        print("Success: both prototype adapters contain the expected PEFT files.")
        return 0

    print("Failure: one or more prototype adapters are missing or incomplete.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
