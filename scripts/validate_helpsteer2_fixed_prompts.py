"""Validate the fixed HelpSteer2 evaluation prompt JSONL file.

The prompt file is used to evaluate every HelpSteer2 coefficient method on the
same inputs. This script checks only the prompt-file structure; it does not run
model generation or scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("prompt_id", "category", "prompt", "notes")
DEFAULT_PROMPT_PATH = "data/evaluation_prompts/helpsteer2_fixed_prompts.jsonl"


def project_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str) -> Path:
    """Resolve a user path relative to the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load and validate JSONL syntax."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as prompt_file:
        for line_number, line in enumerate(prompt_file, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Empty line found at line {line_number}.")
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Validate required fields, uniqueness, and non-empty text fields."""
    if not rows:
        raise ValueError("Prompt file contains no prompts.")

    prompt_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        missing_fields = [field for field in REQUIRED_FIELDS if field not in row]
        if missing_fields:
            raise ValueError(
                f"Line {index} is missing required fields: {', '.join(missing_fields)}"
            )

        for field in REQUIRED_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Line {index} has an empty or non-string {field}.")

        prompt_ids.append(row["prompt_id"])

    duplicate_ids = sorted(
        prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate prompt_id values found: {duplicate_ids}")


def print_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Print a compact validation summary."""
    category_counts = Counter(row["category"] for row in rows)
    print(f"Validated prompt file: {path}")
    print(f"Number of prompts: {len(rows)}")
    print("Categories:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate the fixed HelpSteer2 evaluation prompt JSONL file."
    )
    parser.add_argument(
        "--prompt_path",
        "--prompt-path",
        dest="prompt_path",
        default=DEFAULT_PROMPT_PATH,
        help=f"Prompt JSONL path. Default: {DEFAULT_PROMPT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation."""
    args = parse_args()
    prompt_path = resolve_project_path(args.prompt_path)
    rows = load_jsonl(prompt_path)
    validate_rows(rows)
    print_summary(rows, prompt_path)


if __name__ == "__main__":
    main()
