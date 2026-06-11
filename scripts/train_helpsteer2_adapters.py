"""Train objective-specific GPT-2 LoRA adapters with HelpSteer2 ratings.

Each attribute adapter starts from a freshly loaded GPT-2 base model and a new
LoRA adapter. Attribute ratings select supervised prompt/response texts; this
is not full RLHF or PPO and does not train reward models.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helpsteer2_utils import (
    HELPSTEER2_ATTRIBUTES,
    make_attribute_training_texts,
)
from src.model_utils import get_device, load_gpt2_with_lora
from src.training_utils import train_lora_on_texts


def adapter_directory_name(model_name: str, attribute: str) -> str:
    """Return the output folder name for one model and attribute."""
    model_slug = model_name.strip().replace("/", "-")
    return f"helpsteer2-{model_slug}-{attribute}-adapter"


def normalize_attributes(attributes: list[str]) -> list[str]:
    """Validate requested attributes and remove duplicates in input order."""
    normalized = []
    for value in attributes:
        attribute = value.strip().lower()
        if attribute not in HELPSTEER2_ATTRIBUTES:
            supported = ", ".join(HELPSTEER2_ATTRIBUTES)
            raise ValueError(
                f"Unsupported HelpSteer2 attribute {value!r}. "
                f"Choose one of: {supported}."
            )
        if attribute not in normalized:
            normalized.append(attribute)

    if not normalized:
        raise ValueError("At least one HelpSteer2 attribute is required.")
    return normalized


def resolve_output_dir(path_value: str) -> Path:
    """Resolve the adapter output directory relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_attribute_adapter(
    *,
    attribute: str,
    output_path: Path,
    model_name: str,
    split: str,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
) -> None:
    """Train and save one attribute adapter from a fresh GPT-2 base model."""
    print(f"\n=== HelpSteer2 {attribute} adapter ===")
    print(f"Loading nvidia/HelpSteer2 split={split!r}")
    training_texts = make_attribute_training_texts(
        attribute=attribute,
        split=split,
    )
    print(
        f"Selected {len(training_texts)} supervised texts with high "
        f"{attribute} ratings."
    )
    print(f"Output adapter path: {output_path}")
    print(
        "Training settings: "
        f"epochs={num_epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}, max_length={max_length}"
    )

    # Repeating the seed and model load keeps every specialist independent.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Loading a fresh {model_name} model with a new LoRA adapter.")
    model, tokenizer = load_gpt2_with_lora(
        model_name=model_name,
        device=device,
    )
    model.print_trainable_parameters()

    train_lora_on_texts(
        model=model,
        tokenizer=tokenizer,
        training_texts=training_texts,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        batch_size=batch_size,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved {attribute} adapter to {output_path}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Parse HelpSteer2 data, model, training, and output settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Train separate GPT-2 LoRA adapters from attribute-rated "
            "HelpSteer2 examples."
        )
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default="gpt2",
    )
    parser.add_argument("--split", default="train[:100]")
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=list(HELPSTEER2_ATTRIBUTES),
        help="One or more HelpSteer2 attributes to train.",
    )
    parser.add_argument(
        "--num_epochs",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max_length",
        "--max-length",
        dest="max_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="adapters",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate scalar training arguments before loading data or models."""
    if not args.model_name.strip():
        raise ValueError("model_name must be non-empty.")
    if not args.split.strip():
        raise ValueError("split must be non-empty.")
    if args.num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")


def main() -> None:
    """Train each selected attribute adapter from another fresh base model."""
    args = parse_args()
    validate_args(args)
    attributes = normalize_attributes(args.attributes)
    output_dir = resolve_output_dir(args.output_dir)
    device = get_device()

    print(f"Using device: {device}")
    print(f"Selected attributes: {', '.join(attributes)}")
    print(
        "This is supervised prototype training from attribute-rated examples, "
        "not full RLHF/PPO or reward-model training."
    )

    for attribute in attributes:
        output_path = output_dir / adapter_directory_name(
            args.model_name,
            attribute,
        )
        train_attribute_adapter(
            attribute=attribute,
            output_path=output_path,
            model_name=args.model_name,
            split=args.split,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )

    print("\nFinished training all selected HelpSteer2 prototype adapters.")


if __name__ == "__main__":
    main()
