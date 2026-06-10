"""Train separate helpful and harmless GPT-2 LoRA prototype adapters.

Both adapters start from a freshly loaded GPT-2 base model. The script uses
the chosen HH-RLHF conversations for lightweight supervised language modeling.
The rejected responses are not used. This is not full RLHF or PPO, and it is
not the final preference-aware coefficient correction method lambda = f(p, R).
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

from src.data_utils import load_hh_rlhf_objective_dataset
from src.model_utils import get_device, load_gpt2_with_lora
from src.training_utils import train_lora_on_texts


def train_objective_adapter(
    *,
    objective_name: str,
    data_dir: str,
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
    """Train and save one objective-specific adapter from a fresh base model."""
    print(f"\n=== {objective_name} adapter ===")
    print(f"Loading Anthropic/hh-rlhf data_dir={data_dir!r}, split={split!r}")
    print(f"Output adapter path: {output_path}")
    training_texts = load_hh_rlhf_objective_dataset(
        data_dir=data_dir,
        split=split,
    )
    print(f"Loaded {len(training_texts)} chosen training texts.")
    print(
        "Training settings: "
        f"epochs={num_epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}, max_length={max_length}"
    )

    # Reset the seed before each load so both objectives start comparably.
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
    print(f"Saved {objective_name.lower()} adapter to {output_path}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Parse lightweight training settings."""
    parser = argparse.ArgumentParser(
        description="Train helpful and harmless GPT-2 LoRA prototype adapters."
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default="gpt2",
    )
    parser.add_argument("--split", default="train[:100]")
    parser.add_argument(
        "--num_epochs",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=2,
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
        help="Small batches keep the prototype stable on Colab GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    """Train helpful first, then harmless from another fresh GPT-2 model."""
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_objective_adapter(
        objective_name="Helpful",
        data_dir="helpful-base",
        output_path=PROJECT_ROOT / "adapters" / "gpt2-helpful-adapter",
        model_name=args.model_name,
        split=args.split,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    train_objective_adapter(
        objective_name="Harmless",
        data_dir="harmless-base",
        output_path=PROJECT_ROOT / "adapters" / "gpt2-harmless-adapter",
        model_name=args.model_name,
        split=args.split,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    print("\nFinished training both prototype adapters.")


if __name__ == "__main__":
    main()
