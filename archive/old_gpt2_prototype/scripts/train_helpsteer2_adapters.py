"""Train objective-specific GPT-2 LoRA adapters with HelpSteer2 ratings.

Each attribute adapter starts from a freshly loaded GPT-2 base model and a new
LoRA adapter. Attribute ratings select supervised prompt/response texts; this
is not full RLHF or PPO and does not train reward models.
"""

from __future__ import annotations

import argparse
import csv
import gc
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass
class TrainingResult:
    """Summary state returned by one monitored adapter training run."""

    global_step: int
    completed_epochs: int
    stop_requested: bool = False
    current_adapter_stop_requested: bool = False
    max_steps_reached: bool = False
    interrupted: bool = False


class CsvTrainingLogger:
    """Append per-step train/eval metrics to one CSV file."""

    fieldnames = (
        "attribute",
        "global_step",
        "epoch",
        "train_loss",
        "eval_loss",
        "learning_rate",
        "timestamp",
    )

    def __init__(self, path: Path, attribute: str, learning_rate: float) -> None:
        self.path = path
        self.attribute = attribute
        self.learning_rate = learning_rate
        self.rows: list[dict[str, object]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        global_step: int,
        epoch: float,
        train_loss: float | None = None,
        eval_loss: float | None = None,
    ) -> None:
        """Append one metric row and flush it to disk immediately."""
        row = {
            "attribute": self.attribute,
            "global_step": global_step,
            "epoch": f"{epoch:.6f}",
            "train_loss": "" if train_loss is None else f"{train_loss:.8f}",
            "eval_loss": "" if eval_loss is None else f"{eval_loss:.8f}",
            "learning_rate": f"{self.learning_rate:.12g}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.rows.append(row)
        self.flush()

    def flush(self) -> None:
        """Write all accumulated rows to the CSV log file."""
        with self.path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


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


def resolve_project_path(path_value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_texts(texts: list[str]) -> list[str]:
    """Remove empty texts while preserving the input order."""
    prepared = [text.strip() for text in texts if text.strip()]
    if not prepared:
        raise ValueError("At least one non-empty training text is required.")
    return prepared


def tokenize_batch(tokenizer, texts: list[str], max_length: int, device: torch.device):
    """Tokenize one causal-LM batch and move tensors to the active device."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def evaluate_lm_loss(
    *,
    model,
    tokenizer,
    eval_texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> float:
    """Compute average causal-language-modeling loss on eval texts."""
    if not eval_texts:
        raise ValueError("At least one eval text is required.")

    model.eval()
    total_loss = 0.0
    processed_examples = 0
    with torch.no_grad():
        for start in range(0, len(eval_texts), batch_size):
            batch_texts = eval_texts[start : start + batch_size]
            input_ids, attention_mask, labels = tokenize_batch(
                tokenizer,
                batch_texts,
                max_length=max_length,
                device=device,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += float(outputs.loss.item()) * len(batch_texts)
            processed_examples += len(batch_texts)

    model.train()
    return total_loss / processed_examples


def checkpoint_root_for_attribute(
    checkpoint_dir: Path,
    model_name: str,
    attribute: str,
) -> Path:
    """Return the checkpoint root for one HelpSteer2 adapter."""
    return checkpoint_dir / adapter_directory_name(model_name, attribute)


def save_adapter_snapshot(
    *,
    model,
    tokenizer,
    path: Path,
    label: str,
) -> None:
    """Save the current LoRA adapter and tokenizer to ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved {label} to {path}")


def prune_old_checkpoints(checkpoint_root: Path, save_total_limit: int) -> None:
    """Keep only the newest checkpoint folders for one attribute."""
    if save_total_limit < 1 or not checkpoint_root.is_dir():
        return

    checkpoint_dirs = sorted(
        [
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.startswith("step-")
        ],
        key=lambda path: path.stat().st_mtime,
    )
    excess = len(checkpoint_dirs) - save_total_limit
    for old_checkpoint in checkpoint_dirs[: max(excess, 0)]:
        shutil.rmtree(old_checkpoint)
        print(f"Removed old checkpoint: {old_checkpoint}")


def save_periodic_checkpoint(
    *,
    model,
    tokenizer,
    checkpoint_root: Path,
    global_step: int,
    save_total_limit: int,
) -> None:
    """Save a step checkpoint and enforce the retention limit."""
    checkpoint_path = checkpoint_root / f"step-{global_step:08d}"
    save_adapter_snapshot(
        model=model,
        tokenizer=tokenizer,
        path=checkpoint_path,
        label=f"checkpoint at global_step={global_step}",
    )
    prune_old_checkpoints(checkpoint_root, save_total_limit)


def create_tensorboard_writer(
    *,
    use_tensorboard: bool,
    tensorboard_log_dir: Path,
    attribute: str,
):
    """Create a TensorBoard writer when requested."""
    if not use_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "TensorBoard logging requires the tensorboard package. "
            "Install it with `pip install tensorboard`."
        ) from error

    log_dir = tensorboard_log_dir / attribute
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log directory for {attribute}: {log_dir}")
    return SummaryWriter(log_dir=str(log_dir))


def train_lora_with_monitoring(
    *,
    model,
    tokenizer,
    attribute: str,
    training_texts: list[str],
    eval_texts: list[str],
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    batch_size: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    max_steps: int,
    stop_file: Path,
    stop_current_adapter_file: Path,
    csv_logger: CsvTrainingLogger,
    checkpoint_root: Path,
    save_total_limit: int,
    use_tensorboard: bool,
    tensorboard_log_dir: Path,
) -> TrainingResult:
    """Train one LoRA adapter with CSV, checkpoint, stop-file, and eval logs."""
    train_texts = prepare_texts(training_texts)
    eval_texts = prepare_texts(eval_texts)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    writer = create_tensorboard_writer(
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        attribute=attribute,
    )

    model.to(device)
    model.train()
    global_step = 0
    completed_epochs = 0
    recent_losses: list[float] = []

    try:
        for epoch_index in range(num_epochs):
            epoch_number = epoch_index + 1
            total_loss = 0.0
            processed_examples = 0
            last_eval_step: int | None = None
            print(f"\nEpoch {epoch_number}/{num_epochs} for {attribute}")

            for start in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[start : start + batch_size]
                input_ids, attention_mask, labels = tokenize_batch(
                    tokenizer,
                    batch_texts,
                    max_length=max_length,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                global_step += 1
                recent_losses.append(loss_value)
                total_loss += loss_value * len(batch_texts)
                processed_examples += len(batch_texts)
                epoch_progress = epoch_index + (
                    processed_examples / len(train_texts)
                )

                if logging_steps > 0 and global_step % logging_steps == 0:
                    train_loss = float(sum(recent_losses) / len(recent_losses))
                    recent_losses.clear()
                    print(
                        f"{attribute} step {global_step}: "
                        f"train_loss={train_loss:.4f}"
                    )
                    csv_logger.log(
                        global_step=global_step,
                        epoch=epoch_progress,
                        train_loss=train_loss,
                    )
                    if writer is not None:
                        writer.add_scalar(
                            f"{attribute}/train_loss",
                            train_loss,
                            global_step,
                        )
                        writer.flush()

                if eval_steps > 0 and global_step % eval_steps == 0:
                    train_loss = (
                        float(sum(recent_losses) / len(recent_losses))
                        if recent_losses
                        else loss_value
                    )
                    eval_loss = evaluate_lm_loss(
                        model=model,
                        tokenizer=tokenizer,
                        eval_texts=eval_texts,
                        device=device,
                        max_length=max_length,
                        batch_size=batch_size,
                    )
                    print(
                        f"{attribute} step {global_step}: "
                        f"eval_loss={eval_loss:.4f}"
                    )
                    csv_logger.log(
                        global_step=global_step,
                        epoch=epoch_progress,
                        train_loss=train_loss,
                        eval_loss=eval_loss,
                    )
                    if writer is not None:
                        writer.add_scalar(
                            f"{attribute}/eval_loss",
                            eval_loss,
                            global_step,
                        )
                        writer.flush()
                    last_eval_step = global_step

                if save_steps > 0 and global_step % save_steps == 0:
                    save_periodic_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        checkpoint_root=checkpoint_root,
                        global_step=global_step,
                        save_total_limit=save_total_limit,
                    )

                if max_steps > 0 and global_step >= max_steps:
                    completed_epochs = epoch_index
                    print(
                        f"Reached max_steps={max_steps} during "
                        f"{attribute}. Saving current adapter."
                    )
                    return TrainingResult(
                        global_step=global_step,
                        completed_epochs=completed_epochs,
                        max_steps_reached=True,
                    )

            completed_epochs = epoch_number
            average_loss = total_loss / processed_examples
            print(
                f"Epoch {epoch_number}/{num_epochs} - "
                f"average train loss: {average_loss:.4f}"
            )

            if last_eval_step != global_step:
                eval_loss = evaluate_lm_loss(
                    model=model,
                    tokenizer=tokenizer,
                    eval_texts=eval_texts,
                    device=device,
                    max_length=max_length,
                    batch_size=batch_size,
                )
                print(
                    f"{attribute} epoch {epoch_number}: "
                    f"eval_loss={eval_loss:.4f}"
                )
                csv_logger.log(
                    global_step=global_step,
                    epoch=float(epoch_number),
                    train_loss=average_loss,
                    eval_loss=eval_loss,
                )
                if writer is not None:
                    writer.add_scalar(
                        f"{attribute}/eval_loss",
                        eval_loss,
                        global_step,
                    )
                    writer.flush()

            if stop_file.is_file():
                print(
                    f"Stop file found after epoch {epoch_number}: "
                    f"{stop_file}. Stopping the full training run cleanly."
                )
                return TrainingResult(
                    global_step=global_step,
                    completed_epochs=completed_epochs,
                    stop_requested=True,
                )

            if stop_current_adapter_file.is_file():
                print(
                    f"Current-adapter stop file found after epoch "
                    f"{epoch_number}: {stop_current_adapter_file}. "
                    "Saving this adapter and continuing with the next "
                    "selected attribute."
                )
                try:
                    stop_current_adapter_file.unlink()
                    print(
                        "Removed current-adapter stop file: "
                        f"{stop_current_adapter_file}"
                    )
                except OSError as error:
                    print(
                        "Warning: could not remove current-adapter stop file "
                        f"{stop_current_adapter_file}: {error}"
                    )
                return TrainingResult(
                    global_step=global_step,
                    completed_epochs=completed_epochs,
                    current_adapter_stop_requested=True,
                )

        return TrainingResult(
            global_step=global_step,
            completed_epochs=completed_epochs,
        )
    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt received. Saving current adapter and logs "
            "before exiting."
        )
        return TrainingResult(
            global_step=global_step,
            completed_epochs=completed_epochs,
            interrupted=True,
        )
    finally:
        csv_logger.flush()
        if writer is not None:
            writer.flush()
            writer.close()


def train_attribute_adapter(
    *,
    attribute: str,
    output_path: Path,
    model_name: str,
    split: str,
    eval_split: str,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    batch_size: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    max_steps: int,
    stop_file: Path,
    stop_current_adapter_file: Path,
    output_log_dir: Path,
    tensorboard_log_dir: Path,
    checkpoint_dir: Path,
    save_total_limit: int,
    use_tensorboard: bool,
    device: torch.device,
    seed: int = 42,
) -> TrainingResult:
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
    print(f"Loading nvidia/HelpSteer2 eval_split={eval_split!r}")
    eval_texts = make_attribute_training_texts(
        attribute=attribute,
        split=eval_split,
    )
    print(f"Selected {len(eval_texts)} eval texts for {attribute}.")
    print(f"Output adapter path: {output_path}")
    print(f"Full-run stop file: {stop_file}")
    print(f"Current-adapter stop file: {stop_current_adapter_file}")
    print(f"CSV log directory: {output_log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(
        "Training settings: "
        f"epochs={num_epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}, max_length={max_length}, "
        f"logging_steps={logging_steps}, eval_steps={eval_steps}, "
        f"save_steps={save_steps}, max_steps={max_steps}"
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

    csv_logger = CsvTrainingLogger(
        output_log_dir / f"helpsteer2_{attribute}_training_log.csv",
        attribute=attribute,
        learning_rate=learning_rate,
    )
    checkpoint_root = checkpoint_root_for_attribute(
        checkpoint_dir,
        model_name,
        attribute,
    )
    result = TrainingResult(global_step=0, completed_epochs=0)
    result = train_lora_with_monitoring(
        model=model,
        tokenizer=tokenizer,
        attribute=attribute,
        training_texts=training_texts,
        eval_texts=eval_texts,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        stop_file=stop_file,
        stop_current_adapter_file=stop_current_adapter_file,
        csv_logger=csv_logger,
        checkpoint_root=checkpoint_root,
        save_total_limit=save_total_limit,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
    )
    csv_logger.flush()

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved {attribute} adapter to {output_path}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


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
        "--eval_split",
        "--eval-split",
        dest="eval_split",
        default="train[1000:1100]",
    )
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
    parser.add_argument(
        "--logging_steps",
        "--logging-steps",
        dest="logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval_steps",
        "--eval-steps",
        dest="eval_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_steps",
        "--save-steps",
        dest="save_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max_steps",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--stop_file",
        "--stop-file",
        dest="stop_file",
        default="STOP_TRAINING",
    )
    parser.add_argument(
        "--stop_current_adapter_file",
        "--stop-current-adapter-file",
        dest="stop_current_adapter_file",
        default="STOP_CURRENT_ADAPTER",
    )
    parser.add_argument(
        "--output_log_dir",
        "--output-log-dir",
        dest="output_log_dir",
        default="results/training_logs",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        "--tensorboard-log-dir",
        dest="tensorboard_log_dir",
        default="results/tensorboard/helpsteer2",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default="checkpoints/helpsteer2",
    )
    parser.add_argument(
        "--save_total_limit",
        "--save-total-limit",
        dest="save_total_limit",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--use_tensorboard",
        "--use-tensorboard",
        dest="use_tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--stop_all_on_max_steps",
        "--stop-all-on-max-steps",
        dest="stop_all_on_max_steps",
        action="store_true",
        help=(
            "Stop the whole run when max_steps is reached. By default, "
            "max_steps ends only the current adapter and then continues with "
            "the next selected attribute."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate scalar training arguments before loading data or models."""
    if not args.model_name.strip():
        raise ValueError("model_name must be non-empty.")
    if not args.split.strip():
        raise ValueError("split must be non-empty.")
    if not args.eval_split.strip():
        raise ValueError("eval_split must be non-empty.")
    if args.num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if args.logging_steps < 1:
        raise ValueError("logging_steps must be at least 1.")
    if args.eval_steps < 1:
        raise ValueError("eval_steps must be at least 1.")
    if args.save_steps < 1:
        raise ValueError("save_steps must be at least 1.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer.")
    if args.save_total_limit < 1:
        raise ValueError("save_total_limit must be at least 1.")


def main() -> None:
    """Train each selected attribute adapter from another fresh base model."""
    args = parse_args()
    validate_args(args)
    attributes = normalize_attributes(args.attributes)
    output_dir = resolve_output_dir(args.output_dir)
    stop_file = resolve_project_path(args.stop_file)
    stop_current_adapter_file = resolve_project_path(
        args.stop_current_adapter_file
    )
    output_log_dir = resolve_project_path(args.output_log_dir)
    tensorboard_log_dir = resolve_project_path(args.tensorboard_log_dir)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
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
        result = train_attribute_adapter(
            attribute=attribute,
            output_path=output_path,
            model_name=args.model_name,
            split=args.split,
            eval_split=args.eval_split,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            batch_size=args.batch_size,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            max_steps=args.max_steps,
            stop_file=stop_file,
            stop_current_adapter_file=stop_current_adapter_file,
            output_log_dir=output_log_dir,
            tensorboard_log_dir=tensorboard_log_dir,
            checkpoint_dir=checkpoint_dir,
            save_total_limit=args.save_total_limit,
            use_tensorboard=args.use_tensorboard,
            device=device,
        )
        if result.stop_requested:
            print(
                "\nStop requested. Finished the current epoch and saved the "
                "current adapter. Remaining attributes will not be trained."
            )
            break
        if result.current_adapter_stop_requested:
            print(
                "\nCurrent-adapter stop requested. Saved the current adapter "
                "and continuing with the next selected attribute."
            )
            continue
        if result.max_steps_reached:
            if args.stop_all_on_max_steps:
                print(
                    "\nmax_steps reached. Saved the current adapter. "
                    "Remaining attributes will not be trained because "
                    "--stop_all_on_max_steps was set."
                )
                break
            print(
                "\nmax_steps reached. Saved the current adapter and "
                "continuing with the next selected attribute."
            )
            continue
        if result.interrupted:
            print(
                "\nTraining interrupted. Saved the current adapter. Remaining "
                "attributes will not be trained."
            )
            break

    print("\nFinished training all selected HelpSteer2 prototype adapters.")


if __name__ == "__main__":
    main()
