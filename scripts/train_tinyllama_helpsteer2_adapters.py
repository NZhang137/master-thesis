"""Train independent TinyLlama LoRA adapters on HelpSteer2 attributes.

Each specialist starts from the same freshly loaded base model. HelpSteer2
ratings select supervised prompt/response texts; this is not RLHF/PPO. Optional
ArmoRM scoring is observational monitoring and never affects optimization.
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

from src.experiment_config import (
    get_attribute_min_ratings,
    get_attribute_order,
    get_max_training_examples_per_attribute,
    load_experiment_config,
    validate_preference_vectors,
)
from src.helpsteer2_utils import HELPSTEER2_ATTRIBUTES, make_attribute_training_texts
from src.tinyllama_training_utils import (
    ARMORM_HELPSTEER_OBJECTIVES,
    CsvTrainingLogger,
    RewardMonitor,
    TrainingResult,
    load_tinyllama_with_lora,
    train_with_monitoring,
)


DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ARMORM_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
DEFAULT_CONFIG_PATH = "configs/tinyllama_helpsteer2_armorm.yaml"


def adapter_directory_name(attribute: str) -> str:
    """Return the stable output directory name for one TinyLlama specialist."""
    return f"tinyllama-helpsteer2-{attribute}-adapter"


def normalize_attributes(values: list[str]) -> list[str]:
    """Validate attributes and remove duplicates while retaining order."""
    attributes = []
    for value in values:
        attribute = value.strip().lower()
        if attribute not in HELPSTEER2_ATTRIBUTES:
            raise ValueError(
                f"Unsupported attribute {value!r}. Choose from: "
                + ", ".join(HELPSTEER2_ATTRIBUTES)
            )
        if attribute not in attributes:
            attributes.append(attribute)
    if not attributes:
        raise ValueError("At least one HelpSteer2 attribute is required.")
    return attributes


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_attribute_adapter(
    *,
    attribute: str,
    model_name: str,
    split: str,
    eval_split: str,
    min_rating: int,
    max_training_examples: int | None,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    lr_scheduler_type: str,
    min_lr_ratio: float,
    warmup_ratio: float,
    max_length: int,
    batch_size: int,
    output_path: Path,
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
    use_armorm_monitoring: bool,
    armorm_model_name: str,
    reward_eval_steps: int,
    reward_monitor_num_prompts: int,
    reward_eval_prompts_path: Path,
    reward_max_new_tokens: int,
    reward_batch_size: int,
    reward_csv_mode: str,
    reward_output_dir: Path,
    seed: int = 67,
) -> TrainingResult:
    """Load a fresh base model, train one specialist, and save its adapter."""
    print(f"\n=== TinyLlama HelpSteer2 {attribute} adapter ===")
    print(f"Attribute selection threshold: rating >= {min_rating}")
    if max_training_examples is not None:
        print(
            "Training text cap: "
            f"top {max_training_examples} examples after descending-rating sorting"
        )
    print(f"Loading training split {split!r}")
    training_texts = make_attribute_training_texts(
        attribute,
        split,
        min_rating=min_rating,
        max_examples=max_training_examples,
    )
    print(f"Selected {len(training_texts)} high-{attribute} training texts.")
    print(f"Loading evaluation split {eval_split!r}")
    eval_texts = make_attribute_training_texts(
        attribute,
        eval_split,
        min_rating=min_rating,
    )
    print(f"Selected {len(eval_texts)} evaluation texts.")
    print(f"Output adapter path: {output_path}")
    if use_armorm_monitoring:
        print("ArmoRM monitoring enabled: True")
        objective_index, objective_name = ARMORM_HELPSTEER_OBJECTIVES[attribute]
        print(
            "ArmoRM monitored objective: "
            f"{objective_name} (rewards index {objective_index})"
        )
        print(f"ArmoRM reward evaluation steps: {reward_eval_steps}")
        print(f"ArmoRM monitoring prompts: {reward_monitor_num_prompts}")
        print(f"ArmoRM prompt file: {reward_eval_prompts_path}")
        print(f"ArmoRM maximum new tokens: {reward_max_new_tokens}")
        print(f"ArmoRM reward batch size: {reward_batch_size}")
        print(f"ArmoRM reward CSV mode: {reward_csv_mode}")
        print(f"ArmoRM reward output directory: {reward_output_dir}")
    else:
        print("ArmoRM monitoring disabled.")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"LR scheduler: {lr_scheduler_type}")
    print(f"Warmup ratio: {warmup_ratio}")
    if lr_scheduler_type == "cosine_decay":
        print(f"Minimum LR ratio: {min_lr_ratio}")
    print("LoRA rank: 8")
    print("LoRA alpha: 16")
    print("LoRA dropout: 0.1")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # A fresh load for every attribute keeps all specialists independent.
    print(f"Loading a fresh base model: {model_name}")
    model, tokenizer = load_tinyllama_with_lora(model_name)
    model.print_trainable_parameters()

    csv_logger = CsvTrainingLogger(
        output_log_dir / f"tinyllama_helpsteer2_{attribute}_training_log.csv",
        attribute,
        learning_rate,
    )
    reward_monitor = None
    if use_armorm_monitoring:
        reward_monitor = RewardMonitor(
            attribute=attribute,
            reward_model_name=armorm_model_name,
            prompts_path=reward_eval_prompts_path,
            num_prompts=reward_monitor_num_prompts,
            max_new_tokens=reward_max_new_tokens,
            batch_size=reward_batch_size,
            prompt_csv_path=(
                reward_output_dir
                / f"tinyllama_helpsteer2_{attribute}_reward_prompts.csv"
            ),
            summary_csv_path=(
                reward_output_dir
                / f"tinyllama_helpsteer2_{attribute}_reward_summary.csv"
            ),
            csv_mode=reward_csv_mode,
        )

    result = train_with_monitoring(
        model=model,
        tokenizer=tokenizer,
        attribute=attribute,
        training_texts=training_texts,
        eval_texts=eval_texts,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        min_lr_ratio=min_lr_ratio,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        stop_file=stop_file,
        stop_current_adapter_file=stop_current_adapter_file,
        csv_logger=csv_logger,
        checkpoint_root=checkpoint_dir / adapter_directory_name(attribute),
        save_total_limit=save_total_limit,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        reward_monitor=reward_monitor,
        reward_eval_steps=reward_eval_steps,
        seed=seed,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved {attribute} adapter to {output_path}")
    if result.current_adapter_stop_requested:
        try:
            stop_current_adapter_file.unlink(missing_ok=True)
            print(f"Removed {stop_current_adapter_file.name}.")
        except OSError as error:
            print(f"Warning: could not remove stop file: {error}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    """Parse TinyLlama HelpSteer2 training settings."""
    parser = argparse.ArgumentParser(
        description="Train independent TinyLlama HelpSteer2 LoRA adapters."
    )
    parser.add_argument(
        "--config_path",
        "--config-path",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default=DEFAULT_MODEL_NAME,
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
    )
    parser.add_argument(
        "--num_epochs", "--num-epochs", dest="num_epochs", type=int, default=1
    )
    parser.add_argument(
        "--learning_rate",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (recommended range: 0.0 to 0.01).",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        "--lr-scheduler-type",
        dest="lr_scheduler_type",
        choices=("constant", "cosine_decay"),
        default="constant",
    )
    parser.add_argument(
        "--min_lr_ratio",
        "--min-lr-ratio",
        dest="min_lr_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--warmup_ratio",
        "--warmup-ratio",
        dest="warmup_ratio",
        type=float,
        default=0.06,
        help="Fraction of total optimizer steps used for linear LR warmup.",
    )
    parser.add_argument(
        "--max_length", "--max-length", dest="max_length", type=int, default=512
    )
    parser.add_argument(
        "--batch_size", "--batch-size", dest="batch_size", type=int, default=1
    )
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir", default="adapters"
    )
    parser.add_argument(
        "--logging_steps",
        "--logging-steps",
        dest="logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval_steps", "--eval-steps", dest="eval_steps", type=int, default=100
    )
    parser.add_argument(
        "--save_steps", "--save-steps", dest="save_steps", type=int, default=500
    )
    parser.add_argument(
        "--max_steps", "--max-steps", dest="max_steps", type=int, default=-1
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
        default="results/tinyllama_helpsteer2_training_logs",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        "--tensorboard-log-dir",
        dest="tensorboard_log_dir",
        default="results/tensorboard/tinyllama_helpsteer2",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default="checkpoints/tinyllama_helpsteer2",
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
        "--use_armorm_monitoring",
        "--use-armorm-monitoring",
        dest="use_armorm_monitoring",
        action="store_true",
    )
    parser.add_argument(
        "--armorm_model_name",
        "--armorm-model-name",
        dest="armorm_model_name",
        default=DEFAULT_ARMORM_MODEL,
    )
    parser.add_argument(
        "--reward_eval_steps",
        "--reward-eval-steps",
        dest="reward_eval_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--reward_monitor_num_prompts",
        "--reward-monitor-num-prompts",
        dest="reward_monitor_num_prompts",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--reward_eval_prompts_path",
        "--reward-eval-prompts-path",
        dest="reward_eval_prompts_path",
        default="data/evaluation_prompts/helpsteer2_reward_monitor_prompts.jsonl",
    )
    parser.add_argument(
        "--reward_max_new_tokens",
        "--reward-max-new-tokens",
        dest="reward_max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--reward_batch_size",
        "--reward-batch-size",
        dest="reward_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--reward_csv_mode",
        "--reward-csv-mode",
        dest="reward_csv_mode",
        choices=("overwrite", "append"),
        default="overwrite",
    )
    parser.add_argument(
        "--reward_output_dir",
        "--reward-output-dir",
        dest="reward_output_dir",
        default="results/tinyllama_helpsteer2_reward_monitoring",
    )
    parser.add_argument(
        "--stop_all_on_max_steps",
        "--stop-all-on-max-steps",
        dest="stop_all_on_max_steps",
        action="store_true",
        help="Stop the full run when max_steps is reached for one adapter.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate scalar settings before downloading data or model weights."""
    if (
        not args.config_path.strip()
        or not args.model_name.strip()
        or not args.split.strip()
        or not args.eval_split.strip()
    ):
        raise ValueError(
            "config_path, model_name, split, and eval_split must be non-empty."
        )
    if args.num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if args.weight_decay > 0.01:
        print("Warning: recommended weight_decay values are between 0.0 and 0.01.")
    if not 0 < args.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be greater than 0 and at most 1.")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1.")
    if args.max_length < 2 or args.batch_size < 1:
        raise ValueError("max_length must be at least 2 and batch_size at least 1.")
    for name in ("logging_steps", "eval_steps", "save_steps", "reward_eval_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be at least 1.")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer.")
    if args.save_total_limit < 1 or args.reward_max_new_tokens < 1:
        raise ValueError("save_total_limit and reward_max_new_tokens must be positive.")
    if args.reward_monitor_num_prompts < 1:
        raise ValueError("reward_monitor_num_prompts must be at least 1.")
    if args.reward_batch_size != 1:
        raise ValueError(
            "reward_batch_size currently supports only 1; true batching is not "
            "implemented."
        )
    if args.use_armorm_monitoring and not args.armorm_model_name.strip():
        raise ValueError("armorm_model_name must be non-empty when monitoring is enabled.")


def main() -> None:
    """Train each selected specialist from an independent TinyLlama base load."""
    args = parse_args()
    validate_args(args)
    config = load_experiment_config(resolve_project_path(args.config_path))
    configured_attributes = get_attribute_order(config)
    if configured_attributes != HELPSTEER2_ATTRIBUTES:
        raise ValueError(
            "Configured attributes must exactly match the fixed HelpSteer2 "
            "order: " + ", ".join(HELPSTEER2_ATTRIBUTES)
        )
    min_ratings = get_attribute_min_ratings(config)
    max_training_examples = get_max_training_examples_per_attribute(config)
    validate_preference_vectors(config)
    attributes = normalize_attributes(args.attributes)
    output_dir = resolve_project_path(args.output_dir)
    stop_file = resolve_project_path(args.stop_file)
    stop_current_adapter_file = resolve_project_path(args.stop_current_adapter_file)
    output_log_dir = resolve_project_path(args.output_log_dir)
    tensorboard_log_dir = resolve_project_path(args.tensorboard_log_dir)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    reward_prompts_path = resolve_project_path(args.reward_eval_prompts_path)
    reward_output_dir = resolve_project_path(args.reward_output_dir)

    print(f"Selected attributes: {', '.join(attributes)}")
    print(
        "Selection thresholds: "
        + ", ".join(
            f"{attribute}>={min_ratings[attribute]}" for attribute in attributes
        )
    )
    if max_training_examples is not None:
        print(
            "Training example cap: "
            f"{max_training_examples} per attribute, using highest ratings first"
        )
    print(
        "Training uses attribute-selected supervised HelpSteer2 texts. "
        "External reward monitoring, when enabled, is evaluation only."
    )

    for attribute in attributes:
        result = train_attribute_adapter(
            attribute=attribute,
            model_name=args.model_name,
            split=args.split,
            eval_split=args.eval_split,
            min_rating=min_ratings[attribute],
            max_training_examples=max_training_examples,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            min_lr_ratio=args.min_lr_ratio,
            warmup_ratio=args.warmup_ratio,
            max_length=args.max_length,
            batch_size=args.batch_size,
            output_path=output_dir / adapter_directory_name(attribute),
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
            use_armorm_monitoring=args.use_armorm_monitoring,
            armorm_model_name=args.armorm_model_name,
            reward_eval_steps=args.reward_eval_steps,
            reward_monitor_num_prompts=args.reward_monitor_num_prompts,
            reward_eval_prompts_path=reward_prompts_path,
            reward_max_new_tokens=args.reward_max_new_tokens,
            reward_batch_size=args.reward_batch_size,
            reward_csv_mode=args.reward_csv_mode,
            reward_output_dir=reward_output_dir,
        )
        if result.stop_requested:
            print("Saved the current adapter; stopping this training run.")
            break
        if result.current_adapter_stop_requested:
            print("Saved the current adapter; stopping this training run.")
            break
        if result.max_steps_reached:
            if args.stop_all_on_max_steps:
                print("max_steps reached; stopping the full training run.")
                break
            print("max_steps reached; continuing with the next attribute.")
            continue
        if result.interrupted:
            print("Saved the interrupted adapter; stopping the full training run.")
            break

    print("Finished TinyLlama HelpSteer2 adapter training.")


if __name__ == "__main__":
    main()
