"""Training utilities for TinyLlama HelpSteer2 LoRA specialists.

The utilities implement supervised causal-language-model training on
attribute-selected HelpSteer2 texts. Reward-model support is monitoring only:
reward scores are never used as a training loss or optimization signal.
"""

from __future__ import annotations

import csv
import gc
import json
import random
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingResult:
    """Control state returned after training one attribute adapter."""

    global_step: int
    completed_epochs: int
    stop_requested: bool = False
    current_adapter_stop_requested: bool = False
    max_steps_reached: bool = False
    interrupted: bool = False


ARMORM_HELPSTEER_OBJECTIVES: dict[str, tuple[int, str]] = {
    "helpfulness": (0, "helpsteer-helpfulness"),
    "correctness": (1, "helpsteer-correctness"),
    "coherence": (2, "helpsteer-coherence"),
    "complexity": (3, "helpsteer-complexity"),
    "verbosity": (4, "helpsteer-verbosity"),
}


class CsvTrainingLogger:
    """Write train and evaluation losses for one attribute to CSV."""

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
        learning_rate: float | None = None,
    ) -> None:
        """Append one metric row and flush it immediately."""
        logged_learning_rate = (
            self.learning_rate if learning_rate is None else learning_rate
        )
        self.rows.append(
            {
                "attribute": self.attribute,
                "global_step": global_step,
                "epoch": f"{epoch:.6f}",
                "train_loss": "" if train_loss is None else f"{train_loss:.8f}",
                "eval_loss": "" if eval_loss is None else f"{eval_loss:.8f}",
                "learning_rate": f"{logged_learning_rate:.12g}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.flush()

    def flush(self) -> None:
        """Write all accumulated rows to disk."""
        with self.path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


class _StreamingCsvLogger:
    """Write and immediately flush rows in explicit overwrite or append mode."""

    def __init__(
        self,
        path: Path,
        fieldnames: tuple[str, ...],
        mode: str,
    ) -> None:
        if mode not in {"overwrite", "append"}:
            raise ValueError("CSV mode must be 'overwrite' or 'append'.")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = mode == "overwrite" or not path.exists() or path.stat().st_size == 0
        self.output_file = path.open(
            "w" if mode == "overwrite" else "a",
            encoding="utf-8",
            newline="",
        )
        self.writer = csv.DictWriter(self.output_file, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()
            self.output_file.flush()

    def log(self, row: dict[str, object]) -> None:
        """Write one row and flush it so monitoring survives interruptions."""
        self.writer.writerow(row)
        self.output_file.flush()

    def close(self) -> None:
        """Flush and close the underlying CSV file."""
        if not self.output_file.closed:
            self.output_file.flush()
            self.output_file.close()


def model_input_device(model) -> torch.device:
    """Return the device holding the model input embeddings."""
    return model.get_input_embeddings().weight.device


def load_tinyllama_with_lora(
    model_name: str,
) -> tuple[Any, Any]:
    """Load a fresh TinyLlama model and attach a trainable LoRA adapter."""
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as error:
        raise ImportError(
            "TinyLlama training requires transformers and peft. Install the "
            "Colab dependencies shown in notebook 21."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        raise ValueError(f"Tokenizer for {model_name!r} has no EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.config.use_cache = False
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if torch.cuda.is_available():
        model.enable_input_require_grads()
    return model, tokenizer


def tokenize_batch(tokenizer, texts: list[str], max_length: int, device: torch.device):
    """Tokenize a causal-LM batch and mask padding tokens in labels."""
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
    max_length: int,
    batch_size: int,
) -> float:
    """Compute mean causal-language-model loss over evaluation texts."""
    if not eval_texts:
        raise ValueError("At least one evaluation text is required.")

    device = model_input_device(model)
    model.eval()
    total_loss = 0.0
    processed = 0
    with torch.no_grad():
        for start in range(0, len(eval_texts), batch_size):
            batch_texts = eval_texts[start : start + batch_size]
            input_ids, attention_mask, labels = tokenize_batch(
                tokenizer,
                batch_texts,
                max_length,
                device,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += float(outputs.loss.item()) * len(batch_texts)
            processed += len(batch_texts)
    model.train()
    return total_loss / processed


def save_adapter_snapshot(model, tokenizer, path: Path, label: str) -> None:
    """Save the current PEFT adapter and tokenizer."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved {label} to {path}")


def prune_old_checkpoints(checkpoint_root: Path, save_total_limit: int) -> None:
    """Keep only the newest step checkpoints for one attribute."""
    if not checkpoint_root.is_dir():
        return
    checkpoints = sorted(
        [
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.startswith("step-")
        ],
        key=lambda path: path.stat().st_mtime,
    )
    for old_checkpoint in checkpoints[: max(len(checkpoints) - save_total_limit, 0)]:
        shutil.rmtree(old_checkpoint)
        print(f"Removed old checkpoint: {old_checkpoint}")


def save_periodic_checkpoint(
    model,
    tokenizer,
    checkpoint_root: Path,
    global_step: int,
    save_total_limit: int,
) -> None:
    """Save a numbered adapter checkpoint and enforce retention."""
    save_adapter_snapshot(
        model,
        tokenizer,
        checkpoint_root / f"step-{global_step:08d}",
        f"checkpoint at global_step={global_step}",
    )
    prune_old_checkpoints(checkpoint_root, save_total_limit)


def create_tensorboard_writer(enabled: bool, log_dir: Path, attribute: str):
    """Create one TensorBoard writer when logging is enabled."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "TensorBoard logging requires `pip install tensorboard`."
        ) from error
    attribute_log_dir = log_dir / attribute
    attribute_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log directory: {attribute_log_dir}")
    return SummaryWriter(log_dir=str(attribute_log_dir))


def load_reward_prompts(path: Path) -> list[dict[str, str]]:
    """Load a small JSONL prompt set for optional reward monitoring."""
    if not path.is_file():
        raise FileNotFoundError(f"Reward-monitoring prompt file not found: {path}")
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            prompt_id = str(record.get("prompt_id", "")).strip()
            prompt = str(record.get("prompt", "")).strip()
            if not prompt_id:
                raise ValueError(f"Missing prompt_id at {path}:{line_number}.")
            if not prompt:
                raise ValueError(f"Missing prompt at {path}:{line_number}.")
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                }
            )
    if not prompts:
        raise ValueError(f"Reward-monitoring prompt file is empty: {path}")
    prompt_ids = [record["prompt_id"] for record in prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError(f"Reward-monitoring prompt IDs must be unique: {path}")
    return prompts


class RewardMonitor:
    """Generate fixed answers and score them with an external reward model.

    The resulting scores are observational monitoring metrics. They are not
    backpropagated and never affect the optimizer.
    """

    def __init__(
        self,
        *,
        attribute: str,
        reward_model_name: str,
        prompts_path: Path,
        num_prompts: int,
        max_new_tokens: int,
        batch_size: int,
        prompt_csv_path: Path,
        summary_csv_path: Path,
        csv_mode: str,
    ) -> None:
        if num_prompts < 1:
            raise ValueError("Reward monitoring requires at least one prompt.")
        if batch_size != 1:
            raise ValueError(
                "reward_batch_size currently supports only 1; true batching "
                "is not implemented."
            )
        available_prompts = load_reward_prompts(prompts_path)
        if len(available_prompts) < num_prompts:
            raise ValueError(
                f"Reward-monitoring prompt file contains {len(available_prompts)} "
                f"prompts, but {num_prompts} are required: {prompts_path}"
            )
        self.attribute = attribute
        if attribute not in ARMORM_HELPSTEER_OBJECTIVES:
            raise ValueError(
                f"No ArmoRM HelpSteer objective mapping is configured for {attribute!r}."
            )
        self.reward_objective_index, self.reward_objective_name = (
            ARMORM_HELPSTEER_OBJECTIVES[attribute]
        )
        self.tensorboard_scalar_name = (
            f"armorm_{self.reward_objective_name.replace('-', '_')}"
        )
        self.reward_model_name = reward_model_name
        # Always use the same deterministic first N prompts for every attribute.
        self.prompts = available_prompts[:num_prompts]
        self.max_new_tokens = max_new_tokens
        self.prompt_csv_logger = _StreamingCsvLogger(
            prompt_csv_path,
            (
                "attribute",
                "global_step",
                "prompt_id",
                "prompt",
                "generated_text",
                "reward_objective",
                "reward_score",
                "reward_model",
                "reward_max_new_tokens",
                "timestamp",
            ),
            csv_mode,
        )
        self.summary_csv_logger = _StreamingCsvLogger(
            summary_csv_path,
            (
                "attribute",
                "global_step",
                "reward_objective",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
                "num_prompts",
                "reward_model",
                "reward_max_new_tokens",
                "timestamp",
            ),
            csv_mode,
        )
        self.reward_model = None
        self.reward_tokenizer = None

    def _load_reward_model(self) -> None:
        if self.reward_model is not None:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise ImportError("Reward monitoring requires transformers.") from error

        print(f"Loading reward monitor model: {self.reward_model_name}")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            self.reward_model_name,
            trust_remote_code=True,
        )
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def _extract_objective_score(self, outputs) -> float:
        rewards = getattr(outputs, "rewards", None)
        if rewards is None:
            raise ValueError(
                "The configured reward model returned no multi-objective "
                "reward vector. Objective-specific ArmoRM monitoring requires "
                "outputs.rewards."
            )
        tensor = torch.as_tensor(rewards).detach().float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                "Objective-specific ArmoRM monitoring expects rewards with "
                "shape [batch, objectives]."
            )
        if tensor.shape[0] != 1:
            raise ValueError(
                "reward_batch_size currently supports only 1, but ArmoRM returned "
                f"a batch of {tensor.shape[0]} rewards."
            )
        if tensor.shape[1] <= self.reward_objective_index:
            raise ValueError(
                f"ArmoRM returned only {tensor.shape[1]} objective rewards; "
                f"cannot read {self.reward_objective_name} at index "
                f"{self.reward_objective_index}."
            )
        return float(tensor[0, self.reward_objective_index].item())

    def _generate_answer(self, model, tokenizer, prompt: str) -> str:
        device = model_input_device(model)
        if getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            generated[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()

    def _score_answer(self, prompt: str, answer: str) -> float:
        self._load_reward_model()
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        tokenizer = self.reward_tokenizer
        if getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
            )
            inputs = {"input_ids": input_ids}
        else:
            text = f"Human: {prompt}\n\nAssistant: {answer}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True)

        reward_device = next(self.reward_model.parameters()).device
        inputs = {key: value.to(reward_device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.reward_model(**inputs)
        return self._extract_objective_score(outputs)

    def evaluate(self, model, tokenizer, global_step: int) -> float:
        """Return the mean selected ArmoRM objective over the fixed prompts."""
        was_training = model.training
        model.eval()
        scores: list[float] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            # Load outside inference_mode so library initialization is unaffected.
            self._load_reward_model()
            with torch.inference_mode():
                for record in self.prompts:
                    answer = self._generate_answer(model, tokenizer, record["prompt"])
                    score = self._score_answer(record["prompt"], answer)
                    scores.append(score)
                    self.prompt_csv_logger.log(
                        {
                            "attribute": self.attribute,
                            "global_step": global_step,
                            "prompt_id": record["prompt_id"],
                            "prompt": record["prompt"],
                            "generated_text": answer,
                            "reward_objective": self.reward_objective_name,
                            "reward_score": f"{score:.8f}",
                            "reward_model": self.reward_model_name,
                            "reward_max_new_tokens": self.max_new_tokens,
                            "timestamp": timestamp,
                        }
                    )
        finally:
            if was_training:
                model.train()
        mean_score = float(sum(scores) / len(scores))
        # Population standard deviation describes the complete fixed prompt set.
        std_score = float(statistics.pstdev(scores))
        self.summary_csv_logger.log(
            {
                "attribute": self.attribute,
                "global_step": global_step,
                "reward_objective": self.reward_objective_name,
                "mean_reward": f"{mean_score:.8f}",
                "std_reward": f"{std_score:.8f}",
                "min_reward": f"{min(scores):.8f}",
                "max_reward": f"{max(scores):.8f}",
                "num_prompts": len(scores),
                "reward_model": self.reward_model_name,
                "reward_max_new_tokens": self.max_new_tokens,
                "timestamp": timestamp,
            }
        )
        print(
            f"ArmoRM monitoring at step {global_step}: "
            f"objective={self.reward_objective_name}, "
            f"mean={mean_score:.4f}, std={std_score:.4f}, "
            f"min={min(scores):.4f}, max={max(scores):.4f}"
        )
        return mean_score

    def close(self) -> None:
        """Release the optional reward model and close monitoring CSV files."""
        self.prompt_csv_logger.close()
        self.summary_csv_logger.close()
        if self.reward_model is not None:
            del self.reward_model, self.reward_tokenizer
            self.reward_model = None
            self.reward_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_with_monitoring(
    *,
    model,
    tokenizer,
    attribute: str,
    training_texts: list[str],
    eval_texts: list[str],
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
    reward_monitor: RewardMonitor | None = None,
    reward_eval_steps: int = 1000,
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "constant",
    lr_decay_after_epoch: int = 1,
    lr_decay_factor: float = 0.8,
    min_lr_ratio: float = 0.1,
    warmup_ratio: float = 0.06,
    seed: int = 67,
) -> TrainingResult:
    """Train one adapter with loss, checkpoint, stop, and reward monitoring."""
    train_texts = [text.strip() for text in training_texts if text.strip()]
    evaluation_texts = [text.strip() for text in eval_texts if text.strip()]
    if not train_texts or not evaluation_texts:
        raise ValueError("Training and evaluation text lists must be non-empty.")
    if lr_scheduler_type not in {"constant", "epoch_decay"}:
        raise ValueError("lr_scheduler_type must be 'constant' or 'epoch_decay'.")
    if lr_decay_after_epoch < 1:
        raise ValueError("lr_decay_after_epoch must be at least 1.")
    if not 0 < lr_decay_factor <= 1:
        raise ValueError("lr_decay_factor must be greater than 0 and at most 1.")
    if not 0 < min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be greater than 0 and at most 1.")
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1.")

    steps_per_epoch = (len(train_texts) + batch_size - 1) // batch_size
    total_steps = num_epochs * steps_per_epoch
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(warmup_ratio * total_steps)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The model has no trainable LoRA parameters.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def current_learning_rate() -> float:
        return float(optimizer.param_groups[0]["lr"])

    def learning_rate_at(global_step: int, epoch_index: int) -> float:
        """Return linear warmup LR followed by the configured epoch schedule."""
        if warmup_steps > 0 and global_step < warmup_steps:
            return learning_rate * (global_step + 1) / warmup_steps
        if lr_scheduler_type == "epoch_decay":
            decay_epochs = max(0, epoch_index - lr_decay_after_epoch + 1)
            decayed_lr = learning_rate * (lr_decay_factor**decay_epochs)
            return max(decayed_lr, learning_rate * min_lr_ratio)
        return learning_rate

    print(
        f"Learning-rate warmup: ratio={warmup_ratio:.4g}, "
        f"warmup_steps={warmup_steps}, total_steps={total_steps}"
    )

    writer = create_tensorboard_writer(
        use_tensorboard,
        tensorboard_log_dir,
        attribute,
    )
    model.train()
    device = model_input_device(model)
    global_step = 0
    completed_epochs = 0
    recent_losses: list[float] = []
    stop_requested = False
    current_adapter_stop_requested = False

    def detect_stop_files() -> None:
        """Latch stop requests so training can finish at the next save step."""
        nonlocal stop_requested, current_adapter_stop_requested
        if not current_adapter_stop_requested and stop_current_adapter_file.is_file():
            current_adapter_stop_requested = True
            print(
                f"{stop_current_adapter_file.name} detected. "
                "Training will stop after the next save step."
            )
        if not stop_requested and stop_file.is_file():
            stop_requested = True
            print(
                f"{stop_file.name} detected. "
                "Training will stop after the next save step."
            )

    try:
        detect_stop_files()
        for epoch_index in range(num_epochs):
            epoch_number = epoch_index + 1
            epoch_shuffle_seed = seed + epoch_index
            epoch_train_texts = list(train_texts)
            # Use an epoch-specific deterministic shuffle for reproducibility.
            random.Random(epoch_shuffle_seed).shuffle(epoch_train_texts)
            total_loss = 0.0
            processed = 0
            last_eval_step: int | None = None
            print(
                f"\nEpoch {epoch_number}/{num_epochs} for {attribute} "
                f"(next_learning_rate="
                f"{learning_rate_at(global_step, epoch_index):.8g}, "
                f"shuffle_seed={epoch_shuffle_seed})"
            )

            for start in range(0, len(epoch_train_texts), batch_size):
                batch_texts = epoch_train_texts[start : start + batch_size]
                input_ids, attention_mask, labels = tokenize_batch(
                    tokenizer,
                    batch_texts,
                    max_length,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                step_learning_rate = learning_rate_at(global_step, epoch_index)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = step_learning_rate
                optimizer.step()

                loss_value = float(loss.item())
                global_step += 1
                recent_losses.append(loss_value)
                total_loss += loss_value * len(batch_texts)
                processed += len(batch_texts)
                epoch_progress = epoch_index + processed / len(train_texts)
                detect_stop_files()

                if global_step % logging_steps == 0:
                    train_loss = float(sum(recent_losses) / len(recent_losses))
                    recent_losses.clear()
                    print(f"{attribute} step {global_step}: train_loss={train_loss:.4f}")
                    csv_logger.log(
                        global_step=global_step,
                        epoch=epoch_progress,
                        train_loss=train_loss,
                        learning_rate=current_learning_rate(),
                    )
                    if writer is not None:
                        writer.add_scalar(f"{attribute}/train_loss", train_loss, global_step)
                        writer.add_scalar(
                            f"{attribute}/learning_rate",
                            current_learning_rate(),
                            global_step,
                        )
                        writer.flush()

                if global_step % eval_steps == 0:
                    train_loss = (
                        float(sum(recent_losses) / len(recent_losses))
                        if recent_losses
                        else loss_value
                    )
                    eval_loss = evaluate_lm_loss(
                        model=model,
                        tokenizer=tokenizer,
                        eval_texts=evaluation_texts,
                        max_length=max_length,
                        batch_size=batch_size,
                    )
                    print(f"{attribute} step {global_step}: eval_loss={eval_loss:.4f}")
                    csv_logger.log(
                        global_step=global_step,
                        epoch=epoch_progress,
                        train_loss=train_loss,
                        eval_loss=eval_loss,
                        learning_rate=current_learning_rate(),
                    )
                    if writer is not None:
                        writer.add_scalar(f"{attribute}/eval_loss", eval_loss, global_step)
                        writer.flush()
                    last_eval_step = global_step

                if reward_monitor is not None and global_step % reward_eval_steps == 0:
                    mean_reward = reward_monitor.evaluate(model, tokenizer, global_step)
                    if writer is not None:
                        # Backward-compatible scalar: now the selected HelpSteer
                        # objective for the active adapter, not the global score.
                        writer.add_scalar(
                            f"{attribute}/armorm_mean_reward",
                            mean_reward,
                            global_step,
                        )
                        writer.add_scalar(
                            f"{attribute}/{reward_monitor.tensorboard_scalar_name}",
                            mean_reward,
                            global_step,
                        )
                        writer.flush()

                if global_step % save_steps == 0:
                    if stop_requested or current_adapter_stop_requested:
                        print(
                            "Reached save step. Saving adapter and stopping gracefully."
                        )
                    save_periodic_checkpoint(
                        model,
                        tokenizer,
                        checkpoint_root,
                        global_step,
                        save_total_limit,
                    )
                    if stop_requested or current_adapter_stop_requested:
                        return TrainingResult(
                            global_step=global_step,
                            completed_epochs=completed_epochs,
                            stop_requested=stop_requested,
                            current_adapter_stop_requested=(
                                current_adapter_stop_requested
                            ),
                        )

                if (
                    max_steps > 0
                    and global_step >= max_steps
                    and not (stop_requested or current_adapter_stop_requested)
                ):
                    print(f"Reached max_steps={max_steps} for {attribute}.")
                    return TrainingResult(
                        global_step=global_step,
                        completed_epochs=completed_epochs,
                        max_steps_reached=True,
                    )

            completed_epochs = epoch_number
            average_loss = total_loss / processed
            print(
                f"Epoch {epoch_number}/{num_epochs} - "
                f"average train loss: {average_loss:.4f}"
            )
            if last_eval_step != global_step:
                eval_loss = evaluate_lm_loss(
                    model=model,
                    tokenizer=tokenizer,
                    eval_texts=evaluation_texts,
                    max_length=max_length,
                    batch_size=batch_size,
                )
                csv_logger.log(
                    global_step=global_step,
                    epoch=float(epoch_number),
                    train_loss=average_loss,
                    eval_loss=eval_loss,
                    learning_rate=current_learning_rate(),
                )
                print(f"{attribute} epoch {epoch_number}: eval_loss={eval_loss:.4f}")
                if writer is not None:
                    writer.add_scalar(f"{attribute}/eval_loss", eval_loss, global_step)
                    writer.flush()
            detect_stop_files()

        if stop_requested or current_adapter_stop_requested:
            print(
                "Training completed before another save step. "
                "Saving adapter and stopping gracefully."
            )
        return TrainingResult(
            global_step=global_step,
            completed_epochs=completed_epochs,
            stop_requested=stop_requested,
            current_adapter_stop_requested=current_adapter_stop_requested,
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received; saving this adapter before stopping.")
        return TrainingResult(
            global_step=global_step,
            completed_epochs=completed_epochs,
            interrupted=True,
        )
    finally:
        csv_logger.flush()
        if reward_monitor is not None:
            reward_monitor.close()
        if writer is not None:
            writer.flush()
            writer.close()
