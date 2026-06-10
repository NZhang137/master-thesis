"""GPT-2 and LoRA model construction helpers."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import torch


def get_device() -> torch.device:
    """Return CUDA when available and CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory() -> None:
    """Release Python garbage and unused CUDA cache memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_gpt2_lora(
    model_name: str = "gpt2",
    *,
    use_8bit: bool | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Sequence[str] = ("c_attn", "c_proj"),
) -> tuple[Any, Any]:
    """Load a causal language model and attach a trainable LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    device = get_device()
    if use_8bit is None:
        use_8bit = device.type == "cuda"
    if use_8bit and device.type != "cuda":
        raise ValueError("8-bit loading requires a supported accelerator.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if use_8bit:
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": "auto",
            }
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if device.type == "cuda" else torch.float32
        )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_8bit:
        from peft import prepare_model_for_kbit_training

        base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
    )
    model = get_peft_model(base_model, lora_config)

    if not use_8bit:
        model = model.to(device)

    return model, tokenizer


def save_adapter(model: Any, tokenizer: Any, output_dir: str | Path) -> Path:
    """Save the active PEFT adapter and its tokenizer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def load_lora_adapter(
    adapter_path: str | Path,
    *,
    base_model_name: str = "gpt2",
    is_trainable: bool = False,
) -> tuple[Any, Any]:
    """Load a saved LoRA adapter on its base causal language model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
    ).to(device)
    return model, tokenizer
