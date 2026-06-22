"""Utilities for loading GPT-2 with a trainable LoRA adapter."""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_device() -> torch.device:
    """Return a CUDA device when available and CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2_with_lora(
    model_name: str = "gpt2",
    device: torch.device | str | None = None,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load GPT-2, attach the prototype LoRA configuration, and move it."""
    if device is None:
        device = get_device()
    device = torch.device(device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    return model, tokenizer
