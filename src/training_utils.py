"""Minimal prompt-only training utilities for the infrastructure prototype."""

from collections.abc import Sequence

import torch


def train_lora_on_prompts(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device | str,
    num_epochs: int = 1,
    lr: float = 1e-4,
    max_length: int = 512,
) -> list[float]:
    """Train LoRA with a minimal language-modeling loop on prompts only.

    This is only a prototype training loop. It is not RLHF, not RLAIF, and not
    preference training. It does not use HH-RLHF chosen/rejected pairs as
    preferences.

    Returns:
        The mean language-modeling loss for each epoch.
    """
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if not prompts:
        raise ValueError("At least one prompt is required.")

    device = torch.device(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)
    epoch_losses = []

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(prompts)
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - average loss: {average_loss:.4f}")

    return epoch_losses
