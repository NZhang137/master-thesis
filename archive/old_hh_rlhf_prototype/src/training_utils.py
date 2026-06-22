"""Minimal supervised training utilities for the infrastructure prototypes."""

from collections.abc import Sequence

import torch


def train_lora_on_texts(
    model,
    tokenizer,
    training_texts: Sequence[str],
    device: torch.device | str,
    num_epochs: int = 2,
    learning_rate: float = 1e-4,
    max_length: int = 512,
    batch_size: int = 1,
) -> list[float]:
    """Train LoRA on prototype texts with causal language modeling.

    This is only a lightweight supervised prototype. It is not full RLHF, not
    PPO, not RLAIF, and not a final preference-training method. In particular,
    it does not compare preference pairs or train a reward model.

    Returns:
        The average language-modeling loss for each epoch.
    """
    texts = [text.strip() for text in training_texts if text.strip()]
    if not texts:
        raise ValueError("At least one non-empty training text is required.")
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    device = torch.device(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
    )
    epoch_losses = []

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        processed_examples = 0

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_texts)
            processed_examples += len(batch_texts)

        average_loss = total_loss / processed_examples
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - average loss: {average_loss:.4f}")

    return epoch_losses


def train_lora_on_prompts(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device | str,
    num_epochs: int = 1,
    lr: float = 1e-4,
    max_length: int = 512,
    batch_size: int = 1,
) -> list[float]:
    """Train LoRA with a minimal language-modeling loop on prompts only.

    This is only a prototype training loop. It is not RLHF, not RLAIF, and not
    preference training. It does not use HH-RLHF chosen/rejected pairs as
    preferences.

    Returns:
        The mean language-modeling loss for each epoch.
    """
    return train_lora_on_texts(
        model=model,
        tokenizer=tokenizer,
        training_texts=prompts,
        device=device,
        num_epochs=num_epochs,
        learning_rate=lr,
        max_length=max_length,
        batch_size=batch_size,
    )
