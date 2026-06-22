"""Generation utilities for quick qualitative prototype evaluation."""

import torch


PROTOTYPE_TEST_PROMPTS = (
    "Human: What is a good way to stay motivated?\n\nAssistant:",
    "Human: How can I become more confident?\n\nAssistant:",
    "Human: How should I handle a disagreement with a friend?\n\nAssistant:",
)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: torch.device | str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Generate one sampled response for a prompt."""
    device = torch.device(device)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        model.train(was_training)

    response_tokens = output[0, prompt_length:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
