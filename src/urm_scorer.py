"""Five-attribute scoring for ``LxzGordon/URM-LLaMa-3.1-8B``.

The public URM forward pass returns only the learned scalar gate output.  Its
custom model code can also return the ten value-head outputs and five gating
weights when ``return_dict=False``.  The ten outputs are ordered as five
``(mean, uncertainty_parameter)`` pairs.  NB13.1 needs the five means, in the
HelpSteer2 order documented by the model authors, rather than the gated scalar.

URM was trained directly on HelpSteer2.  It is therefore a sensitivity
evaluator for HelpSteer2-trained adapters, not an independent confirmation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


DEFAULT_REWARD_MODEL = "LxzGordon/URM-LLaMa-3.1-8B"
DEFAULT_REVISION = "7b5a6446f55725ae6d86002fdc30f702a2272e0f"
URM_ATTRIBUTES: tuple[str, ...] = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

GOLDEN_SAMPLE = {
    "prompt": "What is the range of the numeric output of a sigmoid node in a neural network?",
    "responses": (
        "The output of a sigmoid node is bounded between -1 and 1.",
        "The output of a sigmoid node is bounded between 0 and 1.",
    ),
    "expected_aggregate_scores": (2.3285412788391113, 12.438033103942871),
    "source": "https://huggingface.co/LxzGordon/URM-LLaMa-3.1-8B",
}


def attribute_indices(attributes: Sequence[str]) -> list[int]:
    """Resolve attribute names to the order declared by the URM authors."""
    indices: list[int] = []
    for attribute in attributes:
        try:
            indices.append(URM_ATTRIBUTES.index(str(attribute)))
        except ValueError as error:
            raise KeyError(
                f"{attribute!r} is not a URM HelpSteer2 attribute; "
                f"known={list(URM_ATTRIBUTES)}"
            ) from error
    return indices


class URMScorer:
    """Return URM's five attribute means for one prompt/answer pair."""

    def __init__(
        self,
        reward_model_name: str = DEFAULT_REWARD_MODEL,
        *,
        revision: str = DEFAULT_REVISION,
        load_in_8bit: bool = True,
        dtype: str = "float16",
        max_length: int = 4096,
    ) -> None:
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported dtype {dtype!r} for URM.")
        self.reward_model_name = reward_model_name
        self.revision = revision
        self.load_in_8bit = bool(load_in_8bit)
        self.dtype = dtype
        self.max_length = int(max_length)
        self._model: Any = None
        self._tokenizer: Any = None
        self._golden_checked = False

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.reward_model_name,
            revision=self.revision,
            trust_remote_code=True,
        )
        if getattr(self._tokenizer, "chat_template", None) is None:
            raise RuntimeError("URM tokenizer exposes no chat template.")

        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, self.dtype),
        }
        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        elif torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name, **kwargs
        )
        if int(getattr(model.config, "num_labels", -1)) != 10:
            raise RuntimeError(
                f"URM must expose 10 value-head outputs, got "
                f"num_labels={getattr(model.config, 'num_labels', None)!r}."
            )
        if not hasattr(model, "weights"):
            raise RuntimeError("Loaded model has no URM gating head named 'weights'.")
        model.requires_grad_(False)
        model.eval()
        self._model = model

    def _encode(self, prompt: str, answer: str) -> dict[str, Any]:
        self._load()
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        rendered = self._tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self._model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items()}

    def _attribute_means_and_weights(
        self, prompt: str, answer: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return five distribution means and the five learned gate weights."""
        import torch

        encoded = self._encode(prompt, answer)
        with torch.inference_mode():
            outputs = self._model(**encoded, return_dict=False)
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise RuntimeError(
                "URM custom forward(return_dict=False) no longer returned "
                "(pooled_logits, pooled_weights)."
            )
        pooled_logits = torch.as_tensor(outputs[0]).detach().float()
        pooled_weights = torch.as_tensor(outputs[1]).detach().float()
        if pooled_logits.shape != (1, 10) or pooled_weights.shape != (1, 5):
            raise RuntimeError(
                "Unexpected URM output shapes: "
                f"logits={tuple(pooled_logits.shape)}, "
                f"weights={tuple(pooled_weights.shape)}"
            )
        means = pooled_logits.reshape(1, 5, 2)[0, :, 0]
        if not torch.isfinite(means).all() or not torch.isfinite(pooled_weights).all():
            raise RuntimeError("URM returned non-finite values.")
        return (
            means.cpu().numpy().astype(np.float64),
            pooled_weights[0].cpu().numpy().astype(np.float64),
        )

    def score(
        self, prompt: str, answer: str, attributes: Sequence[str]
    ) -> np.ndarray:
        if not self._golden_checked:
            self.assert_golden_sample()
        means, _ = self._attribute_means_and_weights(prompt, answer)
        return means[attribute_indices(attributes)]

    __call__ = score

    def aggregate_score(self, prompt: str, answer: str) -> float:
        """Reconstruct the model's public scalar score from means and weights."""
        means, weights = self._attribute_means_and_weights(prompt, answer)
        return float(means @ weights)

    def assert_golden_sample(self, atol: float | None = None) -> np.ndarray:
        """Anchor the tokenizer/model path to the two public model-card values."""
        if atol is None:
            atol = 0.75 if self.load_in_8bit else 0.15
        observed = np.asarray(
            [
                self.aggregate_score(GOLDEN_SAMPLE["prompt"], response)
                for response in GOLDEN_SAMPLE["responses"]
            ],
            dtype=np.float64,
        )
        expected = np.asarray(GOLDEN_SAMPLE["expected_aggregate_scores"], dtype=np.float64)
        deviation = np.abs(observed - expected)
        if not np.all(deviation <= float(atol)):
            raise RuntimeError(
                "URM golden-sample check FAILED.\n"
                f"  expected: {expected.tolist()}\n"
                f"  observed: {observed.tolist()}\n"
                f"  max deviation: {deviation.max():.4f} > atol={atol}\n"
                f"  source: {GOLDEN_SAMPLE['source']}"
            )
        if not observed[1] > observed[0]:
            raise RuntimeError("URM golden ranking is reversed.")
        self._golden_checked = True
        print(
            f"[golden] URM anchored, max aggregate deviation "
            f"{deviation.max():.4f} <= {atol}."
        )
        return observed

    def describe(self) -> dict[str, Any]:
        return {
            "reward_model_name": self.reward_model_name,
            "revision": self.revision,
            "precision": "int8" if self.load_in_8bit else self.dtype,
            "batch_size": 1,
            "input_format": "tokenizer chat template (user, assistant)",
            "output": "five normal-distribution means; uncertainty parameters unused",
            "attribute_order": list(URM_ATTRIBUTES),
            "golden_sample_checked": self._golden_checked,
            "independence_role": "sensitivity only; trained directly on HelpSteer2",
        }

    def unload(self) -> None:
        """Release the large evaluator before another reward model is loaded."""
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

