"""Multi-head ArmoRM scoring for coefficient-method evaluation (NB10 Phase B).

Why this module exists
----------------------
`RewardMonitor._score_answer` in `src/tinyllama_training_utils.py` already scores
one (prompt, answer) pair with ArmoRM, but returns a SINGLE head, selected by
`self.reward_objective_index`. NB10 needs all `m` heads at once: one merged model
is expensive to build, so every merge point must yield the full reward vector in
one forward pass.

This module does not introduce a fourth head-name vocabulary. It resolves head
positions exclusively through `src.armorm_objectives.helpsteer_head_indices`,
which is the only mapping anchored by an external golden sample.

Quantization
------------
`dtype="bfloat16"` is the default because the model card's published numbers are
bf16, which lets `assert_golden_sample` run at a tight tolerance, and because
`RewardMonitor` loads bf16 during training - matching it keeps NB10's rewards on
the same scale as the training curves. `load_in_8bit=True` halves the memory at
the cost of speed and small shifts in the regression heads. Whatever
`train_rs_ppo.py` used for the PPO reward signal is what NB10 should mirror; the
resolved choice is returned by `describe()` so it can be frozen into the
pre-registration.

Batching
--------
`batch_size=1` only. ArmoRM's batched outputs are known not to reproduce its
single-example outputs in this project, and `RewardMonitor` rejects any other
value for the same reason. Do not add batching without first measuring the
deviation against this path and recording it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.armorm_objectives import (
    GOLDEN_SAMPLE,
    assert_model_agrees,
    helpsteer_head_indices,
    to_helpsteer_scale,
)


DEFAULT_REWARD_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"


class ArmoRMScorer:
    """Score (prompt, answer) pairs on several ArmoRM heads at once."""

    def __init__(
        self,
        reward_model_name: str = DEFAULT_REWARD_MODEL,
        *,
        dtype: str = "bfloat16",
        load_in_8bit: bool = False,
        max_length: int = 4096,
    ) -> None:
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported dtype {dtype!r} for ArmoRM.")
        self.reward_model_name = reward_model_name
        self.dtype = dtype
        self.load_in_8bit = bool(load_in_8bit)
        self.max_length = int(max_length)
        self._model: Any = None
        self._tokenizer: Any = None
        self._golden_checked = False

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.reward_model_name, trust_remote_code=True
        )
        if getattr(self._tokenizer, "chat_template", None) is None:
            raise RuntimeError(
                "ArmoRM tokenizer exposes no chat template. The golden sample was "
                "produced with apply_chat_template; scoring in any other format "
                "would silently change the reward scale."
            )

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            # LLM.int8 keeps the non-quantized parts in a compute dtype. The training
            # scorer loads bf16, so this branch must set it too - otherwise "8-bit"
            # in NB10 and "8-bit" in training are not the same numerical path.
            load_kwargs["torch_dtype"] = getattr(torch, self.dtype)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = getattr(torch, self.dtype)
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"

        model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_name, **load_kwargs
        )
        model.requires_grad_(False)
        model.eval()
        self._model = model
        assert_model_agrees(model.config)

    # -- core ---------------------------------------------------------------

    def _raw_rewards(self, prompt: str, answer: str) -> np.ndarray:
        """Return all 19 raw ArmoRM heads for one (prompt, answer) pair."""
        import torch

        self._load()
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            outputs = self._model(input_ids=input_ids.to(device))

        rewards = getattr(outputs, "rewards", None)
        if rewards is None:
            raise ValueError(
                "The reward model returned no multi-objective reward vector; "
                "outputs.rewards is required for multi-head scoring."
            )
        tensor = torch.as_tensor(rewards).detach().float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, ArmoRM returned {tensor.shape[0]} rows. "
                "Batched scoring is not equivalent to single scoring here."
            )
        assert_model_agrees(self._model.config, tensor.shape[1])
        return tensor[0].cpu().numpy().astype(np.float64)

    def score(
        self,
        prompt: str,
        answer: str,
        attributes: Sequence[str],
    ) -> np.ndarray:
        """Return one reward per attribute, in the order of ``attributes``."""
        if not self._golden_checked:
            self.assert_golden_sample()
        indices = helpsteer_head_indices(list(attributes))
        raw = self._raw_rewards(prompt, answer)
        return raw[indices]

    __call__ = score

    # -- verification -------------------------------------------------------

    def assert_golden_sample(self, atol: float | None = None) -> np.ndarray:
        """Check the loaded model against the authors' published numbers.

        This is the only assertion in the pipeline that a self-referential check
        cannot satisfy: the expected values come from the model card, not from
        this repository. It touches ArmoRM but evaluates no merge point, so it
        does not consume the one-shot Phase B budget and may be re-run freely.
        """
        raw = self._raw_rewards(GOLDEN_SAMPLE["prompt"], GOLDEN_SAMPLE["response"])
        observed = to_helpsteer_scale(raw[:5])
        expected = np.asarray(GOLDEN_SAMPLE["expected_helpsteer_rewards"], dtype=np.float64)
        if atol is None:
            # The published numbers are bf16. Only int8 needs the loose default.
            atol = float(GOLDEN_SAMPLE["atol"]) if self.load_in_8bit else 0.05
        deviation = np.abs(observed - expected)
        if not np.all(deviation <= atol):
            raise RuntimeError(
                "ArmoRM golden-sample check FAILED.\n"
                f"  expected (model card): {np.round(expected, 4).tolist()}\n"
                f"  observed:              {np.round(observed, 4).tolist()}\n"
                f"  max deviation:         {deviation.max():.4f} > atol={atol}\n"
                f"  source: {GOLDEN_SAMPLE['source']}\n"
                "Do NOT score any merge point until this passes: the head mapping "
                "or the quantization is wrong."
            )
        self._golden_checked = True
        print(
            f"[golden] ArmoRM anchored, max deviation {deviation.max():.4f} "
            f"<= atol {atol} ({self.describe()['precision']})."
        )
        return observed

    def compare_against(
        self,
        other_score_fn: Any,
        prompt: str,
        answer: str,
        attributes: Sequence[str],
    ) -> dict[str, Any]:
        """Compare this scorer against another on one (prompt, answer) pair.

        `other_score_fn(prompt, answer, attributes)` is expected to return one
        value per attribute - typically the NB08 training scorer. Reporting the
        maximum deviation turns "same precision as training" from a claim in a
        comment into a measured number that belongs in the pre-registration.
        """
        mine = np.asarray(self.score(prompt, answer, attributes), dtype=np.float64)
        theirs = np.asarray(other_score_fn(prompt, answer, attributes), dtype=np.float64)
        if mine.shape != theirs.shape:
            raise ValueError(f"Shape mismatch: {mine.shape} vs {theirs.shape}.")
        deviation = np.abs(mine - theirs)
        return {
            "attributes": list(attributes),
            "nb10_scorer": mine.tolist(),
            "reference_scorer": theirs.tolist(),
            "max_abs_deviation": float(deviation.max()),
            "precision": self.describe()["precision"],
        }

    def describe(self) -> dict[str, Any]:
        """Return the resolved load settings, for the pre-registration record."""
        return {
            "reward_model_name": self.reward_model_name,
            "precision": "int8" if self.load_in_8bit else self.dtype,
            "batch_size": 1,
            "input_format": "apply_chat_template(user, assistant)",
            "max_length": self.max_length,
            "golden_sample_checked": self._golden_checked,
        }


def make_score_prompt_answer(
    reward_model_name: str = DEFAULT_REWARD_MODEL,
    *,
    dtype: str = "bfloat16",
    load_in_8bit: bool = False,
) -> tuple[Any, ArmoRMScorer]:
    """Return the `score_prompt_answer(prompt, answer, attributes)` NB10 expects.

    The scorer instance is returned alongside it so the notebook can call
    `assert_golden_sample()` explicitly before the first merge point and can
    freeze `describe()` into the pre-registration.
    """
    scorer = ArmoRMScorer(reward_model_name, dtype=dtype, load_in_8bit=load_in_8bit)

    def score_prompt_answer(
        prompt: str, answer: str, attributes: Sequence[str]
    ) -> np.ndarray:
        return scorer.score(prompt, answer, attributes)

    return score_prompt_answer, scorer
