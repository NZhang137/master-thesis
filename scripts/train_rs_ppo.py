"""
RS-faithful specialist training: theta_SFT -> N independent PPO runs (one per proxy reward)
============================================================================================

Follows Rame et al. 2023 (Rewarded Soups), Appendix D / Table 1, adapted TinyLlama-1.1B.
Betreuer-Leitplanke: Training ist Nebensache -> RS-Defaults 1:1, KEIN Tuning.

Pipeline (Handoff v7):
  Phase 1: SFT on HelpSteer2 -> theta_SFT  (shared init, RS's "Alpaca step")
  Phase 2: N independent PPO runs from theta_SFT, reward R_i from a HELD-OUT RM.
           A deliberately circular ArmoRM reward path exists only when
           --circular_armorm_acknowledged is passed. That path is invalid for
           RQ2 proxy validity and is meant only as an upper-bound experiment.
  Phase 3: extract delta_i = theta_i - theta_SFT -> R = D^T D  (notebook)

Integrity rules (hard):
  - ArmoRM is blocked by default. If used with explicit acknowledgement, the
    run must be reported as circular and RQ2 proxy-validity claims are retired.
  - Checkpointing by reward/KL trace, never by a later eval metric.
  - Equal-N analogue: identical PPO steps, prompt set, and seed schedule across axes.

RS Table-1 defaults (text-to-text):
  PPO (TRL), LoRA alpha=16 dropout=0.05, Adam lr=1.41e-5, batch 128,
  KL coeff 0.2 (dialog-task value), output length uniform 16..32 tokens, 1 epoch.

FIXES vs. the first draft (audit findings):
  [F1] CLI args for out_dir/batch_size/total_ppo_steps/n_prompts/4bit.
       The notebook previously mutated CFG in-process and then launched training
       via subprocess -> every mutation was silently lost and the adapters were
       written to the wrong directory. CFG is now settable from the command line
       AND via apply_overrides() for in-process calls.
  [F2] ArmoRM objective indices are resolved BY NAME against the model-card list
       and pinned with the authors' published golden sample. ArmoRM ships no id2label.
       (This is the exact bug class that produced the QRM verbosity == 0 disaster.)
  [F3] Batched reward scoring is validated against single-example scoring; the
       padding side is resolved empirically and falls back to batch_size=1.
       Attention masks are built from true lengths, not by comparing to pad_id
       (pad_token == eos_token would otherwise mask real <|eot_id|> tokens).
  [F4] `pad_token_id or eos_token_id` falsy bug fixed (pad_token_id == 0 is valid).
  [F5] Head sanity must run on ACTUAL policy generations, not on dataset responses:
       generate_responses() is exposed so the notebook can score the distribution
       the reward model will really see during PPO.
  [F6] Reward-plateau detection: the upper-bound argument requires a SHORT horizon
       (delta ~ eta * grad r). If the reward curve flattens, Assumption 1 is broken
       again and the Best-Case claim loses its basis. Logged as `reward_plateaued`.

Usage (one axis per invocation to keep runs independent):
  python train_rs_ppo.py --phase sft --out_dir results/.../rs_runs
  python train_rs_ppo.py --phase ppo --axis helpfulness \
      --reward_model RLHFlow/ArmoRM-Llama3-8B-v0.1 --circular_armorm_acknowledged \
      --out_dir results/.../rs_runs --batch_size 64 --total_ppo_steps 200
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.helpsteer2_utils import HELPSTEER2_ATTRIBUTES

# ----------------------------------------------------------------------------
# Configuration -- RS Table 1 defaults. Do not tune (Betreuer: Training = Nebensache).
# ----------------------------------------------------------------------------

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ARMORM_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
ATTRIBUTES = tuple(HELPSTEER2_ATTRIBUTES)
assert ATTRIBUTES == tuple(HELPSTEER2_ATTRIBUTES), (
    f"axis order drift: {ATTRIBUTES} vs {HELPSTEER2_ATTRIBUTES}"
)

# NOTE: no positional indices here on purpose. The head index is resolved BY NAME
# by NAME against src/armorm_objectives.py (the model card) and pinned by the
# published golden sample at load time -- never trust a hard-coded
# reward-head index (this is how QRM's verbosity head silently returned zeros).
from src.armorm_objectives import (  # single source of truth (model card)
    ARMORM_HELPSTEER_OBJECTIVE_NAMES,
    ARMORM_OBJECTIVES,
    GOLDEN_SAMPLE,
    assert_model_agrees,
    helpsteer_head_indices,
    to_helpsteer_scale,
)

CIRCULARITY_WARNING = (
    "CIRCULAR ArmoRM PPO acknowledged: PPO reward model and evaluation model "
    "are both ArmoRM. This retires RQ2 proxy-validity claims. Valid use is only "
    "upper-bound, R-minus, Wall-A/R2, and LMC diagnostics."
)

# >>> Held-out RMs (the non-circular path). Must NOT be ArmoRM.
HELD_OUT_RM = {
    "helpfulness": "<HF_ID_OF_HELD_OUT_RM_OR_HEAD>",
    "correctness": "<HF_ID_OF_HELD_OUT_RM_OR_HEAD>",
    "coherence": "<HF_ID_OF_HELD_OUT_RM_OR_HEAD>",
    "complexity": "<HF_ID_OF_HELD_OUT_RM_OR_HEAD>",
    "verbosity": "<HF_ID_OF_HELD_OUT_RM_OR_HEAD>",
}

CFG = dict(
    # --- LoRA (RS Table 1; r=8 kept byte-compatible with existing adapters) ---
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    # --- PPO (RS Table 1) ---
    learning_rate=1.41e-5,
    batch_size=64,          # RS: 128; halve for Colab VRAM. If A100-80GB: use 128.
    mini_batch_size=8,
    ppo_epochs=1,           # RS: 1-2; fixed at 1 for ALL axes (Equal-N analogue)
    init_kl_coef=0.2,       # RS: 0.2 for dialog-style tasks (0.05 only for summarization)
    # --- Generation (RS Table 1: output length uniform 16..32) ---
    output_min_len=16,
    output_max_len=32,
    # --- Equal-N analogue: identical across ALL axes, frozen before any run ---
    total_ppo_steps=200,    # pilot-scale; same number for every axis, never per-axis tuned
    prompt_seed=137,        # same prompt subsample for every axis
    train_seed=911,         # same train seed schedule for every axis
    n_prompts=2005,         # mirror the frozen N from the SFT recipe (ignored if full_epoch=True)
    full_epoch=False,       # if True: use entire HelpSteer2-train split, one pass per axis (RS-faithful)
    # --- Plateau detection (audit fix [F6]) ---
    plateau_window=50,      # steps used to estimate the tail slope of the reward curve
    plateau_slope_eps=0.02, # |slope| * window < eps * std(reward)  =>  plateaued
    # --- SFT phase ---
    sft_epochs=1,
    sft_lr=2e-5,
    sft_batch_size=16,
    # --- paths / reward model ---
    out_dir="rs_runs",
    armorm_model=ARMORM_MODEL,
    armorm_load_in_4bit=True,
    armorm_load_in_8bit=False,   # RS-faithful LLM.int8 (~8 GB); takes precedence over 4bit if True
    armorm_reward_batch_size=8,
)

# Keys that may be overridden from the CLI or from the notebook (audit fix [F1]).
OVERRIDABLE = (
    "out_dir", "batch_size", "mini_batch_size", "total_ppo_steps", "n_prompts",
    "armorm_model", "armorm_load_in_4bit", "armorm_load_in_8bit", "armorm_reward_batch_size",
    "learning_rate", "init_kl_coef", "ppo_epochs",
    "output_min_len", "output_max_len", "prompt_seed", "train_seed", "full_epoch",
    "plateau_window", "plateau_slope_eps",
)


def apply_overrides(**kwargs) -> dict:
    """Set CFG entries from the notebook (in-process) or the CLI. -- audit fix [F1]"""
    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in OVERRIDABLE:
            raise KeyError(f"{key!r} is not overridable; allowed: {OVERRIDABLE}")
        CFG[key] = value
    return CFG


# ----------------------------------------------------------------------------


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_helpsteer2_prompts(n: int | None, seed: int, split: str = "train"):
    """Load prompts from HelpSteer2.
    
    If n is None: use entire split (RS-faithful full epoch).
    If n is int: uniform subsample of size n (default short horizon).
    Both shuffle deterministically by seed for Equal-N (all axes see same permutation).
    """
    from datasets import load_dataset
    ds = load_dataset("nvidia/HelpSteer2", split=split)
    rng = np.random.default_rng(seed)
    size = len(ds) if n is None else min(n, len(ds))
    idx = rng.choice(len(ds), size=size, replace=False)
    return [ds[int(i)]["prompt"] for i in idx]


def is_armorm_model(model_id: str) -> bool:
    return "armorm" in model_id.lower() or model_id == ARMORM_MODEL


def check_reward_firewall(
    axis: str,
    reward_model_id: str,
    circular_armorm_acknowledged: bool = False,
) -> dict:
    """Return firewall metadata and fail unless circular ArmoRM is explicit."""
    if axis not in ATTRIBUTES:
        raise AssertionError(f"unknown axis {axis!r}; choose one of {ATTRIBUTES}")
    if not is_armorm_model(reward_model_id):
        return {
            "axis": axis,
            "reward_model": reward_model_id,
            "circularity_acknowledged": False,
            "warning": None,
        }
    if not circular_armorm_acknowledged:
        raise AssertionError(
            "FIREWALL: ArmoRM cannot be used as a PPO reward unless "
            "--circular_armorm_acknowledged is passed. This is circular and "
            "invalidates RQ2 proxy-validity claims."
        )
    warnings.warn(CIRCULARITY_WARNING, RuntimeWarning)
    print("WARNING:", CIRCULARITY_WARNING)
    return {
        "axis": axis,
        "reward_model": reward_model_id,
        "circularity_acknowledged": True,
        "warning": CIRCULARITY_WARNING,
        "retired_research_questions": ["RQ2 (proxy validity)"],
        "valid_claims": ["upper bound", "R-minus", "wall A (R2)", "LMC"],
        "invalid_claims": ["proxy validity", "generalization beyond this reward model"],
    }


# ----------------------------------------------------------------------------
# Frozen ArmoRM head scorer -- verified indices, validated batching
# ----------------------------------------------------------------------------

class ArmoRMHeadScorer:
    """Frozen ArmoRM scorer returning ONE verified HelpSteer2 head per text pair.

    Audit fixes baked in:
      [F2] objective index resolved BY NAME against the model-card objective list
           and pinned with the authors' published golden sample. ArmoRM ships NO
           id2label (it is literally {0: 'LABEL_0'}), so the old id2label route
           was itself an unverified assumption about an external artifact.
      [F3] padding side resolved empirically; batched == single is asserted
      [F4] no `pad_token_id or eos_token_id` falsy bug
    """

    def __init__(self, axis: str, model_id: str | None = None):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        model_id = model_id or CFG["armorm_model"]
        if axis not in ARMORM_HELPSTEER_OBJECTIVE_NAMES:
            raise ValueError(f"No ArmoRM HelpSteer head configured for {axis!r}.")
        self.axis = axis
        self.model_id = model_id
        self.objective_name = ARMORM_HELPSTEER_OBJECTIVE_NAMES[axis]

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # [F5] The pad id is resolved AFTER the model is loaded, from the MODEL's config --
        # not from the tokenizer. See the long note at self._resolve_pad_id().

        quantization_config = None
        if CFG.get("armorm_load_in_8bit", False):
            # LLM.int8 -- ~8 GB, far more faithful than 4-bit nf4, and what RS itself used.
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif CFG.get("armorm_load_in_4bit", True):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # ArmoRM's config.json carries "num_objectives": 19 but an EMPTY id2label
        # ({"0": "LABEL_0"}). The count is therefore the only thing the model itself
        # can confirm; the names come from the model card and the mapping is pinned
        # empirically by verify_against_model_card() below.
        assert_model_agrees(self.model.config)
        self.objective_index = self._name_to_index(self.objective_name)
        self.pad_id = self._resolve_pad_id()
        print(f"[armorm] axis={self.axis} -> {self.objective_name!r} at index "
              f"{self.objective_index} (num_objectives={self.model.config.num_objectives}, "
              f"pad_id={self.pad_id})")

        # ArmoRM reads the reward at the token BEFORE the first pad id, so right padding
        # is the only side that keeps position_ids unshifted for the real tokens.
        self.padding_side = "right"
        self.batching_validated = False

        # [F2] The head order is pinned against the AUTHORS' published numbers before this
        # scorer is allowed to return a single reward. This cannot be skipped: it is the
        # only expected value in the pipeline that does not come from our own code.
        self.golden = self.verify_against_model_card()

    # ---- [F2] index verification -------------------------------------------

    def _resolve_pad_id(self) -> int:
        """Return the pad id the MODEL looks for -- never the tokenizer's.

        ArmoRM's modeling_custom.forward locates the reward position with

            sequence_lengths = eq(input_ids, config.pad_token_id).argmax(-1) - 1   (mod L)

        i.e. it searches for the FIRST occurrence of *config.pad_token_id* (128256, a
        dedicated pad token appended to the vocabulary: vocab_size is 128257) and reads the
        hidden state one position earlier.

        The Llama-3 tokenizer ships NO pad token. Padding with eos (128001) therefore makes
        eq(...) match nothing, argmax return 0, sequence_lengths become -1 % L = L-1, and the
        reward gets read at the LAST slot of the padded row -- under right padding that is a
        PAD token. Worse, if eos appears inside the text (it does: the chat template emits
        <|eot_id|>), argmax lands in the middle of the prompt. This is exactly why
        validate_batching() reported that neither padding side reproduces the single-example
        scores.

        Using the model's own pad id makes argmax land on the first real pad, so L-1 is the
        last REAL token, and right padding leaves position_ids unshifted.
        """
        model_pad = getattr(self.model.config, "pad_token_id", None)
        if model_pad is None:
            raise RuntimeError(
                "ArmoRM config has no pad_token_id. Its forward() cannot locate the reward "
                "position for batch sizes > 1; refusing to guess a pad token."
            )
        pad_id = int(model_pad)
        vocab_size = int(getattr(self.model.config, "vocab_size", 0))
        if vocab_size and not 0 <= pad_id < vocab_size:
            raise RuntimeError(f"pad_token_id={pad_id} outside vocab_size={vocab_size}.")
        tok_pad = self.tokenizer.pad_token_id
        if tok_pad is not None and int(tok_pad) != pad_id:
            print(f"[armorm] NOTE: tokenizer pad id {tok_pad} != model pad id {pad_id}; "
                  "padding with the MODEL's id, which is what forward() searches for.")
        # Keep the tokenizer consistent so any apply_chat_template(padding=True) agrees.
        self.tokenizer.pad_token_id = pad_id
        return pad_id

    def _name_to_index(self, name: str) -> int:
        """Resolve an objective name to its head index against the model-card list."""
        if name not in ARMORM_OBJECTIVES:
            raise RuntimeError(
                f"Objective {name!r} is not in the ArmoRM objective list. "
                f"Known: {list(ARMORM_OBJECTIVES)}"
            )
        return ARMORM_OBJECTIVES.index(name)

    def helpsteer_indices(self) -> list[int]:
        """Indices of the five HelpSteer2 heads, in ATTRIBUTES order, resolved BY NAME."""
        return helpsteer_head_indices(ATTRIBUTES)

    def verify_against_model_card(self) -> dict:
        """Pin the head order with the AUTHORS' published numbers, not with our own.

        The model card prints, for one fixed prompt/response, the first five raw heads
        mapped onto the original HelpSteer scale (rewards[:5] * 5 - 0.5). Reproducing
        those five numbers is the only check in this pipeline whose expected value does
        not come from our own code -- it cannot be satisfied by a self-referential test.

        A shuffled head order would move complexity/verbosity (low: ~1.3) into the
        helpfulness/correctness slots (high: ~2.8) and blow the tolerance immediately.
        """
        encoded = [self._encode(GOLDEN_SAMPLE["prompt"], GOLDEN_SAMPLE["response"])]
        rewards = self._forward_rewards(encoded, "right")
        assert_model_agrees(self.model.config, rewards_last_dim=int(rewards.shape[-1]))

        idx = helpsteer_head_indices(ATTRIBUTES)
        observed = to_helpsteer_scale(rewards[0, idx].numpy().astype(float))
        expected = np.asarray(GOLDEN_SAMPLE["expected_helpsteer_rewards"], dtype=float)
        max_abs = float(np.max(np.abs(observed - expected)))

        result = {
            "axes": list(ATTRIBUTES),
            "head_indices": [int(i) for i in idx],
            "observed_helpsteer_scale": [float(v) for v in observed],
            "expected_helpsteer_scale": [float(v) for v in expected],
            "helpsteer_ground_truth": list(GOLDEN_SAMPLE["helpsteer_ground_truth"]),
            "max_abs_error": max_abs,
            "atol": float(GOLDEN_SAMPLE["atol"]),
            "source": GOLDEN_SAMPLE["source"],
            "passed": max_abs <= float(GOLDEN_SAMPLE["atol"]),
        }
        if not result["passed"]:
            raise RuntimeError(
                "ArmoRM golden sample MISMATCH -- the head order or the loading path is "
                f"wrong. observed={result['observed_helpsteer_scale']} "
                f"expected={result['expected_helpsteer_scale']} "
                f"max_abs={max_abs:.4f} > atol={GOLDEN_SAMPLE['atol']}. "
                "Refusing to score anything with an unverified head mapping."
            )
        print(f"[armorm] golden sample OK (max_abs={max_abs:.4f}); head order pinned "
              f"against {GOLDEN_SAMPLE['source']}")
        return result

    # ---- scoring ------------------------------------------------------------

    def _encode(self, prompt: str, response: str) -> torch.Tensor:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        else:
            text = f"Human: {prompt}\n\nAssistant: {response}"
            ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=4096).input_ids
        return ids.squeeze(0)

    def _forward_rewards(self, encoded: list[torch.Tensor], padding_side: str) -> torch.Tensor:
        """Pad a list of id-tensors and return the full reward matrix [B, n_obj].

        [F3] The attention mask is built from TRUE LENGTHS. Building it as
        `(padded != pad_id)` is wrong whenever pad_token == eos_token, because the
        Llama-3 chat template contains real <|eot_id|> tokens *inside* the
        sequence -- those would be masked away.
        """
        lengths = [int(e.shape[0]) for e in encoded]
        max_len = max(lengths)
        batch = len(encoded)
        input_ids = torch.full((batch, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        for k, (ids, length) in enumerate(zip(encoded, lengths)):
            if padding_side == "right":
                input_ids[k, :length] = ids
                attention_mask[k, :length] = 1
            else:  # left padding
                input_ids[k, max_len - length:] = ids
                attention_mask[k, max_len - length:] = 1

        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
        rewards = getattr(outputs, "rewards", None)
        if rewards is None:
            raise RuntimeError("ArmoRM output has no .rewards tensor.")
        tensor = torch.as_tensor(rewards).detach().float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.cpu()

    def validate_batching(self, pairs: list[tuple[str, str]], atol: float = 1e-3) -> dict:
        """[F3] Resolve the padding side empirically and prove batched == single.

        ArmoRM pools at the last non-padding token. Whether right- or left-padding
        reads the correct token is an empirical property of the custom modelling
        code, not something to assume. We score a probe batch one-by-one (which is
        unambiguous) and compare against both padding sides. If neither matches we
        force batch_size=1 rather than silently produce garbage rewards.
        """
        probe = pairs[: min(8, len(pairs))]
        assert len(probe) >= 2, "need at least 2 probe pairs to validate batching"
        encoded = [self._encode(p, r) for p, r in probe]

        singles = torch.cat([self._forward_rewards([e], "right") for e in encoded], dim=0)

        report: dict = {"atol": atol, "n_probe": len(probe)}
        for side in ("right", "left"):
            batched = self._forward_rewards(encoded, side)
            diff = float((batched - singles).abs().max())
            report[f"max_abs_diff_{side}"] = diff
            if diff <= atol:
                self.padding_side = side
                self.batching_validated = True
                report["resolved_padding_side"] = side
                report["fallback_to_batch_size_1"] = False
                print(f"[armorm] batched scoring validated with {side} padding "
                      f"(max|diff|={diff:.2e})")
                return report

        # Neither side reproduces single-example scores -> do not gamble.
        CFG["armorm_reward_batch_size"] = 1
        self.padding_side = "right"
        self.batching_validated = True   # batch of 1 is trivially valid
        report["resolved_padding_side"] = None
        report["fallback_to_batch_size_1"] = True
        warnings.warn(
            "ArmoRM batched scoring does not reproduce single-example scores with "
            "either padding side. Falling back to batch_size=1 (slower but correct).",
            RuntimeWarning,
        )
        print("[armorm] WARNING: falling back to reward batch_size=1")
        return report

    def _score_matrix(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length.")
        if not self.batching_validated:
            raise RuntimeError(
                "Refusing to score before validate_batching() has run. "
                "Call scorer.validate_batching(probe_pairs) first."
            )
        chunks = []
        bs = int(CFG.get("armorm_reward_batch_size", 8))
        for start in range(0, len(prompts), bs):
            stop = min(start + bs, len(prompts))
            encoded = [self._encode(p, r)
                       for p, r in zip(prompts[start:stop], responses[start:stop])]
            chunks.append(self._forward_rewards(encoded, self.padding_side).numpy())
        return np.concatenate(chunks, axis=0).astype(np.float64)

    def score_batch(self, prompts: list[str], responses: list[str]) -> list[torch.Tensor]:
        """Scalar reward for THIS axis (PPO signal)."""
        matrix = self._score_matrix(prompts, responses)
        column = matrix[:, self.objective_index]
        return [torch.tensor(float(v), dtype=torch.float32) for v in column]

    def score_all_heads(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        """Full [N, 5] HelpSteer2 head matrix (head sanity / Wall-A reward collection)."""
        matrix = self._score_matrix(prompts, responses)
        return matrix[:, self.helpsteer_indices()]


# ----------------------------------------------------------------------------
# [F5] Generation helper -- head sanity must see the PPO distribution
# ----------------------------------------------------------------------------

def generate_responses(model_path: str, prompts: list[str], seed: int,
                       batch_size: int = 8,
                       max_new_tokens: int | None = None) -> list[str]:
    """Generate with the SAME config PPO uses (uniform 16..32 new tokens).

    Head sanity on HelpSteer2 *reference* responses is the wrong distribution: the
    reward model will only ever see short TinyLlama rollouts. A head can vary nicely
    on human text and collapse on 16-32-token model output -- that is precisely the
    QRM verbosity failure, one level deeper.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_all_seeds(seed)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # correct side for decoder-only generation
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    responses: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        n_new = max_new_tokens or random.randint(CFG["output_min_len"],
                                                 CFG["output_max_len"])
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=n_new, do_sample=True,
                                 top_k=0, top_p=1.0, pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        responses.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"[gen] {min(start + batch_size, len(prompts))}/{len(prompts)}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


# ----------------------------------------------------------------------------
# [F6] Plateau detection -- the upper-bound argument needs a SHORT horizon
# ----------------------------------------------------------------------------

def detect_reward_plateau(log: list[dict]) -> dict:
    """If the reward curve flattens, PPO has converged and delta != eta * grad r.

    That breaks Assumption 1 again and with it the Best-Case ("upper bound")
    reading of this experiment. Recorded so the thesis claim can be checked
    instead of assumed.
    """
    window = int(CFG["plateau_window"])
    rewards = np.asarray([row["mean_reward"] for row in log], dtype=np.float64)
    if len(rewards) < max(10, window):
        return {"reward_plateaued": None,
                "reason": "too few steps to judge",
                "n_steps": int(len(rewards)),
                "interpretation": (
                    f"too few steps ({len(rewards)} < {max(10, window)}) to judge a plateau; "
                    "not meaningful for a short probe run.")}
    tail = rewards[-window:]
    steps = np.arange(len(tail), dtype=np.float64)
    slope = float(np.polyfit(steps, tail, 1)[0])
    scale = float(rewards.std())
    if scale == 0.0:
        scale = 1.0
    drift = abs(slope) * window
    threshold = CFG["plateau_slope_eps"] * scale
    plateaued = bool(drift < threshold)
    return {
        "reward_plateaued": plateaued,
        "tail_window": window,
        "tail_slope_per_step": slope,
        "tail_drift_over_window": drift,
        "reward_std_full_run": scale,
        "threshold": threshold,
        "interpretation": (
            "PLATEAUED -> converged; Assumption 1 (delta ~ eta*grad r) is broken and "
            "the upper-bound / Best-Case reading no longer holds."
            if plateaued else
            "still ascending -> short-horizon regime intact; upper-bound reading holds."
        ),
    }


# ----------------------------------------------------------------------------
# Phase 1: SFT base  (RS: pretrain -> SFT -> PPO; SFT is the shared init)
# ----------------------------------------------------------------------------

def run_sft():
    """TinyLlama + SFT on HelpSteer2 responses -> theta_SFT (shared init).

    Deliberately minimal: generic instruction-SFT on HelpSteer2 (prompt -> response),
    NO attribute filtering here -- theta_SFT must be axis-neutral, specialization
    happens only in the PPO phase (RS-faithful).
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from peft import LoraConfig

    set_all_seeds(CFG["train_seed"])
    out = Path(CFG["out_dir"]) / "theta_sft"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("nvidia/HelpSteer2", split="train")

    def fmt(ex):
        msgs = [{"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["response"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    peft_cfg = LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target_modules"], task_type="CAUSAL_LM")

    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=CFG["sft_epochs"],
        per_device_train_batch_size=CFG["sft_batch_size"],
        learning_rate=CFG["sft_lr"], bf16=True,
        logging_steps=25, save_strategy="epoch", seed=CFG["train_seed"],
        report_to="none")

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg,
                         dataset_text_field="text", max_seq_length=1024)
    trainer.train()

    # merge LoRA into the base so theta_SFT is a plain checkpoint = the single
    # shared init all PPO runs start from (and the KL reference).
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out / "merged"))
    tok.save_pretrained(str(out / "merged"))
    print(f"[sft] theta_SFT saved to {out / 'merged'}")
    return str(out / "merged")


# ----------------------------------------------------------------------------
# Phase 2: one independent PPO run for ONE axis (call once per axis)
# ----------------------------------------------------------------------------

def run_ppo(
    axis: str,
    reward_model_id: str | None = None,
    circular_armorm_acknowledged: bool = False,
):
    from transformers import AutoTokenizer, pipeline
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    from peft import LoraConfig

    assert axis in ATTRIBUTES, f"unknown axis {axis}; choose one of {ATTRIBUTES}"
    rm_id = reward_model_id or HELD_OUT_RM.get(axis)
    assert rm_id, f"no reward model configured for axis {axis}"
    assert "<HF_ID" not in rm_id, "Phase-0 decision missing: set the reward model id"
    firewall = check_reward_firewall(
        axis, rm_id, circular_armorm_acknowledged=circular_armorm_acknowledged)

    set_all_seeds(CFG["train_seed"])   # identical seed schedule across axes
    sft_path = Path(CFG["out_dir"]) / "theta_sft" / "merged"
    assert sft_path.exists(), (
        f"theta_SFT not found at {sft_path}. Run --phase sft first with the SAME "
        f"--out_dir. (This is the path bug that used to bite after five PPO runs.)"
    )
    out = Path(CFG["out_dir"]) / f"ppo_{axis}"
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(sft_path))
    tok.pad_token = tok.eos_token

    # policy = theta_SFT + fresh LoRA; TRL keeps a frozen reference copy of
    # theta_SFT internally -> KL penalty is measured against theta_SFT (RS-faithful)
    peft_cfg = LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target_modules"], task_type="CAUSAL_LM")
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(sft_path), peft_config=peft_cfg,
        torch_dtype=torch.bfloat16)

    ppo_cfg = PPOConfig(
        learning_rate=CFG["learning_rate"],
        batch_size=CFG["batch_size"],
        mini_batch_size=CFG["mini_batch_size"],
        ppo_epochs=CFG["ppo_epochs"],
        init_kl_coef=CFG["init_kl_coef"],
        seed=CFG["train_seed"])
    trainer = PPOTrainer(config=ppo_cfg, model=policy, tokenizer=tok)

    # Load prompts: full split (full_epoch=True, RS-faithful) or subsample (full_epoch=False, short horizon)
    if CFG.get("full_epoch"):
        prompts = load_helpsteer2_prompts(None, CFG["prompt_seed"])  # entire split
        n_steps = len(prompts) // CFG["batch_size"]
        print(f"[ppo:{axis}] FULL EPOCH mode: {len(prompts)} prompts -> {n_steps} steps "
              f"(batch {CFG['batch_size']}, remainder {len(prompts) % CFG['batch_size']} dropped)")
    else:
        prompts = load_helpsteer2_prompts(CFG["n_prompts"], CFG["prompt_seed"])  # subsample
        n_steps = CFG["total_ppo_steps"]

    # Reward pipeline. In acknowledged circular mode this is a frozen ArmoRM head.
    batching_report = None
    if is_armorm_model(rm_id):
        reward_pipe = ArmoRMHeadScorer(axis=axis, model_id=rm_id)
        # [F3] prove batched == single BEFORE a single PPO step is taken.
        probe_pairs = [(p, "This is a short probe response used to validate the scorer.")
                       for p in prompts[:8]]
        batching_report = reward_pipe.validate_batching(probe_pairs)
    else:
        reward_pipe = pipeline("text-classification", model=rm_id,
                               device_map="auto", torch_dtype=torch.bfloat16)

    gen_kwargs = dict(do_sample=True, top_k=0, top_p=1.0,
                      pad_token_id=tok.eos_token_id)

    log, step = [], 0
    for step in range(1, n_steps + 1):
        if CFG.get("full_epoch"):
            # Sequential indexing: each prompt seen exactly once per epoch
            s = (step - 1) * CFG["batch_size"]
            e = min(s + CFG["batch_size"], len(prompts))
            batch_prompts = prompts[s:e]
            if len(batch_prompts) < CFG["batch_size"]:
                break  # incomplete final batch, skip it
        else:
            # Sampling mode: random sample from subsample
            batch_prompts = random.sample(prompts, CFG["batch_size"])
        queries = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in batch_prompts]
        q_tensors = [tok(q, return_tensors="pt").input_ids.squeeze(0).to(trainer.accelerator.device)
                     for q in queries]

        # RS Table 1: output length sampled uniformly in [16, 32]
        r_tensors = [trainer.generate(
            q, max_new_tokens=random.randint(CFG["output_min_len"],
                                             CFG["output_max_len"]),
            **gen_kwargs).squeeze(0)[q.shape[0]:] for q in q_tensors]
        responses = tok.batch_decode(r_tensors, skip_special_tokens=True)

        if is_armorm_model(rm_id):
            rewards = reward_pipe.score_batch(batch_prompts, responses)
        else:
            texts = [p + "\n" + r for p, r in zip(batch_prompts, responses)]
            scores = reward_pipe(texts, truncation=True, max_length=1024)
            rewards = [torch.tensor(s["score"], dtype=torch.float32) for s in scores]

        stats = trainer.step(q_tensors, r_tensors, rewards)
        values = [float(r.item()) for r in rewards]
        log.append({"step": step,
                    "mean_reward": float(np.mean(values)),
                    "reward_std": float(np.std(values)),
                    "kl": float(stats.get("objective/kl", float("nan")))})
        if step % 10 == 0:
            print(f"[ppo:{axis}] step {step}/{n_steps} "
                  f"reward={log[-1]['mean_reward']:.4f} kl={log[-1]['kl']:.3f}")

    # [F6] horizon diagnostic -- decides whether the upper-bound reading survives
    plateau = detect_reward_plateau(log)
    print(f"[ppo:{axis}] plateau check: {plateau['interpretation']}")

    # checkpoint selection: end-of-run LoRA adapter, judged by the reward/KL trace only.
    # The TRL wrapper also owns a PPO value head; it is infrastructure, not delta_i.
    adapter_dir = out / "adapter"
    policy.pretrained_model.save_pretrained(str(adapter_dir))
    if hasattr(policy, "v_head"):
        torch.save(policy.v_head.state_dict(), out / "value_head.pt")

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        adapter_state = load_file(str(safetensors_path))
    elif bin_path.exists():
        adapter_state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")
    bad = [k for k in adapter_state if "lora_" not in k]
    assert not bad, f"non-LoRA keys in adapter: {bad[:5]}"
    (out / "ppo_log.json").write_text(json.dumps(
        {
            "axis": axis,
            "rm": rm_id,
            "config": CFG,
            "firewall": firewall,
            "armorm_batching_report": batching_report,
            "armorm_objective_index": getattr(reward_pipe, "objective_index", None),
            "armorm_objective_name": getattr(reward_pipe, "objective_name", None),
            "plateau": plateau,
            "log": log,
        },
        indent=2, default=str), encoding="utf-8")
    print(f"[ppo:{axis}] adapter saved to {out / 'adapter'}; log written.")
    return {"adapter": str(out / "adapter"), "plateau": plateau,
            "batching": batching_report}


# ----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sft", "ppo"], required=True)
    ap.add_argument("--axis", default=None, help="required for --phase ppo")
    ap.add_argument("--reward_model", default=None,
                    help="Reward model id. ArmoRM only with --circular_armorm_acknowledged.")
    ap.add_argument("--circular_armorm_acknowledged", action="store_true",
                    help="Explicitly acknowledge circular PPO against ArmoRM.")
    # [F1] overrides that used to be silently lost across the subprocess boundary
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--mini_batch_size", type=int, default=None)
    ap.add_argument("--total_ppo_steps", type=int, default=None)
    ap.add_argument("--n_prompts", type=int, default=None)
    ap.add_argument("--armorm_model", default=None)
    ap.add_argument("--armorm_reward_batch_size", type=int, default=None)
    ap.add_argument("--armorm_load_in_4bit", type=lambda s: s.lower() == "true",
                    default=None)
    ap.add_argument("--armorm_load_in_8bit", type=lambda s: s.lower() == "true",
                    default=None, help="LLM.int8 (~8 GB, RS-faithful); overrides 4bit if true.")
    ap.add_argument("--full_epoch", type=lambda s: s.lower() == "true", default=None,
                    help="If true: one full pass over HelpSteer2-train (RS-faithful). "
                         "If false: sample n_prompts for total_ppo_steps (short horizon).")
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    apply_overrides(
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        total_ppo_steps=args.total_ppo_steps,
        n_prompts=args.n_prompts,
        armorm_model=args.armorm_model,
        armorm_reward_batch_size=args.armorm_reward_batch_size,
        armorm_load_in_4bit=args.armorm_load_in_4bit,
        armorm_load_in_8bit=args.armorm_load_in_8bit,
        full_epoch=args.full_epoch,
    )
    epoch_str = "full_epoch" if CFG.get("full_epoch") else f"{CFG['n_prompts']} sampled prompts"
    _prec = ("8bit" if CFG.get("armorm_load_in_8bit")
             else "4bit" if CFG.get("armorm_load_in_4bit") else "bf16")
    print(f"[cfg] out_dir={CFG['out_dir']} batch_size={CFG['batch_size']} "
          f"mode={epoch_str} armorm_precision={_prec}")
    if args.phase == "sft":
        run_sft()
    else:
        assert args.axis, "--axis required for ppo phase"
        run_ppo(args.axis,
                reward_model_id=args.reward_model,
                circular_armorm_acknowledged=args.circular_armorm_acknowledged)
