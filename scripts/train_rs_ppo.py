"""
RS-faithful specialist training: theta_SFT -> N independent PPO runs (one per proxy reward)
============================================================================================

Follows Rame et al. 2023 (Rewarded Soups), Appendix D / Table 1, adapted TinyLlama-1.1B.
Betreuer-Leitplanke: Training ist Nebensache -> RS-Defaults 1:1, KEIN Tuning.

Pipeline (Handoff v6 §1.4):
  Phase 1: SFT on HelpSteer2 -> theta_SFT  (shared init, RS's "Alpaca step")
  Phase 2: N independent PPO runs from theta_SFT, reward R_i from a HELD-OUT RM.
           A deliberately circular ArmoRM reward path exists only when
           --circular_armorm_acknowledged is passed. That path is invalid for
           RQ2 proxy validity and is meant only as an upper-bound experiment.
  Phase 3: extract delta_i = theta_i - theta_SFT -> R = D^T D  (separate notebook/cells)

Integrity rules (hard):
  - ArmoRM is blocked by default. If used with explicit acknowledgement, the
    run must be reported as circular and RQ2 proxy-validity claims are retired.
  - Checkpointing by held-out RM reward / KL, never by ArmoRM.
  - Equal-N analogue: identical PPO steps, prompt set, and seed schedule across axes.

RS Table-1 defaults (text-to-text):
  PPO (TRL), LoRA alpha=16 dropout=0.05, Adam lr=1.41e-5, batch 128,
  KL coeff 0.2 (dialog-task value), output length uniform 16..32 tokens, 1 epoch.

Usage (Colab, one axis per invocation to keep runs independent):
  python train_rs_ppo.py --phase sft
  python train_rs_ppo.py --phase ppo --axis helpfulness
  python train_rs_ppo.py --phase ppo --axis <axis2>          # pilot: m=2 first!
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Configuration -- RS Table 1 defaults. Do not tune (Betreuer: Training = Nebensache).
# ----------------------------------------------------------------------------

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ARMORM_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
ARMORM_HELPSTEER_OBJECTIVES = {
    "helpfulness": (0, "helpsteer-helpfulness"),
    "correctness": (1, "helpsteer-correctness"),
    "coherence": (2, "helpsteer-coherence"),
    "complexity": (3, "helpsteer-complexity"),
    "verbosity": (4, "helpsteer-verbosity"),
}
CIRCULARITY_WARNING = (
    "CIRCULAR ArmoRM PPO acknowledged: PPO reward model and evaluation model "
    "are both ArmoRM. This retires RQ2 proxy-validity claims. Valid use is only "
    "upper-bound, R-minus, Wall-A/R2, and LMC diagnostics."
)

# >>> DECISION REQUIRED (Phase 0, Step 1): held-out RM, one per axis or multi-head.
# Must NOT be ArmoRM. Fill in after the pre-check (held-out-RM correlation on HelpSteer2).
# RS analogue: different open-source RMs from HF, used as black-box scoring pipelines.
HELD_OUT_RM = {
    "helpfulness": "<HF_ID_OF_HELD_OUT_RM_OR_HEAD>",   # e.g. an OASST-style RM
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
    batch_size=64,          # RS: 128; halve for Colab VRAM (RS themselves halved from
                            # their reference for the same reason). If A100: use 128.
    mini_batch_size=8,
    ppo_epochs=1,           # RS: 1-2; fixed at 1 for ALL axes (Equal-N analogue)
    init_kl_coef=0.2,       # RS: 0.2 for dialog-style tasks (0.05 only for summarization)
    # --- Generation (RS Table 1: output length uniform 16..32) ---
    output_min_len=16,
    output_max_len=32,
    # --- Equal-N analogue: identical across ALL axes, frozen before any run ---
    total_ppo_steps=200,    # pilot-scale; same number for every axis, никогда per-axis tuned
    prompt_seed=137,        # same prompt subsample for every axis
    train_seed=911,         # same train seed schedule for every axis
    n_prompts=2005,         # mirror the frozen N from the SFT recipe
    # --- SFT phase ---
    sft_epochs=1,
    sft_lr=2e-5,
    sft_batch_size=16,
    # --- paths ---
    out_dir="rs_runs",
    armorm_model=ARMORM_MODEL,
    armorm_load_in_4bit=True,
    armorm_reward_batch_size=8,
)

# ----------------------------------------------------------------------------


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_helpsteer2_prompts(n: int, seed: int):
    """Same prompt subsample for every axis (Equal-N analogue)."""
    from datasets import load_dataset
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    prompts = [ds[int(i)]["prompt"] for i in idx]
    return prompts


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


class ArmoRMHeadScorer:
    """Frozen ArmoRM scorer returning one HelpSteer2 head per text pair."""

    def __init__(self, axis: str, model_id: str = ARMORM_MODEL):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if axis not in ARMORM_HELPSTEER_OBJECTIVES:
            raise ValueError(f"No ArmoRM HelpSteer head configured for {axis!r}.")
        self.axis = axis
        self.objective_index, self.objective_name = ARMORM_HELPSTEER_OBJECTIVES[axis]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = None
        if CFG.get("armorm_load_in_4bit", True):
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

    def _format(self, prompt: str, response: str) -> torch.Tensor:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        text = f"Human: {prompt}\n\nAssistant: {response}"
        return self.tokenizer(text, return_tensors="pt", truncation=True).input_ids

    def score_batch(self, prompts: list[str], responses: list[str]) -> list[torch.Tensor]:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length.")
        rewards = []
        reward_device = next(self.model.parameters()).device
        batch_size = int(CFG.get("armorm_reward_batch_size", 8))
        for start in range(0, len(prompts), batch_size):
            stop = min(start + batch_size, len(prompts))
            encoded = []
            for prompt, response in zip(prompts[start:stop], responses[start:stop]):
                encoded.append(self._format(prompt, response).squeeze(0))
            padded = torch.nn.utils.rnn.pad_sequence(
                encoded,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            attention_mask = (padded != pad_id).long()
            inputs = {
                "input_ids": padded.to(reward_device),
                "attention_mask": attention_mask.to(reward_device),
            }
            with torch.inference_mode():
                outputs = self.model(**inputs)
            head_rewards = getattr(outputs, "rewards", None)
            if head_rewards is None:
                raise RuntimeError("ArmoRM output has no .rewards tensor.")
            tensor = torch.as_tensor(head_rewards).detach().float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[1] <= self.objective_index:
                raise RuntimeError(
                    f"ArmoRM returned only {tensor.shape[1]} heads; "
                    f"needed index {self.objective_index}."
                )
            rewards.extend(
                torch.tensor(value, dtype=torch.float32)
                for value in tensor[:, self.objective_index].cpu().tolist()
            )
        return rewards


# ----------------------------------------------------------------------------
# Phase 1: SFT base  (RS: pretrain -> SFT -> PPO; SFT is the shared init)
# ----------------------------------------------------------------------------

def run_sft():
    """TinyLlama + SFT on HelpSteer2 responses -> theta_SFT (full pipeline shared init).

    Deliberately minimal: generic instruction-SFT on HelpSteer2 (prompt -> response),
    NO attribute filtering here -- theta_SFT must be axis-neutral, specialization
    happens only in the PPO phase (RS-faithful).
    """
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments)
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
        # TinyLlama chat template, response supervised
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

    # merge LoRA into the base so theta_SFT is a plain checkpoint =
    # the single shared init all PPO runs start from (and the KL reference)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(out / "merged"))
    tok.save_pretrained(str(out / "merged"))
    print(f"[sft] theta_SFT saved to {out/'merged'}")


# ----------------------------------------------------------------------------
# Phase 2: one independent PPO run for ONE axis (call once per axis)
# ----------------------------------------------------------------------------

def run_ppo(
    axis: str,
    reward_model_id: str | None = None,
    circular_armorm_acknowledged: bool = False,
):
    from transformers import AutoTokenizer, pipeline
    from trl import (AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer)
    from peft import LoraConfig

    assert axis in ATTRIBUTES, f"unknown axis {axis}; choose one of {ATTRIBUTES}"
    rm_id = reward_model_id or HELD_OUT_RM.get(axis)
    assert rm_id, f"no reward model configured for axis {axis}"
    assert "<HF_ID" not in rm_id, "Phase-0 decision missing: set the reward model id"
    firewall = check_reward_firewall(
        axis,
        rm_id,
        circular_armorm_acknowledged=circular_armorm_acknowledged,
    )

    set_all_seeds(CFG["train_seed"])   # identical seed schedule across axes
    sft_path = Path(CFG["out_dir"]) / "theta_sft" / "merged"
    assert sft_path.exists(), "run --phase sft first (theta_SFT is the shared init)"
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
        torch_dtype=torch.bfloat16, device_map="auto")

    ppo_cfg = PPOConfig(
        learning_rate=CFG["learning_rate"],
        batch_size=CFG["batch_size"],
        mini_batch_size=CFG["mini_batch_size"],
        ppo_epochs=CFG["ppo_epochs"],
        init_kl_coef=CFG["init_kl_coef"],
        seed=CFG["train_seed"])
    trainer = PPOTrainer(config=ppo_cfg, model=policy, tokenizer=tok)

    # Held-out RM as a black-box scoring pipeline. In the explicitly
    # acknowledged circular mode, this becomes a frozen ArmoRM head scorer.
    if is_armorm_model(rm_id):
        reward_pipe = ArmoRMHeadScorer(axis=axis, model_id=rm_id)
    else:
        reward_pipe = pipeline("text-classification", model=rm_id,
                               device_map="auto", torch_dtype=torch.bfloat16)

    prompts = load_helpsteer2_prompts(CFG["n_prompts"], CFG["prompt_seed"])
    gen_kwargs = dict(do_sample=True, top_k=0, top_p=1.0,
                      pad_token_id=tok.eos_token_id)

    log, step = [], 0
    while step < CFG["total_ppo_steps"]:
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
        step += 1
        log.append({"step": step,
                    "mean_reward": float(np.mean([r.item() for r in rewards])),
                    "kl": float(stats.get("objective/kl", float("nan")))})
        if step % 10 == 0:
            print(f"[ppo:{axis}] step {step}/{CFG['total_ppo_steps']} "
                  f"reward={log[-1]['mean_reward']:.4f} kl={log[-1]['kl']:.3f}")

    # checkpoint selection: end-of-run adapter, chosen by held-out-RM/KL trace
    # (NEVER by ArmoRM). Only the LoRA adapter is saved -> delta_i is exactly
    # this adapter's effective update relative to theta_SFT.
    policy.save_pretrained(str(out / "adapter"))
    (out / "ppo_log.json").write_text(json.dumps(
        {
            "axis": axis,
            "rm": rm_id,
            "config": CFG,
            "firewall": firewall,
            "log": log,
        },
        indent=2,
    ))
    print(f"[ppo:{axis}] adapter saved to {out/'adapter'}; log written.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["sft", "ppo"], required=True)
    ap.add_argument("--axis", default=None, help="required for --phase ppo")
    ap.add_argument(
        "--reward_model",
        default=None,
        help="Reward model id for PPO. Use ArmoRM only with --circular_armorm_acknowledged.",
    )
    ap.add_argument(
        "--circular_armorm_acknowledged",
        action="store_true",
        help="Explicitly acknowledge circular PPO against ArmoRM.",
    )
    a = ap.parse_args()
    if a.phase == "sft":
        run_sft()
    else:
        assert a.axis, "--axis required for ppo phase"
        run_ppo(
            a.axis,
            reward_model_id=a.reward_model,
            circular_armorm_acknowledged=a.circular_armorm_acknowledged,
        )
