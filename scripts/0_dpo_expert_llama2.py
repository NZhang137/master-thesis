"""Train one Llama-2-7B HelpSteer2 DPO expert (NB14).

This is an additive companion to ``scripts/0_dpo_expert.py``. Pair construction,
pair selection, and hashing are imported unchanged from that file, so NB14 uses
exactly the same 2,690 preference pairs per attribute as NB11/NB11.1 for the same
pair seed. Only the parts that cannot carry over to the raw Llama-2 base differ:

* Prompt format. ``meta-llama/Llama-2-7b-hf`` is a raw base model without a chat
  template, so the TinyLlama chat template cannot be used. NB14 uses the HH-RLHF
  format of the supervisor's assistant-task pipeline,
  ``"\\n\\nHuman: {prompt} " + response_split``, with ``response_split`` defaulting to
  ``"\\n\\nAssistant:"`` (``Instructions.response_split``).
* Tokenizer. Loaded with ``use_fast=False``, as in the supervisor's
  ``load_main_tokenizer`` for Llama. No tokens are added, so no embedding resize
  happens and theta_0 is exactly the raw base.
* Batch shape. Defaults to batch 1 x gradient accumulation 8, the same effective
  batch of 8 as NB11, to fit a 40 GB A100.

ArmoRM is never loaded. Checkpoint selection is the fixed final epoch.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import Version

_ORIGINAL_PATH = Path(__file__).resolve().with_name("0_dpo_expert.py")
_spec = importlib.util.spec_from_file_location("nb11_dpo_expert", _ORIGINAL_PATH)
assert _spec is not None and _spec.loader is not None
nb11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb11)

HELPSTEER_ATTRIBUTES = nb11.HELPSTEER_ATTRIBUTES
DEFAULT_BASE_MODEL = "meta-llama/Llama-2-7b-hf"
DEFAULT_RESPONSE_SPLIT = "\n\nAssistant:"
HUMAN_PREFIX = "\n\nHuman: "
LORA = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
}


def render_hh_prompt(raw_prompt: str, response_split: str = DEFAULT_RESPONSE_SPLIT) -> str:
    """Exact prompt string of the supervisor snippet: Human turn, space, response split."""
    return f"{HUMAN_PREFIX}{raw_prompt} {response_split}"


def format_pairs_hh(
    pairs: Sequence[Mapping[str, str]],
    tokenizer: Any,
    response_split: str = DEFAULT_RESPONSE_SPLIT,
) -> list[dict[str, str]]:
    """Render DPO rows in HH-RLHF format; responses start after one space and end with EOS."""
    eos = tokenizer.eos_token or ""
    return [
        {
            "prompt": render_hh_prompt(pair["raw_prompt"], response_split),
            "chosen": " " + pair["chosen"] + eos,
            "rejected": " " + pair["rejected"] + eos,
        }
        for pair in pairs
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name", required=True, choices=HELPSTEER_ATTRIBUTES)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base_revision", required=True,
                        help="Immutable revision; resolved once in NB14 for all five axes.")
    parser.add_argument("--dataset_name", default=nb11.DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", required=True)
    parser.add_argument("--split", default=nb11.DEFAULT_SPLIT)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--response_split", default=DEFAULT_RESPONSE_SPLIT)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_pairs", type=int, default=nb11.DEFAULT_MAX_PAIRS)
    parser.add_argument("--seed", type=int, default=nb11.DEFAULT_SEED)
    parser.add_argument("--pair_seed", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.beta <= 0 or args.lr <= 0:
        raise ValueError("epochs, beta, and lr must be positive.")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be positive integers.")
    transformers_version = Version(importlib.metadata.version("transformers"))
    trl_version = Version(importlib.metadata.version("trl"))
    if trl_version < Version("0.12") and transformers_version >= Version("4.46"):
        raise RuntimeError("Use the NB11 pins trl==0.11.4 and transformers==4.45.2.")

    from datasets import Dataset, load_dataset

    dataset = load_dataset(args.dataset_name, split=args.split, revision=args.dataset_revision)
    pair_seed = nb11.resolve_pair_seed(args.seed, args.pair_seed)
    candidates = nb11.build_pairs_from_rows(dataset, args.reward_name)
    selected = nb11.select_pairs(candidates, seed=pair_seed, max_pairs=args.max_pairs)
    print(f"[pairs] axis={args.reward_name} candidates={len(candidates)} "
          f"selected={len(selected)} prompts={len({p['raw_prompt'] for p in selected})}")
    if args.inspect_only:
        return

    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("NB14 DPO training requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("NB14 is fixed to bf16 and requires a bf16-capable GPU.")

    expert_dir = Path(args.output_root) / f"dpo_{args.reward_name}"
    adapter_dir = expert_dir / "adapter"
    trainer_dir = expert_dir / "trainer"
    binding_path = expert_dir / "run_binding.json"
    manifest_path = expert_dir / "training_manifest.json"

    binding = {
        "schema_version": 1,
        "training_method": "DPO",
        "trainer_script_sha256": nb11.sha256_file(Path(__file__).resolve()),
        "pair_logic_script_sha256": nb11.sha256_file(_ORIGINAL_PATH),
        "runtime_versions": {package: importlib.metadata.version(package) for package in
                             ("torch", "transformers", "tokenizers", "peft", "accelerate",
                              "trl", "datasets")},
        "reward_name": args.reward_name,
        "base_model_name": args.base_model_name,
        "base_revision": args.base_revision,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "pair_rule": "within-prompt; higher target rating chosen; ties discarded",
        "pair_seed": pair_seed,
        "candidate_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "selected_pair_ids_sha256": nb11.canonical_hash(
            {"pair_ids": [pair["pair_id"] for pair in selected]}),
        "prompt_format": "hh_rlhf: '\\n\\nHuman: {prompt} ' + response_split; response = ' ' + text + eos",
        "response_split": args.response_split,
        "tokenizer_use_fast": False,
        "embedding_resize": False,
        "beta": args.beta,
        "loss_type": "sigmoid",
        "dpo_disable_dropout": True,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "precision": "bf16",
        "seed": args.seed,
        "lora": LORA,
        "reference_model": "same frozen base; active LoRA disabled by PEFT-DPO",
        "armorm_used_during_training": False,
        "checkpoint_selection": "fixed final epoch; no reward-model selection",
    }
    binding_sha256 = nb11.canonical_hash(binding)

    if manifest_path.exists() and adapter_dir.joinpath("adapter_model.safetensors").exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("binding_sha256") != binding_sha256:
            raise RuntimeError(f"Existing {expert_dir} has a different binding.")
        print(f"[skip] completed adapter already matches binding: {adapter_dir}")
        return

    if args.overwrite and expert_dir.exists():
        shutil.rmtree(expert_dir)
    expert_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir.mkdir(parents=True, exist_ok=True)
    if binding_path.exists():
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing_binding.get("binding_sha256") != binding_sha256:
            raise RuntimeError(f"Partial run at {expert_dir} has a different binding; refusing to resume.")
    elif any(trainer_dir.glob("checkpoint-*")):
        raise RuntimeError(f"Checkpoints in {trainer_dir} without run_binding.json; unsafe to resume.")
    else:
        binding_path.write_text(json.dumps({**binding, "binding_sha256": binding_sha256},
                                           indent=2, sort_keys=True) + "\n", encoding="utf-8")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, revision=args.base_revision,
                                              use_fast=False)
    if tokenizer.eos_token is None:
        raise RuntimeError("Base tokenizer has no EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = Dataset.from_list(format_pairs_hh(selected, tokenizer, args.response_split))

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, revision=args.base_revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), "embedding/tokenizer mismatch"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA["r"], lora_alpha=LORA["alpha"],
        lora_dropout=LORA["dropout"], bias=LORA["bias"], target_modules=LORA["target_modules"],
    )
    training_args = DPOConfig(
        output_dir=str(trainer_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        run_name=f"dpo_llama2_{args.reward_name}",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        beta=args.beta,
        loss_type="sigmoid",
        disable_dropout=True,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        truncation_mode="keep_end",
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=training_args, train_dataset=train_dataset,
        tokenizer=tokenizer, peft_config=lora_config,
    )
    checkpoint = nb11.latest_checkpoint(trainer_dir) if args.resume else None
    if checkpoint:
        print(f"[resume] {checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise RuntimeError(f"Training completed but {adapter_weights} is missing.")
    manifest = {
        **binding,
        "binding_sha256": binding_sha256,
        "adapter_path": str(adapter_dir),
        "adapter_model_sha256": nb11.sha256_file(adapter_weights),
        "train_metrics": {key: float(value) if isinstance(value, (int, float)) else value
                          for key, value in train_result.metrics.items()},
        "completed": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] axis={args.reward_name} adapter={adapter_dir} sha256={manifest['adapter_model_sha256']}")
    del trainer, model, train_dataset, dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
