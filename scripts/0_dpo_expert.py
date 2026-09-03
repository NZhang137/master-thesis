"""Train one TinyLlama HelpSteer2 DPO expert for the NB11 RQ2 regime.

This is the repository-local version of the supplied MOD-style
``0_dpo_expert.py``.  For every HelpSteer2 prompt it forms all within-prompt
response pairs, discards ties on the requested attribute, and makes the
higher-rated response ``chosen``.  ArmoRM is deliberately absent from this
file: pair construction, optimization, resumption, and completion checks use
only HelpSteer2 and DPO state.

The five NB11 calls must use the same base revision, seed, pair cap, LoRA
configuration, and optimization settings.  Only ``--reward_name`` changes.
The fixed pair cap gives every expert the same number of optimizer steps even
though attribute-specific ties produce differently sized candidate pools.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from packaging.version import Version


HELPSTEER_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATASET = "nvidia/HelpSteer2"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 137
DEFAULT_MAX_PAIRS = 2690


def _clean_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"HelpSteer2 row has an empty {field}.")
    return text


def _rating(row: Mapping[str, object], attribute: str) -> float:
    try:
        value = float(row[attribute])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {attribute!r} rating in HelpSteer2 row.") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {attribute!r} rating in HelpSteer2 row.")
    return value


def pair_identifier(prompt: str, response_a: str, response_b: str) -> str:
    """Return an orientation-independent identifier for one response pair."""
    responses = sorted((response_a, response_b))
    payload = json.dumps([prompt, *responses], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pairs_from_rows(
    rows: Iterable[Mapping[str, object]],
    attribute: str,
) -> list[dict[str, str]]:
    """Create deterministic within-prompt DPO pairs for one attribute.

    Grouping is by prompt value rather than by adjacency, so a future dataset
    reordering cannot silently create cross-prompt pairs.
    """
    if attribute not in HELPSTEER_ATTRIBUTES:
        raise ValueError(
            f"Unsupported attribute {attribute!r}; choose from {HELPSTEER_ATTRIBUTES}."
        )

    groups: OrderedDict[str, list[Mapping[str, object]]] = OrderedDict()
    for row in rows:
        prompt = _clean_text(row.get("prompt"), "prompt")
        _clean_text(row.get("response"), "response")
        _rating(row, attribute)
        groups.setdefault(prompt, []).append(row)

    pairs: list[dict[str, str]] = []
    for prompt, group in groups.items():
        for row_a, row_b in combinations(group, 2):
            score_a = _rating(row_a, attribute)
            score_b = _rating(row_b, attribute)
            if score_a == score_b:
                continue
            high, low = (row_a, row_b) if score_a > score_b else (row_b, row_a)
            chosen = _clean_text(high.get("response"), "response")
            rejected = _clean_text(low.get("response"), "response")
            if chosen == rejected:
                continue
            pairs.append(
                {
                    "pair_id": pair_identifier(prompt, chosen, rejected),
                    "raw_prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
    return pairs


def select_pairs(
    pairs: Sequence[Mapping[str, str]],
    *,
    seed: int,
    max_pairs: int,
) -> list[dict[str, str]]:
    """Shuffle reproducibly and take exactly ``max_pairs`` candidate pairs."""
    if max_pairs < 1:
        raise ValueError("max_pairs must be at least 1.")
    if len(pairs) < max_pairs:
        raise ValueError(
            f"Only {len(pairs)} non-tied candidate pairs exist; {max_pairs} requested."
        )
    selected = [dict(pair) for pair in pairs]
    random.Random(seed).shuffle(selected)
    return selected[:max_pairs]


def format_pairs_for_dpo(
    pairs: Sequence[Mapping[str, str]],
    tokenizer: Any,
) -> list[dict[str, str]]:
    """Render TinyLlama's exact user/assistant boundary for TRL DPOTrainer."""
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The base tokenizer has no chat template.")
    eos = tokenizer.eos_token or ""
    formatted: list[dict[str, str]] = []
    for pair in pairs:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": pair["raw_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append(
            {
                "prompt": prompt,
                "chosen": pair["chosen"] + eos,
                "rejected": pair["rejected"] + eos,
            }
        )
    return formatted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_revision(model_name: str, requested: str | None) -> str:
    if requested:
        return requested
    from huggingface_hub import HfApi

    revision = HfApi().model_info(model_name).sha
    if not revision:
        raise RuntimeError(f"Could not resolve an immutable revision for {model_name}.")
    return revision


def latest_checkpoint(trainer_dir: Path) -> str | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in trainer_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return str(max(checkpoints)[1]) if checkpoints else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name", required=True, choices=HELPSTEER_ATTRIBUTES)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base_revision", default=None)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output_root", default="results/dpo_rq2/nb11_pairs2690_seed137_run1"
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save_steps", type=int, default=100)
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
        raise RuntimeError(
            "Incompatible DPO stack: TRL < 0.12 overrides get_batch_samples with "
            "a signature that conflicts with Transformers >= 4.46. Use the NB11 "
            "pins trl==0.11.4 and transformers==4.45.2."
        )

    from datasets import Dataset, load_dataset

    dataset = load_dataset(
        args.dataset_name, split=args.split, revision=args.dataset_revision
    )
    candidates = build_pairs_from_rows(dataset, args.reward_name)
    selected = select_pairs(candidates, seed=args.seed, max_pairs=args.max_pairs)
    print(
        f"[pairs] axis={args.reward_name} candidates={len(candidates)} "
        f"selected={len(selected)} prompts={len({p['raw_prompt'] for p in selected})}"
    )
    if args.inspect_only:
        return

    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("NB11 DPO training requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("NB11 is fixed to bf16 and requires a bf16-capable GPU.")

    revision = resolve_revision(args.base_model_name, args.base_revision)
    output_root = Path(args.output_root)
    expert_dir = output_root / f"dpo_{args.reward_name}"
    adapter_dir = expert_dir / "adapter"
    trainer_dir = expert_dir / "trainer"
    binding_path = expert_dir / "run_binding.json"
    manifest_path = expert_dir / "training_manifest.json"

    binding = {
        "schema_version": 1,
        "training_method": "DPO",
        "trainer_script_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "transformers",
                "tokenizers",
                "peft",
                "accelerate",
                "trl",
                "datasets",
            )
        },
        "reward_name": args.reward_name,
        "base_model_name": args.base_model_name,
        "base_revision": revision,
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "pair_rule": "within-prompt; higher target rating chosen; ties discarded",
        "pair_seed": args.seed,
        "candidate_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "selected_pair_ids_sha256": canonical_hash(
            {"pair_ids": [pair["pair_id"] for pair in selected]}
        ),
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
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "reference_model": "same frozen base; active LoRA disabled by PEFT-DPO",
        "armorm_used_during_training": False,
        "checkpoint_selection": "fixed final epoch; no reward-model selection",
    }
    binding_sha256 = canonical_hash(binding)

    if manifest_path.exists() and adapter_dir.joinpath("adapter_model.safetensors").exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("binding_sha256") != binding_sha256:
            raise RuntimeError(
                f"Existing {expert_dir} has a different binding. Use a new run tag "
                "or pass --overwrite explicitly."
            )
        print(f"[skip] completed adapter already matches binding: {adapter_dir}")
        return

    if args.overwrite and expert_dir.exists():
        shutil.rmtree(expert_dir)
    expert_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir.mkdir(parents=True, exist_ok=True)

    binding_record = {**binding, "binding_sha256": binding_sha256}
    if binding_path.exists():
        existing_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing_binding.get("binding_sha256") != binding_sha256:
            raise RuntimeError(
                f"Partial run at {expert_dir} has a different binding; refusing "
                "to resume it. Use a new run tag or --overwrite."
            )
    elif any(trainer_dir.glob("checkpoint-*")):
        raise RuntimeError(
            f"Partial checkpoints exist in {trainer_dir} without run_binding.json. "
            "Their provenance is unknown, so automatic resume is unsafe."
        )
    else:
        binding_path.write_text(
            json.dumps(binding_record, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, revision=revision, use_fast=True
    )
    if tokenizer.eos_token is None:
        raise RuntimeError("Base tokenizer has no EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = Dataset.from_list(format_pairs_for_dpo(selected, tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=binding["lora"]["target_modules"],
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
        run_name=f"dpo_{args.reward_name}",
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
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )
    checkpoint = latest_checkpoint(trainer_dir) if args.resume else None
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
        "adapter_model_sha256": sha256_file(adapter_weights),
        "train_metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in train_result.metrics.items()
        },
        "completed": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] axis={args.reward_name} adapter={adapter_dir} "
        f"sha256={manifest['adapter_model_sha256']}"
    )

    del trainer, model, train_dataset, dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
