"""Train one Llama-2-7B DPO expert on one of four preference datasets (NB16).

Additive companion to ``scripts/0_dpo_expert_llama2.py`` (NB14). Base model, prompt
format, tokenizer, LoRA, DPO hyperparameters and checkpointing are the NB14 recipe;
only the source of the preference pairs differs. Fourteen experts in total:

=================  ==========================  ==================================================
dataset key        Hugging Face dataset        objectives (one adapter each)
=================  ==========================  ==================================================
``hh_rlhf``        Anthropic/hh-rlhf           helpful (helpful-base), harmless (harmless-base),
                                               humor (helpful-base pairs re-ordered by a humor
                                               classifier, see below)
``pku_saferlhf``   PKU-Alignment/PKU-SafeRLHF  better (better_response_id), safer (safer_response_id)
``helpsteer1``     nvidia/HelpSteer            helpfulness, correctness, coherence, complexity,
                                               verbosity (NB11 pair rule, unchanged)
``ultrafeedback``  openbmb/UltraFeedback       helpfulness, honesty, instruction_following,
                                               truthfulness (per-aspect ratings of 4 completions)
=================  ==========================  ==================================================

Pair rules
----------
* HelpSteer1 and UltraFeedback: all within-prompt response pairs whose rating on the
  target attribute differs; the higher-rated response is chosen, ties are discarded.
* PKU-SafeRLHF: one pair per row, oriented by the human label of the objective. Both
  objectives use the same response pairs; only the orientation differs.
* HH-RLHF helpful / harmless: the human chosen/rejected pairs of the respective subset.
  Multi-turn dialogues keep their full context. Rows whose chosen and rejected dialogues
  do not share the same context are discarded.
* HH-RLHF humor: HH-RLHF has no humor labels. As in the Rewarded-Soups / RiC assistant
  task, humor is defined by the classifier ``mohameddhiab/humor-no-humor``. Both
  responses of every helpful-base pair are scored with P(HUMOR); a pair is kept when the
  scores differ by at least ``humor_margin`` and the funnier response is chosen. The
  scores are computed once in NB16 and passed in as a JSON file, whose hash is bound.

Prompt format (as NB14 and the supervisor's assistant task)
------------------------------------------------------------
Single-turn: ``"\\n\\nHuman: {prompt} " + response_split``. HH-RLHF multi-turn: the
original dialogue up to the last human turn, then ``" " + response_split``; for a
single-turn HH dialogue this is exactly the single-turn string. Responses are
``" " + text + eos``.

No reward model is used for training or checkpoint selection; the humor classifier only
orders pairs before training. Checkpoint selection is the fixed final epoch.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from packaging.version import Version

_LLAMA2_PATH = Path(__file__).resolve().with_name("0_dpo_expert_llama2.py")
_spec = importlib.util.spec_from_file_location("nb14_dpo_expert_llama2", _LLAMA2_PATH)
assert _spec is not None and _spec.loader is not None
llama2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llama2)
nb11 = llama2.nb11

DEFAULT_BASE_MODEL = llama2.DEFAULT_BASE_MODEL
DEFAULT_RESPONSE_SPLIT = llama2.DEFAULT_RESPONSE_SPLIT
LORA = llama2.LORA
HUMOR_MODEL = "mohameddhiab/humor-no-humor"
HH_ASSISTANT = "\n\nAssistant:"
HH_HUMAN = "\n\nHuman:"

DATASETS: dict[str, dict[str, Any]] = {
    "hh_rlhf": {"hf_name": "Anthropic/hh-rlhf",
                "objectives": ("helpful", "harmless", "humor")},
    "pku_saferlhf": {"hf_name": "PKU-Alignment/PKU-SafeRLHF",
                     "objectives": ("better", "safer")},
    "helpsteer1": {"hf_name": "nvidia/HelpSteer",
                   "objectives": tuple(nb11.HELPSTEER_ATTRIBUTES)},
    "ultrafeedback": {"hf_name": "openbmb/UltraFeedback",
                      "objectives": ("helpfulness", "honesty", "instruction_following", "truthfulness")},
}
HH_SUBSET = {"helpful": "helpful-base", "harmless": "harmless-base", "humor": "helpful-base"}
PKU_LABEL = {"better": "better_response_id", "safer": "safer_response_id"}
PAIR_RULES = {
    "hh_rlhf": "human chosen/rejected of the subset; shared dialogue context required",
    "hh_rlhf/humor": "helpful-base pairs ordered by P(HUMOR); |difference| >= margin; funnier chosen",
    "pku_saferlhf": "one pair per row, oriented by the objective's human label",
    "helpsteer1": "within-prompt; higher target rating chosen; ties discarded (NB11 rule)",
    "ultrafeedback": "within-instruction over the 4 completions; higher aspect Rating chosen; ties and N/A discarded",
}


# --------------------------------------------------------------------------- helpers

def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def objectives(dataset_key: str) -> tuple[str, ...]:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset_key!r}; choose from {sorted(DATASETS)}.")
    return tuple(DATASETS[dataset_key]["objectives"])


def pair_rule(dataset_key: str, objective: str) -> str:
    return PAIR_RULES.get(f"{dataset_key}/{objective}", PAIR_RULES[dataset_key])


def _pair(prompt_key: str, chosen: str, rejected: str, *, context: str | None = None) -> dict[str, str]:
    pair = {"pair_id": nb11.pair_identifier(prompt_key, chosen, rejected),
            "raw_prompt": prompt_key, "chosen": chosen, "rejected": rejected}
    if context is not None:
        pair["hh_context"] = context
    return pair


def split_hh_dialogue(text: str) -> tuple[str, str] | None:
    """Split an HH-RLHF dialogue into (context up to the last human turn, final response)."""
    text = str(text or "")
    index = text.rfind(HH_ASSISTANT)
    if index < 0 or HH_HUMAN not in text[:index]:
        return None
    context = text[:index].rstrip()
    response = text[index + len(HH_ASSISTANT):].strip()
    if not context.lstrip().startswith(HH_HUMAN.strip()) or not response:
        return None
    if not context.startswith("\n\n"):
        context = "\n\n" + context.lstrip()
    return context, response


def parse_hh_rows(rows: Iterable[Mapping[str, object]]) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Return (context, chosen, rejected) triples and drop counts."""
    triples: list[tuple[str, str, str]] = []
    dropped = {"unparsable": 0, "context_mismatch": 0, "identical": 0}
    for row in rows:
        chosen = split_hh_dialogue(row.get("chosen"))
        rejected = split_hh_dialogue(row.get("rejected"))
        if chosen is None or rejected is None:
            dropped["unparsable"] += 1
            continue
        if chosen[0] != rejected[0]:
            dropped["context_mismatch"] += 1
            continue
        if chosen[1] == rejected[1]:
            dropped["identical"] += 1
            continue
        triples.append((chosen[0], chosen[1], rejected[1]))
    return triples, dropped


def _uf_rating(completion: Mapping[str, Any], aspect: str) -> float | None:
    try:
        value = completion["annotations"][aspect]["Rating"]
    except (KeyError, TypeError):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None          # "N/A" and empty ratings
    return rating if math.isfinite(rating) else None


# ----------------------------------------------------------------------- pair builders

def build_hh_pairs(rows: Iterable[Mapping[str, object]], objective: str,
                   humor_scores: Mapping[str, float] | None = None,
                   humor_margin: float = 0.2) -> list[dict[str, str]]:
    triples, _ = parse_hh_rows(rows)
    pairs: list[dict[str, str]] = []
    if objective in ("helpful", "harmless"):
        for context, chosen, rejected in triples:
            pairs.append(_pair(context, chosen, rejected, context=context))
        return pairs
    if objective != "humor":
        raise ValueError(f"Unknown HH-RLHF objective {objective!r}.")
    if humor_scores is None:
        raise ValueError("The humor objective needs humor scores.")
    for context, first, second in triples:
        try:
            s_first = float(humor_scores[text_sha256(first)])
            s_second = float(humor_scores[text_sha256(second)])
        except KeyError as error:
            raise KeyError("Humor score missing for an HH-RLHF response; rescore the dataset.") from error
        if abs(s_first - s_second) < humor_margin:
            continue
        chosen, rejected = (first, second) if s_first > s_second else (second, first)
        pairs.append(_pair(context, chosen, rejected, context=context))
    return pairs


def build_pku_pairs(rows: Iterable[Mapping[str, object]], objective: str) -> list[dict[str, str]]:
    label = PKU_LABEL[objective]
    pairs: list[dict[str, str]] = []
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        responses = [str(row.get("response_0") or "").strip(), str(row.get("response_1") or "").strip()]
        winner = row.get(label)
        if not prompt or not all(responses) or responses[0] == responses[1] or winner not in (0, 1):
            continue
        pairs.append(_pair(prompt, responses[int(winner)], responses[1 - int(winner)]))
    return pairs


def build_ultrafeedback_pairs(rows: Iterable[Mapping[str, object]], objective: str) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in rows:
        prompt = str(row.get("instruction") or "").strip()
        if not prompt:
            continue
        rated = []
        for completion in row.get("completions") or []:
            response = str(completion.get("response") or "").strip()
            rating = _uf_rating(completion, objective)
            if response and rating is not None:
                rated.append((rating, response))
        for (score_a, text_a), (score_b, text_b) in combinations(rated, 2):
            if score_a == score_b or text_a == text_b:
                continue
            high, low = (text_a, text_b) if score_a > score_b else (text_b, text_a)
            pairs.append(_pair(prompt, high, low))
    return pairs


def build_pairs(dataset_key: str, objective: str, rows: Iterable[Mapping[str, object]], *,
                humor_scores: Mapping[str, float] | None = None,
                humor_margin: float = 0.2) -> list[dict[str, str]]:
    if objective not in objectives(dataset_key):
        raise ValueError(f"{objective!r} is not an objective of {dataset_key!r}.")
    if dataset_key == "hh_rlhf":
        pairs = build_hh_pairs(rows, objective, humor_scores, humor_margin)
    elif dataset_key == "pku_saferlhf":
        pairs = build_pku_pairs(rows, objective)
    elif dataset_key == "helpsteer1":
        # The NB11 builder rejects empty texts; HelpSteer1 rows without prompt or response are skipped.
        usable = (row for row in rows
                  if str(row.get("prompt") or "").strip() and str(row.get("response") or "").strip())
        pairs = nb11.build_pairs_from_rows(usable, objective)
    else:
        pairs = build_ultrafeedback_pairs(rows, objective)
    # One orientation per unordered pair: duplicates with the same orientation are dropped,
    # contradictory duplicates (both orientations present) are removed entirely.
    seen: dict[str, dict[str, str]] = {}
    contradictory: set[str] = set()
    for pair in pairs:
        previous = seen.get(pair["pair_id"])
        if previous is None:
            seen[pair["pair_id"]] = pair
        elif previous["chosen"] != pair["chosen"]:
            contradictory.add(pair["pair_id"])
    return [pair for pair_id, pair in seen.items() if pair_id not in contradictory]


def load_objective_rows(dataset_key: str, objective: str, revision: str, split: str = "train"):
    """Load the Hugging Face rows an objective's pairs are built from."""
    from datasets import load_dataset

    name = DATASETS[dataset_key]["hf_name"]
    if dataset_key == "hh_rlhf":
        return load_dataset(name, data_dir=HH_SUBSET[objective], split=split, revision=revision)
    return load_dataset(name, split=split, revision=revision)


# --------------------------------------------------------------------------- humor

def humor_texts(rows: Iterable[Mapping[str, object]]) -> list[str]:
    """Unique final responses of the parsed HH-RLHF pairs (both sides)."""
    triples, _ = parse_hh_rows(rows)
    texts = {text for _, first, second in triples for text in (first, second)}
    return sorted(texts)


def score_humor(texts: Sequence[str], *, model_name: str = HUMOR_MODEL, revision: str | None = None,
                batch_size: int = 256, device: str = "cuda") -> dict[str, float]:
    """P(HUMOR) for each text, keyed by the text's SHA-256."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision).to(device).eval()
    label2id = {str(k).upper(): int(v) for k, v in model.config.label2id.items()}
    if "HUMOR" not in label2id:
        raise RuntimeError(f"No HUMOR label in {model_name}: {model.config.label2id}")
    humor_index = label2id["HUMOR"]
    scores: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            batch = tokenizer(chunk, truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**batch).logits.float(), dim=-1)[:, humor_index].tolist()
            scores.update({text_sha256(text): float(p) for text, p in zip(chunk, probs)})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


# ------------------------------------------------------------------------- training

def render_prompt(pair: Mapping[str, str], response_split: str = DEFAULT_RESPONSE_SPLIT) -> str:
    if pair.get("hh_context"):
        return f"{pair['hh_context']} {response_split}"
    return llama2.render_hh_prompt(pair["raw_prompt"], response_split)


def format_pairs(pairs: Sequence[Mapping[str, str]], tokenizer: Any,
                 response_split: str = DEFAULT_RESPONSE_SPLIT) -> list[dict[str, str]]:
    eos = tokenizer.eos_token or ""
    return [{"prompt": render_prompt(pair, response_split),
             "chosen": " " + pair["chosen"] + eos,
             "rejected": " " + pair["rejected"] + eos} for pair in pairs]


def selected_pairs(dataset_key: str, objective: str, rows: Iterable[Mapping[str, object]], *,
                   pair_seed: int, max_pairs: int, humor_scores: Mapping[str, float] | None = None,
                   humor_margin: float = 0.2) -> tuple[list[dict[str, str]], int]:
    candidates = build_pairs(dataset_key, objective, rows, humor_scores=humor_scores, humor_margin=humor_margin)
    return nb11.select_pairs(candidates, seed=pair_seed, max_pairs=max_pairs), len(candidates)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", required=True, choices=sorted(DATASETS))
    parser.add_argument("--objective", required=True)
    parser.add_argument("--dataset_revision", required=True)
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base_revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--max_pairs", type=int, required=True)
    parser.add_argument("--humor_scores", default=None, help="JSON {sha256: P(HUMOR)}; humor objective only")
    parser.add_argument("--humor_margin", type=float, default=0.2)
    parser.add_argument("--humor_model_revision", default=None)
    parser.add_argument("--response_split", default=DEFAULT_RESPONSE_SPLIT)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--pair_seed", type=int, default=137)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    args = parser.parse_args(argv)
    if args.objective not in objectives(args.dataset_key):
        parser.error(f"--objective must be one of {objectives(args.dataset_key)} for {args.dataset_key}")
    if args.objective == "humor" and not args.humor_scores:
        parser.error("--humor_scores is required for the humor objective")
    return args


def build_binding(args: argparse.Namespace, *, candidates: int, selected: Sequence[Mapping[str, str]],
                  dataset_fingerprint: str | None, humor_scores_sha256: str | None) -> dict[str, Any]:
    humor = args.objective == "humor"
    return {
        "schema_version": 1,
        "training_method": "DPO",
        "trainer_script_sha256": nb11.sha256_file(Path(__file__).resolve()),
        "recipe_script_sha256": nb11.sha256_file(_LLAMA2_PATH),
        "pair_logic_script_sha256": nb11.sha256_file(Path(nb11.__file__).resolve()),
        "runtime_versions": {package: importlib.metadata.version(package) for package in
                             ("torch", "transformers", "tokenizers", "peft", "accelerate", "trl", "datasets")},
        "dataset_key": args.dataset_key,
        "objective": args.objective,
        "reward_name": f"{args.dataset_key}/{args.objective}",
        "base_model_name": args.base_model_name,
        "base_revision": args.base_revision,
        "dataset_name": DATASETS[args.dataset_key]["hf_name"],
        "dataset_subset": HH_SUBSET.get(args.objective) if args.dataset_key == "hh_rlhf" else None,
        "dataset_revision": args.dataset_revision,
        "dataset_split": args.split,
        "dataset_fingerprint": dataset_fingerprint,
        "pair_rule": pair_rule(args.dataset_key, args.objective),
        "pair_seed": args.pair_seed,
        "candidate_pair_count": candidates,
        "selected_pair_count": len(selected),
        "selected_pair_ids_sha256": nb11.canonical_hash({"pair_ids": [pair["pair_id"] for pair in selected]}),
        "humor_model": HUMOR_MODEL if humor else None,
        "humor_model_revision": args.humor_model_revision if humor else None,
        "humor_scores_sha256": humor_scores_sha256 if humor else None,
        "humor_margin": args.humor_margin if humor else None,
        "prompt_format": "hh_rlhf: '\\n\\nHuman: {prompt} ' + response_split (HH multi-turn: context + ' ' + "
                         "response_split); response = ' ' + text + eos",
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
        "reward_model_used_during_training": False,
        "checkpoint_selection": "fixed final epoch; no reward-model selection",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs <= 0 or args.beta <= 0 or args.lr <= 0:
        raise ValueError("epochs, beta, and lr must be positive.")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("batch_size and grad_accum must be positive integers.")
    transformers_version = Version(importlib.metadata.version("transformers"))
    trl_version = Version(importlib.metadata.version("trl"))
    if trl_version < Version("0.12") and transformers_version >= Version("4.46"):
        raise RuntimeError("Use the NB11 pins trl==0.11.4 and transformers==4.45.2.")

    humor_scores, humor_sha = None, None
    if args.objective == "humor":
        scores_path = Path(args.humor_scores)
        humor_sha = nb11.sha256_file(scores_path)
        humor_scores = json.loads(scores_path.read_text(encoding="utf-8"))["scores"]

    rows = load_objective_rows(args.dataset_key, args.objective, args.dataset_revision, args.split)
    selected, n_candidates = selected_pairs(args.dataset_key, args.objective, rows, pair_seed=args.pair_seed,
                                            max_pairs=args.max_pairs, humor_scores=humor_scores,
                                            humor_margin=args.humor_margin)
    print(f"[pairs] {args.dataset_key}/{args.objective} candidates={n_candidates} selected={len(selected)} "
          f"prompts={len({p['raw_prompt'] for p in selected})}")
    if args.inspect_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("NB16 DPO training requires a bf16-capable CUDA GPU (A100).")

    expert_dir = Path(args.output_root) / args.dataset_key / f"dpo_{args.objective}"
    adapter_dir = expert_dir / "adapter"
    trainer_dir = expert_dir / "trainer"
    binding_path = expert_dir / "run_binding.json"
    manifest_path = expert_dir / "training_manifest.json"
    binding = build_binding(args, candidates=n_candidates, selected=selected,
                            dataset_fingerprint=getattr(rows, "_fingerprint", None), humor_scores_sha256=humor_sha)
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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, revision=args.base_revision, use_fast=False)
    if tokenizer.eos_token is None:
        raise RuntimeError("Base tokenizer has no EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = Dataset.from_list(format_pairs(selected, tokenizer, args.response_split))

    model = AutoModelForCausalLM.from_pretrained(args.base_model_name, revision=args.base_revision,
                                                 torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer), "embedding/tokenizer mismatch"

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA["r"], lora_alpha=LORA["alpha"],
                             lora_dropout=LORA["dropout"], bias=LORA["bias"], target_modules=LORA["target_modules"])
    training_args = DPOConfig(
        output_dir=str(trainer_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        report_to="none", run_name=f"dpo_llama2_{args.dataset_key}_{args.objective}",
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False, beta=args.beta,
        loss_type="sigmoid", disable_dropout=True, max_length=args.max_length,
        max_prompt_length=args.max_prompt_length, truncation_mode="keep_end",
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=training_args, train_dataset=train_dataset,
                         tokenizer=tokenizer, peft_config=lora_config)
    checkpoint = nb11.latest_checkpoint(trainer_dir) if args.resume else None
    if checkpoint:
        print(f"[resume] {checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    log_history = list(trainer.state.log_history)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise RuntimeError(f"Training completed but {adapter_weights} is missing.")
    (expert_dir / "log_history.json").write_text(json.dumps(log_history, indent=2) + "\n", encoding="utf-8")
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
    print(f"[done] {args.dataset_key}/{args.objective} adapter={adapter_dir} sha256={manifest['adapter_model_sha256']}")
    del trainer, model, train_dataset, rows
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
