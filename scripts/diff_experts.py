"""Post-hoc differentiation test for the five NB11 DPO experts.

This is the repository-local version of the supplied ``diff_experts.py``.
It generates deterministic responses from the untouched base and every DPO
adapter on the same held-out HelpSteer2 prompts, unloads all policy models, and
only then loads the existing project ArmoRM scorer.  It must never be used to
select DPO epochs, checkpoints, or hyperparameters; doing so would invalidate
the RQ2 evaluator firewall.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.armorm_scorer import ArmoRMScorer


ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATASET = "nvidia/HelpSteer2"


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


def select_unique_prompts(
    rows: Sequence[Mapping[str, object]], *, n_prompts: int, seed: int
) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        if prompt and prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    random.Random(seed).shuffle(unique)
    if len(unique) < n_prompts:
        raise ValueError(f"Only {len(unique)} unique prompts; {n_prompts} requested.")
    return unique[:n_prompts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base_revision", default=None)
    parser.add_argument("--expert_root", required=True)
    parser.add_argument("--adapter_subdir", default="adapter")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n_prompts", type=int, default=64)
    parser.add_argument("--prompt_seed", type=int, default=991)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--armorm_precision", choices=("8bit", "bf16"), default="8bit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_prompts < 1 or args.max_new_tokens < 1:
        raise ValueError("n_prompts and max_new_tokens must be positive.")

    import pandas as pd
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Expert generation and ArmoRM evaluation require CUDA.")

    expert_root = Path(args.expert_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, Any]] = {}
    for attribute in ATTRIBUTES:
        manifest_path = expert_root / f"dpo_{attribute}" / "training_manifest.json"
        adapter_path = expert_root / f"dpo_{attribute}" / args.adapter_subdir
        weights_path = adapter_path / "adapter_model.safetensors"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {attribute} expert: expected {manifest_path} and {weights_path}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("completed") or manifest.get("armorm_used_during_training") is not False:
            raise RuntimeError(f"{attribute} manifest does not prove evaluator-free completion.")
        if manifest.get("adapter_model_sha256") != sha256_file(weights_path):
            raise RuntimeError(f"{attribute} adapter SHA256 does not match its manifest.")
        manifests[attribute] = manifest

    shared_fields = (
        "trainer_script_sha256",
        "runtime_versions",
        "base_model_name",
        "base_revision",
        "dataset_name",
        "dataset_revision",
        "dataset_split",
        "selected_pair_count",
        "beta",
        "dpo_disable_dropout",
        "epochs",
        "learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_prompt_length",
        "precision",
        "seed",
        "lora",
    )
    first = manifests[ATTRIBUTES[0]]
    for field in shared_fields:
        values = {json.dumps(manifests[a].get(field), sort_keys=True) for a in ATTRIBUTES}
        if len(values) != 1:
            raise RuntimeError(f"Experts differ on required shared field {field}: {values}")
    if args.base_model_name != first["base_model_name"]:
        raise RuntimeError("Evaluation base model does not match the training manifests.")
    revision = args.base_revision or first["base_revision"]
    if revision != first["base_revision"]:
        raise RuntimeError("Evaluation base revision does not match the training manifests.")
    dataset_revision = args.dataset_revision or first.get("dataset_revision")
    if dataset_revision != first.get("dataset_revision"):
        raise RuntimeError("Evaluation dataset revision does not match the training manifests.")

    dataset = load_dataset(
        args.dataset_name, split=args.split, revision=dataset_revision
    )
    prompts = select_unique_prompts(dataset, n_prompts=args.n_prompts, seed=args.prompt_seed)
    prompt_payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    prompt_sha256 = hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest()

    evaluator_script_sha256 = sha256_file(Path(__file__).resolve())
    armorm_scorer_sha256 = sha256_file(PROJECT_ROOT / "src" / "armorm_scorer.py")
    evaluation_binding = {
        "schema_version": 1,
        "role": "post-hoc differentiation; forbidden for training decisions",
        "evaluator_script_sha256": evaluator_script_sha256,
        "armorm_scorer_sha256": armorm_scorer_sha256,
        "base_model_name": args.base_model_name,
        "base_revision": revision,
        "dataset_name": args.dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_split": args.split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "n_prompts": len(prompts),
        "prompt_seed": args.prompt_seed,
        "prompt_sha256": prompt_sha256,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "armorm_precision": args.armorm_precision,
        "adapter_sha256": {
            attribute: manifests[attribute]["adapter_model_sha256"]
            for attribute in ATTRIBUTES
        },
    }
    evaluation_binding_sha256 = canonical_hash(evaluation_binding)
    report_path = output_dir / "dpo_expert_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("evaluation_binding_sha256") != evaluation_binding_sha256:
            raise RuntimeError(
                "Existing post-hoc report has a different binding. Use a new output "
                "directory instead of silently overwriting consumed evaluations."
            )
        print(f"[skip] post-hoc evaluation already complete: {report_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, revision=revision, use_fast=True
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("The base tokenizer has no chat template.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def generate(adapter_path: Path | None) -> list[str]:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_name, revision=revision, torch_dtype=torch.bfloat16
        ).to("cuda")
        model = PeftModel.from_pretrained(base, adapter_path).eval() if adapter_path else base.eval()
        responses: list[str] = []
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=512,
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer_ids = generated[0, encoded.input_ids.shape[1] :]
            responses.append(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        return responses

    generated: dict[str, list[str]] = {}
    print(f"[generate] untouched base on {len(prompts)} fixed prompts")
    generated["base"] = generate(None)
    for attribute in ATTRIBUTES:
        adapter_path = expert_root / f"dpo_{attribute}" / args.adapter_subdir
        print(f"[generate] dpo_{attribute}")
        generated[attribute] = generate(adapter_path)

    generations_path = output_dir / "dpo_expert_generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as handle:
        for model_name, responses in generated.items():
            for index, (prompt, response) in enumerate(zip(prompts, responses)):
                handle.write(
                    json.dumps(
                        {
                            "model": model_name,
                            "prompt_index": index,
                            "prompt": prompt,
                            "response": response,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    # The supplied script used ArmoRM/URM/SteerLM here.  NB11 deliberately
    # standardizes on the repository's golden-tested ArmoRM implementation.
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
    scorer = ArmoRMScorer(
        dtype="bfloat16", load_in_8bit=args.armorm_precision == "8bit"
    )
    scorer.assert_golden_sample()
    rows: list[dict[str, object]] = []
    model_order = ["base", *ATTRIBUTES]
    matrix = np.zeros((len(model_order), len(ATTRIBUTES)), dtype=np.float64)
    for model_index, model_name in enumerate(model_order):
        all_scores = []
        print(f"[score] {model_name}: {len(prompts)} prompt/response pairs")
        for prompt, response in zip(prompts, generated[model_name]):
            all_scores.append(scorer.score(prompt, response, ATTRIBUTES))
        score_array = np.asarray(all_scores, dtype=np.float64)
        matrix[model_index] = score_array.mean(axis=0)
        for prompt_index, values in enumerate(score_array):
            row: dict[str, object] = {
                "model": model_name,
                "prompt_index": prompt_index,
            }
            row.update({a: float(v) for a, v in zip(ATTRIBUTES, values)})
            rows.append(row)

    score_df = pd.DataFrame(rows)
    score_df.to_csv(output_dir / "dpo_expert_scores.csv", index=False)
    matrix_df = pd.DataFrame(matrix, index=model_order, columns=ATTRIBUTES)
    matrix_df.index.name = "model"
    matrix_df.to_csv(output_dir / "dpo_expert_payoff_matrix.csv")

    expert_matrix = matrix[1:]
    own_column_leaders = [
        bool(int(np.argmax(expert_matrix[:, index])) == index)
        for index in range(len(ATTRIBUTES))
    ]
    own_minus_base = {
        attribute: float(expert_matrix[index, index] - matrix[0, index])
        for index, attribute in enumerate(ATTRIBUTES)
    }
    report = {
        "status": "PASS",
        "role": "post-hoc evaluation only; forbidden for training/checkpoint selection",
        "evaluation_binding": evaluation_binding,
        "evaluation_binding_sha256": evaluation_binding_sha256,
        "evaluator_script_sha256": evaluator_script_sha256,
        "armorm_scorer_sha256": armorm_scorer_sha256,
        "attributes": list(ATTRIBUTES),
        "model_order": model_order,
        "payoff_matrix": matrix.tolist(),
        "own_column_leader": dict(zip(ATTRIBUTES, own_column_leaders)),
        "n_own_column_leaders": int(sum(own_column_leaders)),
        "own_axis_minus_base": own_minus_base,
        "base_model_name": args.base_model_name,
        "base_revision": revision,
        "dataset_name": args.dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_split": args.split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "n_prompts": len(prompts),
        "prompt_seed": args.prompt_seed,
        "prompt_sha256": prompt_sha256,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "scorer": scorer.describe(),
        "adapter_sha256": {
            attribute: manifests[attribute]["adapter_model_sha256"]
            for attribute in ATTRIBUTES
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== DPO expert payoff matrix (raw ArmoRM heads) ===")
    print(matrix_df.round(4).to_string())
    print(f"\nown expert leads its column: {sum(own_column_leaders)}/{len(ATTRIBUTES)}")
    for attribute in ATTRIBUTES:
        print(f"  {attribute:12} own-minus-base={own_minus_base[attribute]:+.4f}")
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
