"""Build the fixed evaluation prompt file for NB10 Phase B.

Two properties matter and both are enforced rather than asserted in prose:

Unseen by training. Prompts are drawn from the HelpSteer2 VALIDATION split. The
RS-PPO adapters consumed the TRAIN split during their PPO steps, so evaluating
on train would be evaluating on training data.

Unseen by earlier evaluation. Fail-closed: every file named in `require_paths`
MUST be found, or the build aborts. Silently drawing prompts because an exclusion
file was not located is exactly the failure this guard exists to prevent. After
selection, disjointness is asserted rather than assumed.

Selection is seeded, so the file regenerates bit-for-bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


MIN_CHARS = 40
MAX_CHARS = 1200
REQUIRED_FIELDS = ("prompt_id", "category", "prompt", "notes")


def _categorize(prompt: str) -> str:
    """Assign a coarse category, used only to report topical balance."""
    import re

    text = prompt.lower()
    if re.search(r"\b(code|function|python|javascript|sql|bug|compile)\b", text):
        return "code"
    if re.search(r"\b(write|draft|story|poem|essay|email)\b", text):
        return "writing"
    if re.search(r"\b(why|how|what|explain|describe)\b", text):
        return "explanation"
    return "general"


# Prompt files known to have been used by earlier notebooks. If the project layout
# changes, update this list deliberately - do not let a missing file pass silently.
KNOWN_EARLIER_PROMPT_FILES = (
    "proxy_validation_fixed_prompts.jsonl",
    "confirmatory_fixed_prompts.jsonl",
    "helpsteer2_fixed_prompts.jsonl",
)


def find_prompt_files(root: Path, names: Sequence[str]) -> tuple[dict[str, Path], list[str]]:
    """Locate named prompt files anywhere under `root`; report which are missing."""
    found: dict[str, Path] = {}
    for name in names:
        hits = sorted(Path(root).rglob(name))
        if len(hits) > 1:
            raise RuntimeError(
                f"Fail-closed: {name} is ambiguous; found {len(hits)} copies:\n"
                + "\n".join(f"  {path}" for path in hits)
            )
        if hits:
            found[name] = hits[0]
    missing = [n for n in names if n not in found]
    return found, missing


def validate_prompt_file(path: Path, expected_n: int | None = None) -> dict[str, Any]:
    """Re-validate an existing prompt file: fields, uniqueness, count, provenance."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty.")
    for position, row in enumerate(rows, start=1):
        missing = [f for f in REQUIRED_FIELDS if f not in row or not str(row[f]).strip()]
        if missing:
            raise ValueError(f"{path} line {position}: missing or empty {missing}.")
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate prompt_id values.")
    texts = [r["prompt"] for r in rows]
    if len(set(texts)) != len(texts):
        raise ValueError(f"{path}: duplicate prompt texts.")
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"{path}: {len(rows)} prompts, expected {expected_n}.")
    return {"path": str(path), "n": len(rows), "created": False, "validated": True,
            "provenance": rows[0]["notes"]}


def collect_excluded_texts(paths: Iterable[Path]) -> tuple[set[str], list[str]]:
    """Return prompt texts used by earlier prompt files, and which files were read."""
    excluded: set[str] = set()
    read: list[str] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        read.append(str(path))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("prompt")
            if isinstance(text, str) and text.strip():
                excluded.add(text.strip())
    return excluded, read


def load_candidate_prompts(split: str = "validation") -> list[str]:
    """Load deduplicated, length-filtered user prompts from a HelpSteer2 split."""
    from datasets import load_dataset

    dataset = load_dataset("nvidia/HelpSteer2", split=split)
    seen: set[str] = set()
    prompts: list[str] = []
    for row in dataset:
        prompt = str(row["prompt"]).strip()
        if not MIN_CHARS <= len(prompt) <= MAX_CHARS or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    if not prompts:
        raise RuntimeError(f"No usable prompts in the {split} split.")
    return prompts


def ensure_nb06_prompt_files(
    project_root: str | Path = ".",
    *,
    dataset_name: str = "nvidia/HelpSteer2",
    split: str = "validation",
    seed: int = 137,
    n_per_set: int = 80,
) -> dict[str, Any]:
    """Reconstruct missing NB06 prompt sets with NB06.1's frozen selection rule.

    The original proxy set is the first 80 entries of the seeded permutation;
    the confirmatory set is the following 80. Existing files are checked against
    that deterministic selection and are never overwritten.
    """
    from datasets import load_dataset
    import numpy as np

    root = Path(project_root)
    names_and_offsets = {
        "proxy_validation_fixed_prompts.jsonl": 0,
        "confirmatory_fixed_prompts.jsonl": n_per_set,
    }
    found, missing = find_prompt_files(root, tuple(names_and_offsets))
    if not missing:
        return {"created": [], "found": {name: str(path) for name, path in found.items()}}

    dataset = load_dataset(dataset_name, split=split)
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    for index, example in enumerate(dataset):
        prompt = str(example.get("prompt", "")).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        candidates.append({"prompt_id": f"helpsteer2_{split}_{index}", "prompt": prompt})
    required = 2 * n_per_set
    if len(candidates) < required:
        raise RuntimeError(f"Need {required} unique {split} prompts, found {len(candidates)}.")
    order = np.random.default_rng(seed).permutation(len(candidates))

    destination_root = root / "results" / "tinyllama_helpsteer2_proxy_validation"
    created: list[str] = []
    for name, offset in names_and_offsets.items():
        expected = [candidates[int(index)] for index in order[offset : offset + n_per_set]]
        if name in found:
            actual = [
                json.loads(line)
                for line in found[name].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            actual_core = [
                {"prompt_id": str(row.get("prompt_id", "")), "prompt": str(row.get("prompt", ""))}
                for row in actual
            ]
            if actual_core != expected:
                raise RuntimeError(
                    f"Existing {found[name]} does not match the frozen NB06.1 "
                    f"selection (split={split}, seed={seed}, offset={offset})."
                )
            continue

        destination = destination_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as output_file:
            for row in expected:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        created.append(str(destination))

    resolved, still_missing = find_prompt_files(root, tuple(names_and_offsets))
    if still_missing:
        raise RuntimeError(f"Prompt reconstruction incomplete: {still_missing}")
    proxy_texts, _ = collect_excluded_texts([resolved["proxy_validation_fixed_prompts.jsonl"]])
    confirm_texts, _ = collect_excluded_texts([resolved["confirmatory_fixed_prompts.jsonl"]])
    overlap = proxy_texts & confirm_texts
    if overlap:
        raise RuntimeError(f"Reconstructed NB06 prompt sets overlap in {len(overlap)} texts.")
    return {"created": created, "found": {name: str(path) for name, path in resolved.items()}}


def build_eval_prompt_file(
    out_path: str | Path,
    *,
    n: int = 80,
    seed: int = 137,
    split: str = "validation",
    project_root: str | Path = ".",
    require_names: Sequence[str] = KNOWN_EARLIER_PROMPT_FILES,
    extra_exclude_paths: Sequence[str | Path] = (),
    allow_missing_exclusions: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the prompt JSONL and return a summary for the pre-registration.

    Refuses to overwrite an existing file unless `overwrite=True`: once Phase B
    has started, replacing the prompt file silently would invalidate the frozen
    pre-registration.
    """
    import numpy as np

    out_path = Path(out_path)
    found, missing = find_prompt_files(Path(project_root), list(require_names))
    if missing and not allow_missing_exclusions:
        raise FileNotFoundError(
            "Fail-closed: these earlier prompt files were expected but not found:\n"
            + "\n".join("  " + name for name in missing)
            + "\nWithout them, disjointness from earlier evaluations cannot be established. "
            "Correct the paths, or pass allow_missing_exclusions=True and report the gap "
            "explicitly instead of claiming novelty."
        )

    exclude_paths = list(found.values()) + [Path(p) for p in extra_exclude_paths]
    excluded, read_files = collect_excluded_texts(exclude_paths)

    if out_path.is_file() and not overwrite:
        summary = validate_prompt_file(out_path, expected_n=n)
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        overlap = {row["prompt"].strip() for row in rows} & excluded
        if overlap:
            raise AssertionError(
                f"{out_path}: {len(overlap)} existing prompts also appear in an "
                "earlier prompt file. Use a new output path or rebuild deliberately."
            )
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        summary.update({
            "split": split,
            "seed": seed,
            "n_excluded_texts": len(excluded),
            "exclusion_files_read": read_files,
            "exclusion_files_missing": missing,
            "disjointness_verified": bool(read_files) and not missing,
            "categories": counts,
        })
        return summary

    candidates = [p for p in load_candidate_prompts(split) if p not in excluded]
    if n > len(candidates):
        raise ValueError(f"Requested {n} prompts, only {len(candidates)} remain after exclusion.")

    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(candidates), size=n, replace=False))

    provenance = f"HelpSteer2 {split} split, seed={seed}, {MIN_CHARS}<=len<={MAX_CHARS}"
    if read_files:
        provenance += f"; disjoint from {len(excluded)} texts in {len(read_files)} earlier prompt file(s)"
    else:
        provenance += "; NO earlier prompt file found, disjointness NOT established"

    rows = [
        {
            "prompt_id": f"nb10_{position:03d}",
            "category": _categorize(candidates[int(index)]),
            "prompt": candidates[int(index)],
            "notes": provenance,
        }
        for position, index in enumerate(indices, start=1)
    ]

    # Assert, do not assume: the drawn texts must not intersect the excluded set.
    drawn = {row["prompt"] for row in rows}
    overlap = drawn & excluded
    if overlap:
        raise AssertionError(f"{len(overlap)} drawn prompts also appear in an earlier prompt file.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return {
        "path": str(out_path),
        "n": len(rows),
        "created": True,
        "split": split,
        "seed": seed,
        "n_candidates_after_exclusion": len(candidates),
        "n_excluded_texts": len(excluded),
        "exclusion_files_read": read_files,
        "exclusion_files_missing": missing,
        "disjointness_verified": bool(read_files) and not missing,
        "categories": counts,
        "provenance": provenance,
    }
