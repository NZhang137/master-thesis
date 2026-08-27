"""One-shot repository preparation for NB10 Phase B.

Applies every source change Phase B needs, in one idempotent pass, so the only
thing left to execute is the notebook itself.

  1. signed-zero normalization in `lambda_key` and `coefficient_key`
  2. `collect_reward_tensor` appended to src/proxy_validation.py (per-prompt rewards,
     cache bound to the run via binding_sha256)
  3. Mean Rank / Selection Regret / paired bootstrap with p-value, appended to
     src/metrics.py
  4. src/armorm_scorer.py installed (8-bit path sets the bf16 compute dtype, so
     "8-bit" here is the same numerical path as in training)
  5. src/eval_prompts.py installed (fail-closed prompt exclusion)
  6. preference vocabulary CHECKED, never rewritten: renaming a preference is a
     pre-registration change, not a code fix
  7. every change verified by import and self-test

Why this is a script and not a notebook cell
--------------------------------------------
NB10 cell 1 pulls the repository from GitHub. If the notebook patched `src/` at
runtime, the committed source and the executed source would differ and the run
would not be reproducible from the commit alone - which is exactly what the
pre-registration hash is supposed to rule out. Run this once, commit, then the
notebook is the only thing that executes.

Usage
-----
    python prepare_nb10.py                 # from the repository root
    python prepare_nb10.py --from ./inbox  # if the new files live elsewhere
    python prepare_nb10.py --check         # report only, change nothing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SIGNED_ZERO_PATCHES = [
    (
        "src/lambda_utils.py",
        "    arr = np.round(np.asarray(lmbda, dtype=np.float64), decimals)\n",
        "    # `+ 0.0` normalizes -0.0 to +0.0 so numerically identical vectors\n"
        "    # cannot produce different byte patterns (cache-resume correctness).\n"
        "    arr = np.round(np.asarray(lmbda, dtype=np.float64), decimals) + 0.0\n",
    ),
    (
        "src/proxy_validation.py",
        '    return "|".join(f"{float(value):.8f}" for value in coefficient)\n',
        "    # Round FIRST, then normalize the sign: `+ 0.0` turns -0.0 into +0.0,\n"
        '    # but leaves -1e-12 untouched, which still formats as "-0.00000000".\n'
        '    return "|".join(f"{round(float(value), 8) + 0.0:.8f}" for value in coefficient)\n',
    ),
]

APPENDS = [
    ("collect_reward_tensor_addition.py", "src/proxy_validation.py", "def collect_reward_tensor("),
    ("metrics_additions.py", "src/metrics.py", "def paired_bootstrap_ci("),
]

COPIES = [
    ("armorm_scorer.py", "src/armorm_scorer.py"),
    ("eval_prompts.py", "src/eval_prompts.py"),
]

# The rename uniform -> balanced was already applied by hand. This is a CHECK, not an
# edit: prepare_nb10 must not silently rewrite preference names.
EXPECTED_ABSENT = [("src/preferences.py", '"uniform"')]
EXPECTED_PRESENT = [("src/preferences.py", '"balanced"')]


class Aborted(RuntimeError):
    """Raised when a target does not match exactly, so nothing is half-applied."""


def _read(path: Path) -> str:
    if not path.is_file():
        raise Aborted(f"{path} not found. Run this from the repository root.")
    return path.read_text(encoding="utf-8")


def apply_signed_zero(root: Path, check: bool) -> list[str]:
    """Normalize negative zero in both cache-key functions."""
    notes = []
    for rel, old, new in SIGNED_ZERO_PATCHES:
        path = root / rel
        text = _read(path)
        if new in text:
            notes.append(f"skip   {rel}: signed-zero already normalized")
            continue
        count = text.count(old)
        if count != 1:
            raise Aborted(f"{rel}: expected 1 occurrence of the target, found {count}.")
        if not check:
            path.write_text(text.replace(old, new), encoding="utf-8")
        notes.append(f"patch  {rel}: signed-zero normalized")
    return notes


def apply_appends(root: Path, source_dir: Path, check: bool) -> list[str]:
    """Append the new functions to their modules, once."""
    notes = []
    for source_name, rel, marker in APPENDS:
        source = source_dir / source_name
        if not source.is_file():
            raise Aborted(f"{source} not found. Pass --from with the directory holding it.")
        path = root / rel
        text = _read(path)
        if marker in text:
            notes.append(f"skip   {rel}: {marker.rstrip('(')} already present")
            continue
        if not check:
            addition = source.read_text(encoding="utf-8")
            path.write_text(text.rstrip("\n") + "\n\n\n" + addition.lstrip("\n"), encoding="utf-8")
        notes.append(f"append {rel}: + {marker.rstrip('(')}")
    return notes


def apply_copies(root: Path, source_dir: Path, check: bool) -> list[str]:
    """Install the new standalone files."""
    notes = []
    for source_name, rel in COPIES:
        source = source_dir / source_name
        if not source.is_file():
            raise Aborted(f"{source} not found. Pass --from with the directory holding it.")
        target = root / rel
        if target.is_file() and target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8"):
            notes.append(f"skip   {rel}: identical")
            continue
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        notes.append(f"write  {rel}")
    return notes


def apply_renames(root: Path, check: bool) -> list[str]:
    """Verify the preference vocabulary; never rewrite it.

    Renaming preference keys is a pre-registration change, not a code fix, so this
    script only checks that the rename already happened and reports if it did not.
    """
    notes = []
    for rel, token in EXPECTED_PRESENT:
        if token not in _read(root / rel):
            raise Aborted(f"{rel}: expected {token} to be present. Rename it deliberately first.")
        notes.append(f"check  {rel}: {token} present")
    for rel, token in EXPECTED_ABSENT:
        if token in _read(root / rel):
            raise Aborted(f"{rel}: {token} is still present. It should have been renamed.")
        notes.append(f"check  {rel}: {token} absent")
    return notes


def verify(root: Path) -> list[str]:
    """Import everything that changed and run the cheap self-tests."""
    import numpy as np

    sys.path.insert(0, str(root))
    for module in [m for m in list(sys.modules) if m.startswith("src.")]:
        del sys.modules[module]

    from src.lambda_utils import lambda_key
    from src.metrics import mean_rank, paired_bootstrap_ci, selection_regret
    from src.preferences import PREFERENCES, preference_regime
    from src.proxy_validation import coefficient_key, collect_reward_tensor  # noqa: F401
    from src.armorm_scorer import ArmoRMScorer

    notes = []

    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, -1e-12, 0.0, 0.0, 0.0])
    if lambda_key(a) != lambda_key(b) or coefficient_key(a) != coefficient_key(b):
        raise Aborted("signed-zero normalization did not take effect.")
    notes.append("ok  keys agree for +0.0 and -1e-12")

    if "balanced" not in PREFERENCES or "uniform" in PREFERENCES:
        raise Aborted("preferences.py still exposes 'uniform'.")
    if preference_regime("balanced") != "balanced":
        raise Aborted("preference_regime does not return 'balanced'.")
    from src.eval_prompts import build_eval_prompt_file  # noqa: F401
    notes.append(f"ok  preferences: {len(PREFERENCES)} entries, 'balanced' present")

    ranks = mean_rank({"a": [1.0, 2.0], "b": [2.0, 1.0]})
    if not np.isclose(ranks["a"], 1.5) or not np.isclose(ranks["b"], 1.5):
        raise Aborted("mean_rank is wrong.")
    if not np.isclose(selection_regret(1.0, [1.0, 2.5]), 1.5):
        raise Aborted("selection_regret is wrong.")
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.1, (40, 5))
    null = paired_bootstrap_ci(base, base, np.full(5, 0.2), n_boot=2000)
    if null["excludes_zero"] or not np.isclose(null["delta_u_p"], 0.0):
        raise Aborted("paired_bootstrap_ci fails the null case.")
    notes.append("ok  metrics: mean_rank, selection_regret, bootstrap null case")

    scorer = ArmoRMScorer(dtype="bfloat16")
    if scorer.describe()["precision"] != "bfloat16" or scorer.describe()["batch_size"] != 1:
        raise Aborted("ArmoRMScorer defaults are wrong.")
    notes.append("ok  armorm_scorer imports, bf16, batch_size=1")
    return notes


def main() -> int:
    """Apply everything, verify, and print the commit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--from", dest="source", default=None,
                        help="Directory holding the new files. Default: this script's directory.")
    parser.add_argument("--check", action="store_true", help="Report only, change nothing.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source_dir = Path(args.source).resolve() if args.source else Path(__file__).resolve().parent

    try:
        notes = []
        notes += apply_signed_zero(root, args.check)
        notes += apply_appends(root, source_dir, args.check)
        notes += apply_copies(root, source_dir, args.check)
        notes += apply_renames(root, args.check)
    except Aborted as error:
        print(f"ABORT: {error}")
        print("Nothing was written past this point; fix and re-run.")
        return 1

    for note in notes:
        print(note)

    if args.check:
        print("\n--check: no files were modified.")
        return 0

    print()
    try:
        for note in verify(root):
            print(note)
    except Aborted as error:
        print(f"VERIFICATION FAILED: {error}")
        return 1

    notebook_path = root / "notebooks" / "10_method_comparison_colab.ipynb"
    notebook_is_patched = (
        notebook_path.is_file()
        and '"BINDING_SHA256 = ' in notebook_path.read_text(encoding="utf-8")
    )
    print("\nStill to do by hand, in this order:")
    if notebook_is_patched:
        print("  1. NB10 is already patched under its canonical notebook path")
    else:
        print("  1. python patch_nb10.py --notebook notebooks/10_method_comparison_colab.ipynb")
    print("  2. review git status, then commit the NB10 source and notebook changes")
    print("  3. git push")
    print("  4. run NB10 - it is now the only notebook to execute")
    try:
        subprocess.run(["git", "-C", str(root), "status", "--short"], check=False)
    except FileNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
