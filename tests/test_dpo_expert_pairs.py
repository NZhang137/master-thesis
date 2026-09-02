from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "0_dpo_expert.py"
SPEC = importlib.util.spec_from_file_location("dpo_expert_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(prompt: str, response: str, helpfulness: int) -> dict[str, object]:
    return {
        "prompt": prompt,
        "response": response,
        "helpfulness": helpfulness,
        "correctness": 0,
        "coherence": 0,
        "complexity": 0,
        "verbosity": 0,
    }


def test_pairs_are_within_prompt_and_oriented_by_target_rating() -> None:
    rows = [
        _row("p1", "low", 1),
        _row("p2", "other-low", 0),
        _row("p1", "high", 4),
        _row("p2", "other-high", 3),
    ]
    pairs = MODULE.build_pairs_from_rows(rows, "helpfulness")
    assert len(pairs) == 2
    assert {(p["raw_prompt"], p["chosen"], p["rejected"]) for p in pairs} == {
        ("p1", "high", "low"),
        ("p2", "other-high", "other-low"),
    }


def test_target_ties_and_identical_responses_are_discarded() -> None:
    rows = [
        _row("tie", "a", 2),
        _row("tie", "b", 2),
        _row("same", "answer", 1),
        _row("same", "answer", 4),
    ]
    assert MODULE.build_pairs_from_rows(rows, "helpfulness") == []


def test_selection_is_exact_and_reproducible() -> None:
    pairs = [
        {"pair_id": str(i), "raw_prompt": str(i), "chosen": "x", "rejected": "y"}
        for i in range(10)
    ]
    first = MODULE.select_pairs(pairs, seed=7, max_pairs=5)
    second = MODULE.select_pairs(pairs, seed=7, max_pairs=5)
    assert first == second
    assert len(first) == 5


def test_training_script_imports_no_reward_model_module() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not any(
        "armorm" in module.lower() or "reward_model" in module.lower()
        for module in imported_modules
    )
