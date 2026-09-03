from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "0_dpo_expert_one_pair.py"
SPEC = importlib.util.spec_from_file_location("dpo_expert_one_pair_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pair(
    pair_id: str,
    prompt: str,
    chosen_rating: int,
    rejected_rating: int,
    other_gap: int = 0,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "raw_prompt": prompt,
        "chosen": f"chosen-{pair_id}",
        "rejected": f"rejected-{pair_id}",
        "target_rating_gap": chosen_rating - rejected_rating,
        "chosen_target_rating": chosen_rating,
        "other_axes_abs_gap_sum": other_gap,
    }


def test_equal_gap_prefers_higher_chosen_rating_across_prompts() -> None:
    pairs = [
        _pair("2-0", "prompt-a", 2, 0),
        _pair("3-1", "prompt-b", 3, 1),
        _pair("4-2", "prompt-c", 4, 2),
    ]
    selected = MODULE.select_pairs(pairs, seed=137, max_pairs=1)
    assert selected[0]["pair_id"] == "4-2"


def test_target_gap_precedes_chosen_rating_and_other_axes_are_final_tie_break() -> None:
    pairs = [
        _pair("4-2", "prompt", 4, 2, other_gap=0),
        _pair("3-0-wide-other", "prompt", 3, 0, other_gap=4),
        _pair("3-0-narrow-other", "prompt", 3, 0, other_gap=1),
    ]
    selected = MODULE.select_pairs(pairs, seed=137, max_pairs=1)
    assert selected[0]["pair_id"] == "3-0-narrow-other"


def test_selection_returns_exactly_one_pair_per_prompt() -> None:
    pairs = [
        _pair("a1", "a", 4, 1),
        _pair("a2", "a", 4, 2),
        _pair("b1", "b", 4, 0),
        _pair("b2", "b", 3, 0),
        _pair("c1", "c", 4, 3),
    ]
    first = MODULE.select_pairs(pairs, seed=137, max_pairs=3)
    second = MODULE.select_pairs(pairs, seed=137, max_pairs=3)
    assert first == second
    assert len(first) == 3
    assert len({pair["raw_prompt"] for pair in first}) == 3
