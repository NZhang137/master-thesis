from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from src import eval_prompts
from src.armorm_scorer import ArmoRMScorer, make_score_prompt_answer


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "13_method_comparison_helpsteer2_dpo_colab.ipynb"


def test_armorm_revision_is_retained_by_factory() -> None:
    revision = "frozen-commit"
    _, scorer = make_score_prompt_answer(revision=revision)
    assert isinstance(scorer, ArmoRMScorer)
    assert scorer.revision == revision
    assert scorer.describe()["revision"] == revision


def test_prompt_builder_passes_dataset_revision_and_prefix(tmp_path, monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_candidates(split, *, dataset_name, revision):
        observed.update(split=split, dataset_name=dataset_name, revision=revision)
        return [f"This is deterministic candidate prompt number {index:03d}." for index in range(8)]

    monkeypatch.setattr(eval_prompts, "load_candidate_prompts", fake_candidates)
    output = tmp_path / "prompts.jsonl"
    summary = eval_prompts.build_eval_prompt_file(
        output,
        n=3,
        seed=17,
        dataset_name="example/dataset",
        dataset_revision="dataset-commit",
        prompt_id_prefix="nb13",
        project_root=tmp_path,
        require_names=(),
        allow_missing_exclusions=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert observed == {
        "split": "validation",
        "dataset_name": "example/dataset",
        "revision": "dataset-commit",
    }
    assert all(row["prompt_id"].startswith("nb13_") for row in rows)
    assert summary["dataset_revision"] == "dataset-commit"


def test_nb13_is_parseable_and_fail_closed_by_default() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 51

    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'RUN_REWARD_COLLECTION = False' in all_source
    assert 'PREREG_CONFIRM = False' in all_source
    assert 'RHO_GRID = [0.0, 0.1, 0.2, 0.5]' in all_source
    assert 'C_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]' in all_source
    assert 'ALPHA_GRID = [0.0, 0.5, 1.0, 2.0]' in all_source
    assert 'EPS_GRID = [0.01, 0.02, 0.05, 0.10]' in all_source
    assert 'for rho in RHO_GRID:' in all_source
    assert 'RHO_AVG' not in all_source
    assert "97f41c1f2290bd1074a261577de5c273c3754d89e311f713d90d7d7cc2a5bbf3" in all_source
    assert "3864d363dbff91fdeda62029a16b0da3b3436fe3557d42cb75739c75649ff51e" in all_source
    assert 'REGIME = "helpsteer2_dpo_nb11"' in all_source
    assert 'REGIME = "rs_ppo"' not in all_source
    assert "results/rs_ppo_armorm_circular" not in all_source

    for position, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs", []) == []
        transformed = []
        for line in "".join(cell.get("source", [])).splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                transformed.append(line[: len(line) - len(stripped)] + "pass")
            else:
                transformed.append(line)
        ast.parse("\n".join(transformed), filename=f"NB13 cell {position}")


def test_prompt_builder_selection_is_deterministic(tmp_path, monkeypatch) -> None:
    candidates = [f"A sufficiently long deterministic prompt {index:03d}." for index in range(25)]
    monkeypatch.setattr(
        eval_prompts,
        "load_candidate_prompts",
        lambda split, *, dataset_name, revision: list(candidates),
    )
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    kwargs = dict(
        n=8,
        seed=1313,
        dataset_revision="fixed",
        prompt_id_prefix="nb13",
        project_root=tmp_path,
        require_names=(),
        allow_missing_exclusions=True,
    )
    eval_prompts.build_eval_prompt_file(left, **kwargs)
    eval_prompts.build_eval_prompt_file(right, **kwargs)
    validated = eval_prompts.build_eval_prompt_file(left, **kwargs)
    left_prompts = [json.loads(line)["prompt"] for line in left.read_text().splitlines()]
    right_prompts = [json.loads(line)["prompt"] for line in right.read_text().splitlines()]
    assert left_prompts == right_prompts
    assert len(np.unique(left_prompts)) == 8
    assert validated["created"] is False
    assert validated["n_candidates_after_exclusion"] == len(candidates)
