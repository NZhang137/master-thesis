from src import helpsteer2_utils


def _row(index, *, helpfulness, correctness, coherence, complexity, verbosity):
    return {
        "prompt": f"Prompt {index}",
        "response": f"Response {index}",
        "helpfulness": helpfulness,
        "correctness": correctness,
        "coherence": coherence,
        "complexity": complexity,
        "verbosity": verbosity,
    }


def _prompts(texts):
    return [text.splitlines()[0] for text in texts]


def test_independent_selection_uses_rating_four_before_three(monkeypatch):
    rows = [
        _row(0, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(1, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(2, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
        _row(3, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
        _row(4, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
    ]
    monkeypatch.setattr(helpsteer2_utils, "load_helpsteer2_split", lambda split: rows)

    selected, summaries = helpsteer2_utils.make_independent_attribute_training_texts(
        attributes=["complexity"],
        split="train",
        max_examples=3,
        seed=1,
    )

    assert _prompts(selected["complexity"]) == [
        "Human: Prompt 1",
        "Human: Prompt 0",
        "Human: Prompt 3",
    ]
    assert summaries["complexity"]["selected_rating_counts"] == {3: 1, 4: 2}
    assert "prior_usage_counts" not in summaries["complexity"]


def test_independent_selection_allows_attribute_overlap(monkeypatch):
    rows = [
        _row(0, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(1, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(2, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
        _row(3, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
    ]
    monkeypatch.setattr(helpsteer2_utils, "load_helpsteer2_split", lambda split: rows)

    selected, summaries = helpsteer2_utils.make_independent_attribute_training_texts(
        attributes=["complexity", "verbosity"],
        split="train",
        max_examples=2,
        seed=1,
    )

    assert _prompts(selected["complexity"]) == [
        "Human: Prompt 1",
        "Human: Prompt 0",
    ]
    assert _prompts(selected["verbosity"]) == [
        "Human: Prompt 3",
        "Human: Prompt 0",
    ]
    report = helpsteer2_utils.compute_prompt_overlap_report(
        summaries,
        ["complexity", "verbosity"],
    )
    assert report["absolute_matrix"]["complexity"]["verbosity"] == 1
    assert report["percent_matrix"]["complexity"]["verbosity"] == 50.0
