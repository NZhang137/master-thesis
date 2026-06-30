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


def test_low_overlap_selection_prefers_unmarked_rows(monkeypatch):
    rows = [
        _row(0, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(1, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
        _row(2, helpfulness=4, correctness=4, coherence=4, complexity=2, verbosity=3),
        _row(3, helpfulness=4, correctness=4, coherence=4, complexity=1, verbosity=3),
        _row(4, helpfulness=4, correctness=4, coherence=4, complexity=2, verbosity=2),
    ]
    monkeypatch.setattr(helpsteer2_utils, "load_helpsteer2_split", lambda split: rows)

    selected, summaries = helpsteer2_utils.make_low_overlap_attribute_training_texts(
        attributes=["complexity", "verbosity", "helpfulness"],
        split="train",
        max_examples=2,
        attribute_min_ratings={
            "complexity": 2,
            "verbosity": 3,
            "helpfulness": 4,
        },
        selection_order=["complexity", "verbosity", "helpfulness"],
    )

    assert [text.splitlines()[0] for text in selected["complexity"]] == [
        "Human: Prompt 0",
        "Human: Prompt 1",
    ]
    assert [text.splitlines()[0] for text in selected["verbosity"]] == [
        "Human: Prompt 2",
        "Human: Prompt 3",
    ]
    assert summaries["verbosity"]["prior_usage_counts"] == {0: 2}
    assert summaries["helpfulness"]["prior_usage_counts"] == {0: 1, 1: 1}


def test_low_overlap_selection_exhausts_rating_before_lower_rating(monkeypatch):
    rows = [
        _row(0, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(1, helpfulness=4, correctness=4, coherence=4, complexity=4, verbosity=3),
        _row(2, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
        _row(3, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
        _row(4, helpfulness=4, correctness=4, coherence=4, complexity=3, verbosity=3),
    ]
    monkeypatch.setattr(helpsteer2_utils, "load_helpsteer2_split", lambda split: rows)

    selected, summaries = helpsteer2_utils.make_low_overlap_attribute_training_texts(
        attributes=["verbosity", "complexity"],
        split="train",
        max_examples=3,
        attribute_min_ratings={
            "verbosity": 3,
            "complexity": 3,
        },
        selection_order=["verbosity", "complexity"],
    )

    assert [text.splitlines()[0] for text in selected["verbosity"]] == [
        "Human: Prompt 0",
        "Human: Prompt 1",
        "Human: Prompt 2",
    ]
    assert [text.splitlines()[0] for text in selected["complexity"]] == [
        "Human: Prompt 0",
        "Human: Prompt 1",
        "Human: Prompt 3",
    ]
    assert summaries["complexity"]["selected_rating_counts"] == {3: 1, 4: 2}
    assert summaries["complexity"]["selected_rating_prior_usage_counts"] == {
        "rating_3_prior_use_0": 1,
        "rating_4_prior_use_1": 2,
    }
