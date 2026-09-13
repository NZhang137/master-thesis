from __future__ import annotations

import numpy as np
import pytest

from src.cert_floor_policy import should_exclude_cert_from_phase_b
from src.steerlm_scorer import (
    ALL_ATTRIBUTES,
    CHECKPOINT_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    SteerLMRemoteScorer,
    format_conversation,
    select_helpsteer_attributes,
)
from src.urm_scorer import DEFAULT_REVISION, URMScorer, attribute_indices


def test_urm_order_and_frozen_revision() -> None:
    assert attribute_indices(
        ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    ) == [0, 1, 2, 3, 4]
    scorer = URMScorer()
    assert scorer.describe()["revision"] == DEFAULT_REVISION
    assert "HelpSteer2" in scorer.describe()["independence_role"]


def test_urm_unknown_attribute_fails_closed() -> None:
    with pytest.raises(KeyError):
        attribute_indices(["quality"])


def test_steerlm_format_and_attribute_projection() -> None:
    text = format_conversation("question", "answer")
    assert text.startswith("System\nA chat between")
    assert text.endswith("User\nquestion\nAssistant\nanswer\n")
    raw = np.arange(len(ALL_ATTRIBUTES), dtype=float)
    observed = select_helpsteer_attributes(
        raw, ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    )
    assert observed.tolist() == [4.0, 5.0, 6.0, 7.0, 8.0]


def test_steerlm_requires_exact_operator_attestation() -> None:
    expected = f"{MODEL_NAME}@{MODEL_REVISION}:sha256={CHECKPOINT_SHA256}"
    scorer = SteerLMRemoteScorer(
        host="example.invalid", operator_attestation=expected
    )
    assert scorer.describe()["operator_attestation"] == expected
    with pytest.raises(RuntimeError, match="Expected"):
        SteerLMRemoteScorer(host="example.invalid", operator_attestation="wrong")


def test_steerlm_score_many_preserves_rows(monkeypatch) -> None:
    expected = f"{MODEL_NAME}@{MODEL_REVISION}:sha256={CHECKPOINT_SHA256}"
    scorer = SteerLMRemoteScorer(host="example.invalid", operator_attestation=expected)
    scorer._health_checked = True
    monkeypatch.setattr(
        scorer,
        "_raw_batch",
        lambda sentences: np.vstack([np.arange(9), np.arange(9) + 10.0]),
    )
    result = scorer.score_many(
        ["q1", "q2"], ["a1", "a2"],
        ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
    )
    assert result.tolist() == [
        [4.0, 5.0, 6.0, 7.0, 8.0],
        [14.0, 15.0, 16.0, 17.0, 18.0],
    ]


def test_cert_is_excluded_only_when_every_phase_b_row_collapses_to_p() -> None:
    rows = [
        {"p_name": "a", "floor_lp_collapsed": True, "lambda_minus_p_l2": 0.0},
        {"p_name": "b", "floor_lp_collapsed": True, "lambda_minus_p_l2": 1e-8},
        {"p_name": "phase_a_only", "floor_lp_collapsed": False, "lambda_minus_p_l2": 1.0},
    ]
    assert should_exclude_cert_from_phase_b(rows, {"a", "b"}, tolerance=2e-6)
    rows[1]["lambda_minus_p_l2"] = 1e-3
    assert not should_exclude_cert_from_phase_b(rows, {"a", "b"}, tolerance=2e-6)


def test_cert_exclusion_rejects_incomplete_audit() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        should_exclude_cert_from_phase_b(
            [{"p_name": "a", "floor_lp_collapsed": True, "lambda_minus_p_l2": 0.0}],
            {"a", "b"},
            tolerance=2e-6,
        )
