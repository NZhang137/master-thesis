from __future__ import annotations

import sys
import types

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


def test_steerlm_uses_lightweight_triton_http_protocol(monkeypatch) -> None:
    captured_inputs = []
    client_state = {"closed": False, "calls": 0}

    class FakeInferInput:
        def __init__(self, name, shape, datatype):
            assert name == "sentences"
            assert shape == [1, 1]
            assert datatype == "BYTES"
            self.data = None

        def set_data_from_numpy(self, data, binary_data):
            assert binary_data is True
            self.data = data.copy()
            captured_inputs.append(self.data)

    class FakeResult:
        def __init__(self, index):
            self.index = index

        def as_numpy(self, name):
            if name == "rewards":
                return (np.arange(9, dtype=float) + 10 * self.index)[None, :]
            if name == "exceeded":
                return np.asarray([[False]])
            return None

    class FakeRequest:
        def __init__(self, index):
            self.index = index

        def get_result(self):
            return FakeResult(self.index)

    class FakeClient:
        def __init__(self, *, url, verbose, concurrency):
            assert url == "example.invalid:1424"
            assert verbose is False
            assert concurrency == 2

        @staticmethod
        def is_server_live():
            return True

        @staticmethod
        def is_model_ready(model_name):
            return model_name == "reward_model"

        def async_infer(self, *, model_name, inputs):
            assert model_name == "reward_model"
            assert len(inputs) == 1
            index = client_state["calls"]
            client_state["calls"] += 1
            return FakeRequest(index)

        @staticmethod
        def close():
            client_state["closed"] = True

    http_module = types.ModuleType("tritonclient.http")
    http_module.InferenceServerClient = FakeClient
    http_module.InferInput = FakeInferInput
    triton_module = types.ModuleType("tritonclient")
    triton_module.__path__ = []
    triton_module.http = http_module
    monkeypatch.setitem(sys.modules, "tritonclient", triton_module)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http_module)

    expected = f"{MODEL_NAME}@{MODEL_REVISION}:sha256={CHECKPOINT_SHA256}"
    scorer = SteerLMRemoteScorer(
        host="example.invalid", operator_attestation=expected
    )
    result = scorer._raw_batch(["grüße", "second"])

    assert result.tolist() == [
        list(np.arange(9, dtype=float)),
        list(np.arange(9, dtype=float) + 10),
    ]
    assert captured_inputs[0].dtype == object
    assert captured_inputs[0][0, 0] == "grüße".encode("utf-8")
    assert client_state == {"closed": True, "calls": 2}


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
