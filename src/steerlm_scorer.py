"""Client for NVIDIA's official NeMo SteerLM reward-model server.

``nvidia/Llama2-13B-SteerLM-RM`` is released as a 26 GB ``.nemo`` checkpoint,
not as a Transformers model.  NVIDIA's documented inference path is a
NeMo-Aligner PyTriton server.  This module talks to that server through
NVIDIA's lightweight Triton HTTP client while retaining the raw regression
outputs (the annotation script rounds them only when creating categorical
training labels).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


MODEL_NAME = "nvidia/Llama2-13B-SteerLM-RM"
MODEL_REVISION = "a8fd6a92a4967f934612f519a2ef078bb87ec431"
CHECKPOINT_SHA256 = "19851451def71de48d044bbebf93481308f1a778153e96e8b93f490047082b27"

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant.\n"
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
ALL_ATTRIBUTES: tuple[str, ...] = (
    "quality",
    "toxicity",
    "humor",
    "creativity",
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)
HELPSTEER_ATTRIBUTES: tuple[str, ...] = ALL_ATTRIBUTES[4:]


def format_conversation(prompt: str, answer: str) -> str:
    """Apply NeMo-Aligner's exact single-turn SteerLM templates."""
    return (
        f"System\n{SYSTEM_PROMPT}\n"
        f"User\n{prompt}\n"
        f"Assistant\n{answer}\n"
    )


def select_helpsteer_attributes(
    rewards: Sequence[float], attributes: Sequence[str]
) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.shape != (len(ALL_ATTRIBUTES),):
        raise ValueError(
            f"SteerLM must return {len(ALL_ATTRIBUTES)} attributes, got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("SteerLM returned non-finite rewards.")
    indices = []
    for attribute in attributes:
        try:
            indices.append(ALL_ATTRIBUTES.index(str(attribute)))
        except ValueError as error:
            raise KeyError(
                f"{attribute!r} is not a SteerLM attribute; known={list(ALL_ATTRIBUTES)}"
            ) from error
    return values[indices]


class SteerLMRemoteScorer:
    """Score through the official NeMo-Aligner PyTriton endpoint."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 1424,
        server_model_name: str = "reward_model",
        operator_attestation: str,
    ) -> None:
        if not str(host).strip():
            raise ValueError("A SteerLM NeMo server host is required.")
        expected = self.expected_attestation()
        if operator_attestation != expected:
            raise RuntimeError(
                "SteerLM server identity is not attestable through NVIDIA's endpoint. "
                "Set STEERLM_OPERATOR_ATTESTATION to the exact frozen model/revision/hash "
                "only after starting that checkpoint.\n"
                f"Expected: {expected}"
            )
        self.host = str(host)
        self.port = int(port)
        self.server_model_name = str(server_model_name)
        self.operator_attestation = operator_attestation
        self._health_checked = False

    @staticmethod
    def expected_attestation() -> str:
        return f"{MODEL_NAME}@{MODEL_REVISION}:sha256={CHECKPOINT_SHA256}"

    @staticmethod
    def _as_server_array(sentences: Sequence[str]) -> np.ndarray:
        # Triton's BYTES representation is an object array.  Explicitly encode
        # each value so non-ASCII prompts have one unambiguous wire format.
        values = np.empty((len(sentences), 1), dtype=object)
        values[:, 0] = [str(sentence).encode("utf-8") for sentence in sentences]
        return values

    def _raw_batch(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, len(ALL_ATTRIBUTES)), dtype=np.float64)
        try:
            import tritonclient.http as httpclient
        except ImportError as error:
            raise RuntimeError(
                "The Triton HTTP client is missing. Install the notebook's "
                "tritonclient[http] dependency before SteerLM scoring."
            ) from error

        encoded = self._as_server_array(sentences)
        rows: list[np.ndarray] = []
        client = httpclient.InferenceServerClient(
            url=f"{self.host}:{self.port}",
            verbose=False,
            concurrency=max(1, min(len(sentences), 32)),
        )
        try:
            if not client.is_server_live():
                raise RuntimeError("SteerLM Triton server is not live.")
            if not client.is_model_ready(self.server_model_name):
                raise RuntimeError(
                    f"SteerLM Triton model {self.server_model_name!r} is not ready."
                )

            # Preserve the reference client's one-request-per-text protocol.
            # Asynchronous HTTP requests still let the server form its configured
            # dynamic micro-batches without pulling PyTriton's bundled server into
            # the Colab environment.
            requests = []
            for single in np.split(encoded, encoded.shape[0]):
                input_tensor = httpclient.InferInput(
                    "sentences", list(single.shape), "BYTES"
                )
                input_tensor.set_data_from_numpy(single, binary_data=True)
                requests.append(
                    client.async_infer(
                        model_name=self.server_model_name,
                        inputs=[input_tensor],
                    )
                )

            for request in requests:
                output = request.get_result()
                raw_rewards = output.as_numpy("rewards")
                raw_exceeded = output.as_numpy("exceeded")
                if raw_rewards is None or raw_exceeded is None:
                    raise RuntimeError(
                        "SteerLM server response must contain rewards and exceeded."
                    )
                reward = np.asarray(raw_rewards, dtype=np.float64).reshape(-1)
                exceeded = np.asarray(raw_exceeded).reshape(-1)
                if exceeded.size and bool(exceeded[0]):
                    raise RuntimeError("SteerLM input exceeded its 4096-token context.")
                if reward.shape != (len(ALL_ATTRIBUTES),):
                    raise RuntimeError(
                        f"SteerLM server returned reward shape {reward.shape}, expected (9,)."
                    )
                rows.append(reward)
        finally:
            client.close()
        result = np.asarray(rows, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("SteerLM server returned non-finite values.")
        return result

    def health_check(self) -> np.ndarray:
        rewards = self._raw_batch(["hello world!"])
        if rewards.shape != (1, len(ALL_ATTRIBUTES)):
            raise RuntimeError(f"Unexpected SteerLM health-check shape: {rewards.shape}")
        self._health_checked = True
        print("[health] SteerLM NeMo server returned nine finite attributes.")
        return rewards[0]

    def score(
        self, prompt: str, answer: str, attributes: Sequence[str]
    ) -> np.ndarray:
        if not self._health_checked:
            self.health_check()
        raw = self._raw_batch([format_conversation(prompt, answer)])[0]
        return select_helpsteer_attributes(raw, attributes)

    __call__ = score

    def score_many(
        self,
        prompts: Sequence[str],
        answers: Sequence[str],
        attributes: Sequence[str],
    ) -> np.ndarray:
        """Score several pairs through one client connection.

        The server still receives one request per text, matching NVIDIA's
        reference client, but the asynchronous futures share one connection and
        let the server form its configured micro-batches.
        """
        if len(prompts) != len(answers):
            raise ValueError("prompts and answers must have the same length")
        if not self._health_checked:
            self.health_check()
        raw = self._raw_batch(
            [format_conversation(prompt, answer) for prompt, answer in zip(prompts, answers)]
        )
        return np.asarray(
            [select_helpsteer_attributes(row, attributes) for row in raw],
            dtype=np.float64,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "reward_model_name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "backend": "official NeMo-Aligner PyTriton server via Triton HTTP client",
            "client_package": "tritonclient[http]",
            "server": f"{self.host}:{self.port}/{self.server_model_name}",
            "operator_attestation": self.operator_attestation,
            "batch_size": 1,
            "attribute_order": list(ALL_ATTRIBUTES),
            "selected_attributes": list(HELPSTEER_ATTRIBUTES),
            "raw_regression_outputs_retained": True,
            "health_checked": self._health_checked,
        }
