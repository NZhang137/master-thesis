"""Lightweight HelpSteer2 proxy scoring for prototype evaluations.

These deterministic surface-level heuristics are infrastructure placeholders.
They are not reward-model scores, factual correctness measurements, or human
HelpSteer2 labels.
"""

from __future__ import annotations

import re


OBJECTIVES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

PROXY_COLUMNS = tuple(f"{objective}_proxy" for objective in OBJECTIVES)

HELPFUL_PHRASES = (
    "you can",
    "try",
    "consider",
    "recommend",
    "help",
    "practice",
    "support",
    "important",
)

EXPLANATION_PHRASES = (
    "because",
    "therefore",
    "for example",
    "this means",
    "the reason",
    "evidence",
    "depends on",
)

CALIBRATED_PHRASES = (
    "may",
    "might",
    "can",
    "often",
    "generally",
    "in some cases",
)

OVERCONFIDENT_PHRASES = (
    "always",
    "never",
    "guaranteed",
    "100 percent",
    "without exception",
)

COHERENCE_PHRASES = (
    "first",
    "second",
    "then",
    "however",
    "also",
    "finally",
    "because",
    "therefore",
)

WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
SENTENCE_END_PATTERN = re.compile(r"[.!?]+")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value to a closed interval."""
    return min(max(value, lower), upper)


def normalize_text(text: str) -> str:
    """Return lowercase text with collapsed whitespace."""
    return " ".join(text.lower().split())


def words(text: str) -> list[str]:
    """Extract simple lowercase word-like tokens."""
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def count_phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    """Count how many distinct phrases occur in normalized text."""
    normalized = normalize_text(text)
    return sum(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None
        for phrase in phrases
    )


def reasonable_length_score(length: int) -> float:
    """Return a bounded score favoring responses between 20 and 120 words."""
    if length <= 0:
        return 0.0
    if length < 20:
        return length / 20
    if length <= 120:
        return 1.0
    return max(0.0, 1.0 - (length - 120) / 180)


def helpfulness_proxy(response: str) -> float:
    """Estimate helpful surface wording, rather than actual task success."""
    response_words = words(response)
    if not response_words:
        return 0.0

    phrase_score = min(count_phrase_hits(response, HELPFUL_PHRASES) / 3, 1.0)
    score = (
        0.20
        + 0.45 * reasonable_length_score(len(response_words))
        + 0.35 * phrase_score
    )
    return round(clamp(score), 6)


def correctness_proxy(response: str) -> float:
    """Estimate explanatory and calibrated wording, not factual correctness."""
    if not response.strip():
        return 0.0

    explanation_score = min(
        count_phrase_hits(response, EXPLANATION_PHRASES) / 3,
        1.0,
    )
    calibration_score = min(
        count_phrase_hits(response, CALIBRATED_PHRASES) / 3,
        1.0,
    )
    overconfidence_score = min(
        count_phrase_hits(response, OVERCONFIDENT_PHRASES) / 2,
        1.0,
    )
    score = (
        0.35
        + 0.35 * explanation_score
        + 0.20 * calibration_score
        + 0.10 * reasonable_length_score(len(words(response)))
        - 0.30 * overconfidence_score
    )
    return round(clamp(score), 6)


def adjacent_repetition_rate(tokens: list[str]) -> float:
    """Return the fraction of adjacent token pairs that repeat a token."""
    if len(tokens) < 2:
        return 0.0
    repeated = sum(a == b for a, b in zip(tokens, tokens[1:]))
    return repeated / (len(tokens) - 1)


def coherence_proxy(response: str) -> float:
    """Estimate basic organization and continuity from textual surface cues."""
    response_words = words(response)
    if not response_words:
        return 0.0

    sentence_score = min(len(SENTENCE_END_PATTERN.findall(response)) / 3, 1.0)
    connector_score = min(
        count_phrase_hits(response, COHERENCE_PHRASES) / 3,
        1.0,
    )
    repetition_penalty = min(adjacent_repetition_rate(response_words) * 5, 1.0)
    score = (
        0.30
        + 0.30 * sentence_score
        + 0.30 * connector_score
        + 0.10 * reasonable_length_score(len(response_words))
        - 0.35 * repetition_penalty
    )
    return round(clamp(score), 6)


def complexity_proxy(response: str) -> float:
    """Estimate lexical and sentence complexity without judging its quality."""
    response_words = words(response)
    if not response_words:
        return 0.0

    average_word_length = sum(map(len, response_words)) / len(response_words)
    long_word_ratio = sum(len(word) >= 8 for word in response_words) / len(
        response_words
    )
    sentence_count = max(len(SENTENCE_END_PATTERN.findall(response)), 1)
    average_sentence_length = len(response_words) / sentence_count

    word_length_score = clamp((average_word_length - 3.5) / 3.0)
    long_word_score = clamp(long_word_ratio / 0.25)
    sentence_length_score = clamp((average_sentence_length - 6) / 24)
    score = (
        0.15
        + 0.35 * word_length_score
        + 0.30 * long_word_score
        + 0.20 * sentence_length_score
    )
    return round(clamp(score), 6)


def verbosity_proxy(response: str) -> float:
    """Use normalized word count as a simple proxy for response verbosity."""
    response_length = len(words(response))
    return round(clamp(response_length / 120), 6)


def score_response(response: str) -> dict[str, float | int | bool]:
    """Compute all response-level HelpSteer2 placeholder proxy fields."""
    response_length = len(words(response))
    return {
        "helpfulness_proxy": helpfulness_proxy(response),
        "correctness_proxy": correctness_proxy(response),
        "coherence_proxy": coherence_proxy(response),
        "complexity_proxy": complexity_proxy(response),
        "verbosity_proxy": verbosity_proxy(response),
        "response_length": response_length,
        "empty_response": response_length == 0,
    }
