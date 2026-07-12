"""Frozen preference vectors for the RS-PPO/ArmoRM upper-bound notebooks."""

from __future__ import annotations

from collections import OrderedDict


PREFERENCES = OrderedDict({
    # Quality-heavy preferences are the pre-registered negative-control family.
    "dominant_helpfulness": [0.5, 0.125, 0.125, 0.125, 0.125],
    "dominant_correctness": [0.125, 0.5, 0.125, 0.125, 0.125],
    "dominant_coherence": [0.125, 0.125, 0.5, 0.125, 0.125],
    "only_helpfulness": [1.0, 0.0, 0.0, 0.0, 0.0],
    "only_correctness": [0.0, 1.0, 0.0, 0.0, 0.0],
    "only_coherence": [0.0, 0.0, 1.0, 0.0, 0.0],
    # Complexity/verbosity is the regime where the earlier geometry worked best.
    "dominant_complexity": [0.125, 0.125, 0.125, 0.5, 0.125],
    "dominant_verbosity": [0.125, 0.125, 0.125, 0.125, 0.5],
    "only_complexity": [0.0, 0.0, 0.0, 1.0, 0.0],
    "only_verbosity": [0.0, 0.0, 0.0, 0.0, 1.0],
    "uniform": [0.2, 0.2, 0.2, 0.2, 0.2],
})

QUALITY_PREFS = [
    "dominant_helpfulness",
    "dominant_correctness",
    "dominant_coherence",
    "only_helpfulness",
    "only_correctness",
    "only_coherence",
]

CV_PREFS = [
    "dominant_complexity",
    "dominant_verbosity",
    "only_complexity",
    "only_verbosity",
]


def preference_regime(name: str) -> str:
    """Return the pre-registered regime label for one preference name."""
    if name in QUALITY_PREFS:
        return "quality"
    if name in CV_PREFS:
        return "complexity/verbosity"
    return "uniform"
