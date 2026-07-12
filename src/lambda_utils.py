"""Small shared utilities for lambda-keying and multiplicity correction."""

from __future__ import annotations

import numpy as np


def lambda_key(lmbda: np.ndarray | list[float] | tuple[float, ...], decimals: int = 8) -> str:
    """Stable hex key for a simplex vector rounded to fixed precision."""
    arr = np.round(np.asarray(lmbda, dtype=np.float64), decimals)
    return arr.tobytes().hex()


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    values = np.asarray(p_values, dtype=np.float64)
    m = len(values)
    order = np.argsort(values)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted
