"""Shared scalar metrics for RS-PPO/ArmoRM analysis notebooks."""

from __future__ import annotations

import numpy as np


def delta_m_percent(
    heads_mean: np.ndarray,
    stl: np.ndarray,
    p: np.ndarray | None = None,
    denom_min: float = 1e-3,
) -> float:
    """Return Delta m% relative to specialist diagonal rewards.

    When `p` is None this is the unweighted MTL-standard mean over heads:
    mean_k (r_k - stl_k) / stl_k * 100.

    When `p` is provided it is the preference-weighted variant:
    sum_k p_k (r_k - stl_k) / stl_k * 100.
    """
    values = np.asarray(heads_mean, dtype=np.float64)
    reference = np.asarray(stl, dtype=np.float64)
    if values.shape != reference.shape:
        raise ValueError(f"heads_mean and stl must have the same shape, got {values.shape} and {reference.shape}.")
    if np.any(np.abs(reference) <= denom_min):
        raise ValueError(f"Delta m% denominator too small; min abs stl={np.abs(reference).min():.4g}.")
    relative = (values - reference) / reference
    if p is None:
        return float(np.mean(relative) * 100.0)
    pref = np.asarray(p, dtype=np.float64)
    if pref.shape != values.shape:
        raise ValueError(f"p must have shape {values.shape}, got {pref.shape}.")
    return float(np.sum(pref * relative) * 100.0)
