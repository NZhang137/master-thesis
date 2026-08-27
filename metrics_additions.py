# --- append to src/metrics.py ------------------------------------------------
# The pre-registration in NB10 cell 23 promises three reporting quantities that
# had no implementation anywhere in `src/`: Mean Rank, Selection Regret, and the
# paired bootstrap CI over prompts. They are defined here so the notebook keeps
# importing metrics rather than defining any of them inline (gate G6).
#
# The primary utility itself stays where it is: `preference_utility` in
# `src.proxy_validation`, identity normalization, raw reward scale.

from typing import Mapping, Sequence


def mean_rank(utilities_by_method: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Return each method's mean rank across preferences (1 = best).

    ``utilities_by_method`` maps a method label to its U_p values, one per
    preference, all in the SAME preference order. Ties share the average rank,
    so a method that returns p on every preference - the expected outcome under
    floor collapse - is not penalised relative to its equals.
    """
    from scipy.stats import rankdata

    names = list(utilities_by_method)
    if not names:
        raise ValueError("utilities_by_method is empty.")
    matrix = np.asarray([np.asarray(utilities_by_method[n], dtype=np.float64) for n in names])
    if matrix.ndim != 2:
        raise ValueError("Every method needs the same number of preference entries.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("utilities contain non-finite values.")
    # Rank within each preference column, descending (higher utility = rank 1).
    ranks = np.column_stack([
        rankdata(-matrix[:, j], method="average") for j in range(matrix.shape[1])
    ])
    return {name: float(ranks[i].mean()) for i, name in enumerate(names)}


def selection_regret(
    utility_of_method: float,
    utilities_of_search_set: Sequence[float],
) -> float:
    """Return U_p(lambda_best) - U_p(lambda_method) over the evaluated set.

    Regret is non-negative by construction when the method's own point is part
    of the search set. It answers the question the raw utility does not: how
    much is lost by following the mapping instead of the best coefficient the
    experiment actually evaluated. Note the reference is the best point IN THIS
    FINITE SET, not a global optimum - report it as such.
    """
    values = np.asarray(utilities_of_search_set, dtype=np.float64)
    if values.size == 0:
        raise ValueError("utilities_of_search_set is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("utilities_of_search_set contains non-finite values.")
    return float(values.max() - float(utility_of_method))


def paired_bootstrap_ci(
    rewards_method: np.ndarray,
    rewards_baseline: np.ndarray,
    p: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 137,
) -> dict[str, float | bool | int]:
    """Paired bootstrap CI over prompts for Delta U_p = U_p(method) - U_p(p).

    Both arguments have shape ``(num_prompts, num_objectives)`` and must be the
    per-prompt scores for the SAME prompts in the SAME order - that pairing is
    what removes prompt difficulty from the comparison and is the reason Phase B
    has to keep the unaveraged scores.

    The percentile interval is descriptive. NB10 applies its confirmatory rule
    afterwards to the returned p-values using Holm-Bonferroni across the full
    family of method/preference comparisons.
    """
    left = np.asarray(rewards_method, dtype=np.float64)
    right = np.asarray(rewards_baseline, dtype=np.float64)
    pref = np.asarray(p, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}.")
    if left.ndim != 2:
        raise ValueError("Per-prompt rewards must be two-dimensional.")
    if pref.shape != (left.shape[1],):
        raise ValueError(f"p must have shape {(left.shape[1],)}, got {pref.shape}.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Per-prompt rewards contain non-finite values.")
    if n_boot < 1000:
        raise ValueError("Use at least 1000 bootstrap draws.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    per_prompt_delta = (left - right) @ pref
    num_prompts = per_prompt_delta.size
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, num_prompts, size=(n_boot, num_prompts))
    boot_means = per_prompt_delta[draws].mean(axis=1)

    low = float(np.quantile(boot_means, alpha / 2.0))
    high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    # Two-sided bootstrap p-value: the share of draws on the far side of zero,
    # doubled. Floored at 1/n_boot, because zero draws beyond zero means the
    # p-value is below the resolution of the bootstrap, not that it is zero.
    share = float(np.mean(boot_means <= 0.0) if per_prompt_delta.mean() > 0
                  else np.mean(boot_means >= 0.0))
    p_value = min(1.0, max(2.0 * share, 1.0 / n_boot))
    return {
        "delta_u_p": float(per_prompt_delta.mean()),
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "improves": bool(low > 0.0),
        "n_prompts": int(num_prompts),
        "n_boot": int(n_boot),
    }
