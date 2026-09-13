"""Fail-closed policy for omitting a collapsed Cert candidate from evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def should_exclude_cert_from_phase_b(
    audit_rows: Iterable[Mapping[str, Any]],
    phase_b_names: Sequence[str] | set[str],
    *,
    tolerance: float,
) -> bool:
    """Return true only when every requested Phase-B Cert point equals baseline.

    A row qualifies only if the independent floor LP reports collapse and the
    measured L2 distance between the Cert lambda and ``p`` is at most
    ``tolerance``. Missing or duplicate Phase-B rows are errors, not evidence of
    collapse.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    expected = {str(name) for name in phase_b_names}
    selected = [row for row in audit_rows if str(row.get("p_name")) in expected]
    observed_names = [str(row.get("p_name")) for row in selected]
    if len(selected) != len(expected) or set(observed_names) != expected:
        raise ValueError(
            "Cert audit does not contain exactly one row per Phase-B preference."
        )
    if len(observed_names) != len(set(observed_names)):
        raise ValueError("Cert audit contains duplicate Phase-B preferences.")
    return all(
        bool(row.get("floor_lp_collapsed"))
        and float(row.get("lambda_minus_p_l2", float("inf"))) <= tolerance
        for row in selected
    )

