"""Numerical check of the floor-collapse sign test (thesis appendix, catalogue P46/P47).

Reproduces the numbers of Remark `rem:open-frequency` and cross-checks the exact sign test
K = {0}  <=>  R^{-1} 1 > 0 (R positive definite) against the linear program of
Proposition `prop:class-collapse`.

    python scripts/floor_sign_test_check.py            # about 10 s on CPU
"""
from __future__ import annotations

import argparse

import numpy as np


def sign_test_open(R: np.ndarray) -> bool:
    """Floor open (K != {0}) for positive definite R: some entry of R^{-1} 1 is <= 0."""
    return bool((np.linalg.solve(R, np.ones(len(R))) <= 0).any())


def lp_open(R: np.ndarray, tol: float = 1e-9) -> bool | None:
    """max min_i (Rv)_i over 1^T v = 0, |v|_inf <= 1. True if the value is > 0, None if it is 0."""
    from scipy.optimize import linprog

    m = len(R)
    c = np.zeros(m + 1)
    c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([-R, np.ones((m, 1))]), b_ub=np.zeros(m),
                  A_eq=np.hstack([np.ones((1, m)), np.zeros((1, 1))]), b_eq=[0.0],
                  bounds=[(-1, 1)] * m + [(None, None)], method="highs")
    return True if -res.fun > tol else None


def open_frequency_m3(n_draws: int, seed: int) -> None:
    """Uniform 3x3 correlation matrices (LKJ, eta = 1): share with an open floor by number of negative pairs."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(-1, 1, (3, n_draws))
    pd = 1 + 2 * a * b * c - a * a - b * b - c * c > 0
    a, b, c = a[pd], b[pd], c[pd]
    k = (a < 0).astype(int) + (b < 0).astype(int) + (c < 0).astype(int)
    opened = (a + b >= 1 + c) | (a + c >= 1 + b) | (b + c >= 1 + a)   # Prop. P47(d)
    print(f"positive definite draws: {len(a):,}")
    for j in range(4):
        s = k == j
        print(f"  {j} negative pairs: {s.mean():6.1%} of draws, floor open in {opened[s].mean():6.1%}")
    print(f"  overall: floor open in {opened.mean():.1%}")


def cross_checks(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # (d) redundancy criterion and adjugate identity against the sign test
    n = mism = 0
    while n < 20000:
        a, b, c = rng.uniform(-1, 1, 3)
        R = np.array([[1, a, b], [a, 1, c], [b, c, 1.0]])
        if np.linalg.eigvalsh(R).min() <= 1e-9:
            continue
        n += 1
        red = (a + b >= 1 + c) or (a + c >= 1 + b) or (b + c >= 1 + a)
        mism += red != sign_test_open(R)
        assert abs(np.linalg.solve(R, np.ones(3))[0] * np.linalg.det(R) - (1 - c) * (1 + c - a - b)) < 1e-8
    print(f"m=3 redundancy criterion vs sign test: {mism} mismatches in {n} matrices")
    # (a) sign test against the LP on random Gram matrices, m = 2..6
    try:
        disagree = 0
        for m in range(2, 7):
            for _ in range(500):
                D = rng.normal(size=(20, m)) + rng.normal() * rng.normal(size=(20, 1))
                R = D.T @ D
                lp = lp_open(R)
                if lp is True and not sign_test_open(R):
                    disagree += 1
        print(f"LP value > 0 but sign test says collapsed: {disagree} cases in 2500 matrices")
    except ImportError:
        print("scipy not installed: LP cross-check skipped")
    # (b) equicorrelated matrices always collapse
    for m in (2, 3, 5, 10):
        for cc in (-0.99 / (m - 1), 0.0, 0.5, 0.999):
            assert not sign_test_open((1 - cc) * np.eye(m) + cc * np.ones((m, m)))
    print("equicorrelated matrices: all collapse")
    # example of (d): no negative entry, floor open
    R = np.array([[1, 0.61, 0.61], [0.61, 1, 0], [0.61, 0, 1.0]])
    assert np.linalg.eigvalsh(R).min() > 0 and sign_test_open(R)
    print("example a=b=0.61, c=0: R^-1 1 =", np.round(np.linalg.solve(R, np.ones(3)), 3))
    # example of (b): near-parallel but not equicorrelated, floor open
    R = np.array([[1, 0.99, 0.99], [0.99, 1, 0.97], [0.99, 0.97, 1.0]])
    assert np.linalg.eigvalsh(R).min() > 0 and sign_test_open(R)
    print(f"example 0.99/0.99/0.97: det = {np.linalg.det(R):.2e}, R^-1 1 =", np.round(np.linalg.solve(R, np.ones(3)), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_draws", type=int, default=6_000_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    open_frequency_m3(args.n_draws, args.seed)
    cross_checks(args.seed)
