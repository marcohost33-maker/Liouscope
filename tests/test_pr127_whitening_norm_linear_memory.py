"""PR #127 review thread (relaxation.py:1042): the whitening-norm bound of the
AICc tie band was formed from a dense ``n x n`` matrix.

``_rank_candidates`` needs ``||W||_2`` of the residual whitening only through
the upper bound ``sqrt(||W||_1 ||W||_inf)``. It used to build
``W = [whiten(e_0), ..., whiten(e_{n-1})]`` from ``np.eye(n)``, plus the stacked
matrix and its absolute values -- quadratic memory for one scalar, per
candidate and per fitted curve. The bound is now formed from two probes of the
(lower-bidiagonal) whitening in O(n) storage.

Oracle: the historical dense expression, kept verbatim below. The new bound
must reproduce it BIT FOR BIT (each sum has at most two non-zero terms, and
every entry is the same floating-point expression), must bound the spectral
norm from above, and must refuse (NaN) a whitening that is not bidiagonal.
"""

from __future__ import annotations

import math
import tracemalloc
import warnings

import numpy as np
import pytest

from liouscope.diagnostics import relaxation as rx
from liouscope.fitting.car1 import whiten_car1
from liouscope.fitting.gls import _whiten as whiten_ar1


def _dense_bound(whiten, n: int) -> float:
    """The pre-fix expression, verbatim (relaxation.py before this change)."""
    w_matrix = np.column_stack([whiten(col) for col in np.eye(n)])
    w_abs = np.abs(w_matrix)
    return float(np.sqrt(np.max(np.sum(w_abs, axis=0)) * np.max(np.sum(w_abs, axis=1))))


def _two_scale(n: int) -> np.ndarray:
    half = n // 2
    fast = np.linspace(0.0, 1.0, half)
    slow = np.linspace(1.0, 1.0e4, n - half + 1)[1:]
    return np.concatenate([fast, slow])


def _car1_factory(theta: float):
    def make(n: int):
        t = _two_scale(n)
        return lambda v: whiten_car1(v, t, theta)

    return make


def _ar1_factory(rho: float):
    def make(n: int):
        return lambda v: whiten_ar1(v, rho)

    return make


_CASES = [
    # (label, factory(n) -> whiten). theta 1e-15 puts 1 - a^2 below the 1e-12
    # variance floor of whiten_car1 on every step for n >= 80 (measured: 79/79,
    # 160/160); rho = 1 - 1e-14 does the same for the AR(1) observation-0 floor.
    *[(f"car1-theta={th:g}", _car1_factory(th)) for th in (1.0e-15, 1.0e-3, 0.7, 50.0, 1.0e6)],
    *[(f"ar1-rho={r:.15g}", _ar1_factory(r)) for r in (0.0, 0.5, 0.99, -0.7, 1.0 - 1.0e-14)],
]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 80, 161])
@pytest.mark.parametrize("label,factory", _CASES, ids=[c[0] for c in _CASES])
def test_bound_is_bit_identical_to_the_dense_form(label, factory, n) -> None:
    whiten = factory(n)
    new = rx._bidiagonal_whitening_norm_bound(whiten, n)
    old = _dense_bound(whiten, n)
    assert new == old, (label, n, new, old)


@pytest.mark.parametrize("label,factory", _CASES, ids=[c[0] for c in _CASES])
def test_bound_dominates_the_spectral_norm(label, factory) -> None:
    """``sqrt(||W||_1 ||W||_inf) >= ||W||_2`` (Hoelder). The 2-norm comes from a
    BLAS SVD, so the comparison allows its backward error, ``n * eps``
    relative -- the bound is never expected to be tight."""
    n = 60
    whiten = factory(n)
    w = np.column_stack([whiten(col) for col in np.eye(n)])
    spectral = float(np.linalg.norm(w, 2))
    bound = rx._bidiagonal_whitening_norm_bound(whiten, n)
    assert math.isfinite(bound)
    assert spectral <= bound * (1.0 + n * float(np.finfo(float).eps)), (label, spectral, bound)


def _second_lag(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=float).copy()
    out[2:] = out[2:] - 0.3 * v[:-2]
    return out


def _upper(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=float).copy()
    out[:-1] = out[:-1] - 0.3 * v[1:]
    return out


@pytest.mark.parametrize("whiten", [_second_lag, _upper], ids=["lag-2", "upper"])
def test_a_whitening_outside_the_two_diagonals_is_refused(whiten) -> None:
    """NEGATIVE CONTROL: the probes assume a lower-bidiagonal ``W``. A whitening
    with an entry two below or one above the diagonal must not be summarised
    from two probes; the bound is NaN (-> the candidate stays tied)."""
    assert math.isnan(rx._bidiagonal_whitening_norm_bound(whiten, 30))
    # Positive control on the SAME size: a bidiagonal whitening is accepted,
    # so the refusal above is the structure check, not the size.
    assert math.isfinite(rx._bidiagonal_whitening_norm_bound(lambda v: whiten_ar1(v, 0.3), 30))


def test_non_finite_whitening_fails_closed() -> None:
    assert math.isnan(rx._bidiagonal_whitening_norm_bound(lambda v: v * np.nan, 5))
    assert math.isnan(rx._bidiagonal_whitening_norm_bound(lambda v: v, 0))


def test_bound_storage_is_linear_in_n() -> None:
    """n = 20000: the dense form needed three 3.2 GB arrays."""
    n = 20000
    t = _two_scale(n)
    tracemalloc.start()
    try:
        bound = rx._bidiagonal_whitening_norm_bound(lambda v: whiten_car1(v, t, 0.5), n)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert math.isfinite(bound)
    # About fifteen float64 vectors of length n (160 kB each); n^2 would be 3.2 GB.
    assert peak < 4.0e6, f"peak {peak / 1e6:.1f} MB"


def test_model_selection_no_longer_allocates_quadratically() -> None:
    """The reviewer's reproduction, at the ``_select_model`` level: 2000
    observations peaked at 64.6 MB before (n^2 float64 alone is 32 MB)."""
    n = 2000
    t = _two_scale(n)
    rng = np.random.default_rng(127)
    y = 0.6 * np.exp(-0.01 * t) + 0.4 * np.exp(-3.0 * t) + 1.0e-3 * rng.standard_normal(n)
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("M0", "M1"):
            fits[name], _ = rx._fit_with_model(name, t, y)
    assert all(f.success and math.isfinite(f.aicc) for f in fits.values())
    tracemalloc.start()
    try:
        winner, _tied = rx._select_model(fits, t, y, 1.0e-12)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert winner in fits
    assert peak < 8.0e6, f"peak {peak / 1e6:.1f} MB scales with n^2"
