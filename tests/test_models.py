"""Tests for the M0..M3b fit-model functions and their initial-guess seeds.

These pin the closed-form model evaluations (Spec Teil 10) and the heuristic
parameter seeds, including the log-linear M0 regression and the FFT-based M3b
frequency pick. The functions are pure, so the tests are exact / deterministic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope.fitting.models import (
    M0,
    M1,
    M2,
    M3a,
    M3b,
    initial_guess_m0,
    initial_guess_m1,
    initial_guess_m2,
    initial_guess_m3a,
    initial_guess_m3b,
)


@pytest.fixture
def t():
    return np.linspace(0.0, 5.0, 64)


def test_m0_is_pure_exponential(t):
    np.testing.assert_allclose(M0(t, np.array([2.0, 0.7])), 2.0 * np.exp(-0.7 * t))


def test_m1_adds_offset(t):
    np.testing.assert_allclose(
        M1(t, np.array([2.0, 0.7, 0.3])), 2.0 * np.exp(-0.7 * t) + 0.3
    )


def test_m2_is_sum_of_two_exponentials(t):
    expected = 1.0 * np.exp(-0.5 * t) + 0.4 * np.exp(-2.0 * t)
    np.testing.assert_allclose(M2(t, np.array([1.0, 0.5, 0.4, 2.0])), expected)


def test_m3a_has_linear_prefactor(t):
    expected = (1.5 + 0.2 * t) * np.exp(-0.6 * t)
    np.testing.assert_allclose(M3a(t, np.array([1.5, 0.2, 0.6])), expected)


def test_m3b_is_damped_cosine(t):
    expected = 2.0 * np.exp(-0.3 * t) * np.cos(1.1 * t + 0.4)
    np.testing.assert_allclose(M3b(t, np.array([2.0, 0.3, 1.1, 0.4])), expected)


def test_initial_guess_m0_recovers_clean_exponential(t):
    y = 3.0 * np.exp(-0.8 * t)
    A, alpha = initial_guess_m0(t, y)
    assert pytest.approx(3.0, rel=1e-6) == A
    assert alpha == pytest.approx(0.8, rel=1e-6)


def test_initial_guess_m0_floors_alpha_positive(t):
    # A growing / flat signal would give a non-positive alpha; it must be floored.
    y = np.full_like(t, 2.0)
    _, alpha = initial_guess_m0(t, y)
    assert alpha >= 1.0e-3


@pytest.mark.parametrize(
    "y, expected_A, expected_alpha",
    [
        # Fewer than 2 positive samples. The fallback rate is now GRID-RELATIVE
        # (~one e-folding across the window) rather than an absolute 1.0: a rate
        # is 1/time, so a fixed fallback silently means "decays instantly" on a
        # long grid and "never decays" on a short one. Here t spans [0, 2].
        (np.array([0.0, -1.0, -2.0]), 0.0, 0.5),
        # Empty input has no window to scale by, so the unit fallback stands.
        (np.array([]), 1.0, 1.0),
    ],
)
def test_initial_guess_m0_fallback_when_too_few_positive(y, expected_A, expected_alpha):
    tt = np.arange(y.size, dtype=float)
    A, alpha = initial_guess_m0(tt, y)
    assert pytest.approx(expected_A) == A
    assert alpha == pytest.approx(expected_alpha)


def test_initial_guess_m0_seed_rate_scales_with_the_time_grid():
    """A pure change of time units must move the seed with it (issue #108 class).

    The absolute `1e-3` floor put the seed 1e7 above the true rate on a grid
    spanning 5e10 — a region where `exp(-alpha t)` has underflowed flat and the
    optimiser has no gradient, so the fit returned the floor itself as the
    "measured" rate and silently corrupted D5/D17.
    """
    for c in (1.0, 1.0e-3, 1.0e-10):
        t = np.linspace(0.0, 5.0 / c, 64)
        y = np.exp(-0.5 * c * t)
        _, alpha = initial_guess_m0(t, y)
        assert alpha == pytest.approx(0.5 * c, rel=1e-6)


def test_initial_guess_m0_floor_is_unchanged_on_the_canonical_grid():
    """The grid-relative floor reproduces the historical 1e-3 at t_span = 5.

    Keeping that identity is what leaves every seed in the existing suite — and
    therefore the anchors — untouched by this change.
    """
    t = np.linspace(0.0, 5.0, 64)
    flat = np.ones_like(t)          # zero slope -> the floor decides
    _, alpha = initial_guess_m0(t, flat)
    assert alpha == pytest.approx(1.0e-3, rel=1e-12)


def test_initial_guess_m1_extends_m0_with_zero_offset(t):
    y = 3.0 * np.exp(-0.8 * t)
    guess = initial_guess_m1(t, y)
    np.testing.assert_allclose(guess[:2], initial_guess_m0(t, y))
    assert guess[2] == 0.0


def test_initial_guess_m2_splits_amplitude_and_separates_rates(t):
    y = 3.0 * np.exp(-0.8 * t)
    A1, beta1, A2, beta2 = initial_guess_m2(t, y)
    assert pytest.approx(A2) == A1
    assert beta2 == pytest.approx(3.0 * beta1)


def test_initial_guess_m3a_adds_small_slope(t):
    y = 3.0 * np.exp(-0.8 * t)
    A, B, alpha = initial_guess_m3a(t, y)
    assert pytest.approx(0.01 * A) == B


def test_initial_guess_m3b_fft_picks_dominant_frequency():
    # Long record -> fine FFT frequency resolution so the picked angular
    # frequency lands close to the true value.
    t = np.linspace(0.0, 80.0, 1024)
    omega_true = 2.0
    y = np.exp(-0.05 * t) * np.cos(omega_true * t)
    A, alpha, omega, phi = initial_guess_m3b(t, y)
    assert omega == pytest.approx(omega_true, rel=0.05)
    assert phi == 0.0
    assert alpha > 0


def test_initial_guess_m3b_uses_supplied_omega():
    t = np.linspace(0.0, 20.0, 256)
    y = np.exp(-0.1 * t) * np.cos(2.0 * t)
    *_, omega, _ = initial_guess_m3b(t, y, omega_guess=5.0)
    assert omega == 5.0


def test_initial_guess_m3b_short_signal_falls_back_to_unit_omega():
    t = np.array([0.0, 1.0, 2.0])  # size <= 4 -> no FFT branch
    y = np.array([1.0, 0.5, 0.25])
    *_, omega, _ = initial_guess_m3b(t, y)
    assert omega == 1.0


# ---------------------------------------------------------------------------
# Overflow robustness on long time grids (issue #108 follow-up)
# ---------------------------------------------------------------------------


def test_models_stay_finite_for_optimiser_probes_on_long_grids():
    """The optimiser probes negative rates; a long grid must not overflow.

    A slow generator expressed in small rate units yields a legitimately long
    time grid (t up to 5e10 in the #108 rescaling tests). A probe with a
    NEGATIVE decay rate then drives the exponent past the float64 overflow
    threshold. `inf`/`nan` residuals give least-squares no gradient to step
    back from, so the failure mode is silent non-convergence on valid physical
    input -- not merely a log line.
    """
    t = np.linspace(0.0, 5.0e10, 64)
    probes = {
        M0: np.array([1.5, -0.043]),
        M1: np.array([1.5, -0.043, 4.1]),
        M2: np.array([0.5, -0.043, 0.5, -0.1]),
        M3a: np.array([1.5, 0.01, -0.043]),
        M3b: np.array([1.5, -0.043, 0.3, 0.0]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow/underflow warning fails
        for model, params in probes.items():
            out = model(t, params)
            assert np.all(np.isfinite(out)), f"{model.__name__} produced non-finite output"


@pytest.mark.parametrize(
    "model,params",
    [
        (M0, np.array([1.5, 0.4])),
        (M1, np.array([1.5, 0.4, 0.2])),
        (M2, np.array([0.5, 0.4, 0.5, 1.2])),
        (M3a, np.array([1.5, 0.01, 0.4])),
        (M3b, np.array([1.5, 0.4, 0.3, 0.0])),
    ],
)
def test_exponent_clip_is_inert_in_the_well_conditioned_regime(model, params):
    """The clip must change nothing where the models are actually fitted.

    Bit-identical, not merely close: a clip that perturbed ordinary fits would
    silently move every published rate estimate.
    """
    t = np.linspace(0.0, 20.0, 128)
    expected = _unclipped_reference(model, t, params)
    assert np.array_equal(model(t, params), expected)


def _unclipped_reference(model, t, params):
    """Recompute the model with the plain, unclipped ``np.exp``."""
    if model is M0:
        A, alpha = params
        return A * np.exp(-alpha * t)
    if model is M1:
        A, alpha, C = params
        return A * np.exp(-alpha * t) + C
    if model is M2:
        A1, b1, A2, b2 = params
        return A1 * np.exp(-b1 * t) + A2 * np.exp(-b2 * t)
    if model is M3a:
        A, B, alpha = params
        return (A + B * t) * np.exp(-alpha * t)
    A, beta, omega, phi = params
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


def test_clip_does_not_truncate_representable_decay():
    """The overflow cap is one-sided; clipping decay would bias every fit.

    `exp(-400) = 1.9e-174` is perfectly representable. A symmetric clip at 345
    reported `1.5e-150` instead -- a factor of 1e24 -- giving every model an
    artificial constant tail that distorts residuals, fitted offsets and AICc
    on high-dynamic-range trajectories.
    """
    for exponent in (300.0, 400.0, 700.0, 745.0):
        got = M0(np.array([exponent]), np.array([1.0, 1.0]))[0]
        assert got == np.exp(-exponent), f"decay truncated at exponent -{exponent}"


def test_extreme_decay_underflows_to_zero_without_warning():
    """Underflow is exact in the limit and needs no guard."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert M0(np.array([1.0e4]), np.array([1.0, 1.0]))[0] == 0.0


def test_extreme_amplitude_probes_saturate_instead_of_overflowing():
    """The factor MULTIPLICATION can overflow before any clamp sees the product.

    An unconstrained fit can probe an amplitude like 1e200; `1e200 * exp(345)`
    is already inf on arrival, and `0 * inf` yields NaN. Both encode "this
    probe is absurdly far from the data", so the models saturate at the finite
    bound instead of raising the (error-promoted) overflow warning.
    """
    t = np.array([0.0, 1000.0])
    probes = {
        M0: np.array([1.0e200, -1.0]),
        M1: np.array([1.0e200, -1.0, 1.0e200]),
        M2: np.array([1.0e200, -1.0, 1.0e200, -1.0]),
        M3a: np.array([1.0e200, 1.0e200, -1.0]),
        M3b: np.array([1.0e200, -1.0, 0.3, 0.0]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for model, params in probes.items():
            out = model(t, params)
            assert np.all(np.isfinite(out)), f"{model.__name__} not finite"
            assert np.all(np.abs(out) <= 1.0e100), f"{model.__name__} unbounded"
