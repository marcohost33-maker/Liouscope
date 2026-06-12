"""Tests for the M0..M3b fit-model functions and their initial-guess seeds.

These pin the closed-form model evaluations (Spec Teil 10) and the heuristic
parameter seeds, including the log-linear M0 regression and the FFT-based M3b
frequency pick. The functions are pure, so the tests are exact / deterministic.
"""

from __future__ import annotations

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
    "y, expected_A",
    [
        (np.array([0.0, -1.0, -2.0]), 0.0),  # fewer than 2 positive samples
        (np.array([]), 1.0),                  # empty -> default amplitude
    ],
)
def test_initial_guess_m0_fallback_when_too_few_positive(y, expected_A):
    tt = np.arange(y.size, dtype=float)
    A, alpha = initial_guess_m0(tt, y)
    assert pytest.approx(expected_A) == A
    assert alpha == pytest.approx(1.0)


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
