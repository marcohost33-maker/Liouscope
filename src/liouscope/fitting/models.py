"""Fit hierarchy M0 ... M3b (Spec Teil 10).

* M0  : ``A * exp(-alpha t)`` -- baseline primitive
* M1  : ``A * exp(-alpha t) + C`` -- offset (often numerical artifact)
* M2  : ``A1 * exp(-beta1 t) + A2 * exp(-beta2 t)`` -- biexponential
* M3a : ``(A + B t) * exp(-alpha t)`` -- Jordan block / LEP signature
* M3b : ``A * exp(-beta t) * cos(omega t + phi)`` -- oscillatory exponential

These are **not nested** in general (anchor F). Use AICc (with N_eff
correction) to compare them, never likelihood-ratio tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ModelFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


def M0(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A * exp(-alpha t)``; params = (A, alpha)."""
    A, alpha = params
    return A * np.exp(-alpha * t)


def M1(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A * exp(-alpha t) + C``; params = (A, alpha, C)."""
    A, alpha, C = params
    return A * np.exp(-alpha * t) + C


def M2(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A1 exp(-beta1 t) + A2 exp(-beta2 t)``; params = (A1, beta1, A2, beta2)."""
    A1, beta1, A2, beta2 = params
    return A1 * np.exp(-beta1 * t) + A2 * np.exp(-beta2 * t)


def M3a(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``(A + B t) exp(-alpha t)``; params = (A, B, alpha)."""
    A, B, alpha = params
    return (A + B * t) * np.exp(-alpha * t)


def M3b(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A exp(-beta t) cos(omega t + phi)``; params = (A, beta, omega, phi)."""
    A, beta, omega, phi = params
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)


def initial_guess_m0(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-linear regression on positive samples (FIX-5)."""
    y = np.asarray(y, dtype=float)
    mask = y > 0
    if mask.sum() < 2:
        return np.array([y[0] if y.size else 1.0, 1.0])
    log_y = np.log(y[mask])
    A_design = np.vstack([np.ones_like(log_y), t[mask]]).T
    coefs, *_ = np.linalg.lstsq(A_design, log_y, rcond=None)
    A_est = float(np.exp(coefs[0]))
    alpha_est = float(-coefs[1])
    return np.array([A_est, max(alpha_est, 1.0e-3)])


def initial_guess_m1(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Seed M1 from M0 plus residual offset."""
    A, alpha = initial_guess_m0(t, y)
    return np.array([A, alpha, 0.0])


def initial_guess_m2(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Seed M2 with two well-separated rates."""
    A, alpha = initial_guess_m0(t, y)
    return np.array([0.5 * A, alpha, 0.5 * A, alpha * 3.0])


def initial_guess_m3a(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Seed (A + B t) exp(-alpha t) from M0 plus small slope."""
    A, alpha = initial_guess_m0(t, y)
    return np.array([A, 0.01 * A, alpha])


def initial_guess_m3b(
    t: np.ndarray,
    y: np.ndarray,
    *,
    omega_guess: float | None = None,
) -> np.ndarray:
    """Seed M3b. Falls back to a damped-oscillator guess if no Prony seed
    is supplied (use :func:`liouscope.fitting.prony.prony_seed` for higher
    quality)."""
    A, alpha = initial_guess_m0(t, np.abs(y) + 1.0e-12)
    if omega_guess is None and t.size > 4:
        # Quick FFT pick of dominant non-DC frequency.
        dt = float(t[1] - t[0]) if t.size > 1 else 1.0
        spectrum = np.abs(np.fft.rfft(y - y.mean()))
        freqs = np.fft.rfftfreq(y.size, d=dt)
        if spectrum.size > 2:
            idx = 1 + int(np.argmax(spectrum[1:]))
            omega_guess = 2.0 * np.pi * float(freqs[idx])
        else:
            omega_guess = 1.0
    elif omega_guess is None:
        omega_guess = 1.0
    return np.array([A, alpha, omega_guess, 0.0])
