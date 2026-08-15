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

# ``np.exp`` overflows for float64 arguments above ~709.78. During least-squares
# optimisation the solver freely probes parameter values that are physically
# meaningless (a NEGATIVE decay rate, say), and on a long time grid -- which is
# exactly what a slow generator in small rate units produces, e.g. t up to 5e10
# for the issue-#108 rescaling tests -- that probe overflows.
#
# Overflow is not a harmless log line here: the project runs pytest with
# ``filterwarnings = ["error"]``, and more importantly ``inf``/``nan`` residuals
# give the optimiser no gradient information to step back from, so the failure
# mode is silent non-convergence on legitimate physical input rather than a
# clean rejection of the bad probe.
#
# Clipping the EXPONENT keeps every model finite and monotone in the parameters
# over the whole real line while changing nothing in the well-conditioned
# regime, where the exponent of a decaying model is <= 0 and this bound is
# unreachable: the value is bit-identical to the unclipped one.
#
# The bound is 345, not the ~709 where ``np.exp`` itself overflows, because
# saturating exp is not sufficient on its own: M3a multiplies it by the
# polynomial prefactor ``(A + B t)``, so a clip at 709 turns an exp overflow
# into a *multiply* overflow one line later. ``exp(345) ~ 4.6e149`` leaves ~158
# decades of headroom for the prefactor, which covers any t and B a fit can
# plausibly see (``B t ~ 5e8`` on the longest grid used here).
_EXP_CLIP: float = 345.0


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """``exp(x)`` with the exponent clipped to a finite, non-overflowing range.

    Bit-identical to ``np.exp`` for ``|x| <= 700``; saturating instead of
    overflowing to ``inf`` (or underflowing to a warning) outside it.
    """
    result: np.ndarray = np.exp(np.clip(x, -_EXP_CLIP, _EXP_CLIP))
    return result


def M0(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A * exp(-alpha t)``; params = (A, alpha)."""
    A, alpha = params
    result: np.ndarray = A * _safe_exp(-alpha * t)
    return result


def M1(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A * exp(-alpha t) + C``; params = (A, alpha, C)."""
    A, alpha, C = params
    result: np.ndarray = A * _safe_exp(-alpha * t) + C
    return result


def M2(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A1 exp(-beta1 t) + A2 exp(-beta2 t)``; params = (A1, beta1, A2, beta2)."""
    A1, beta1, A2, beta2 = params
    result: np.ndarray = A1 * _safe_exp(-beta1 * t) + A2 * _safe_exp(-beta2 * t)
    return result


def M3a(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``(A + B t) exp(-alpha t)``; params = (A, B, alpha)."""
    A, B, alpha = params
    result: np.ndarray = (A + B * t) * _safe_exp(-alpha * t)
    return result


def M3b(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A exp(-beta t) cos(omega t + phi)``; params = (A, beta, omega, phi)."""
    A, beta, omega, phi = params
    result: np.ndarray = A * _safe_exp(-beta * t) * np.cos(omega * t + phi)
    return result


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
