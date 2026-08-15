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
# Capping the exponent from ABOVE keeps every model finite over the whole real
# line while changing nothing in the regime where models are actually fitted:
# a decaying model has exponent <= 0, so this bound is unreachable and the value
# is bit-identical to the unclipped one.
#
# The bound is 345, not the ~709 where ``np.exp`` itself overflows, because
# saturating exp is not sufficient on its own: M3a multiplies it by the
# polynomial prefactor ``(A + B t)``, so a clip at 709 turns an exp overflow
# into a *multiply* overflow one line later. ``exp(345) ~ 4.6e149`` leaves ~158
# decades of headroom for the prefactor, which covers any t and B a fit can
# plausibly see (``B t ~ 5e8`` on the longest grid used here).
#
# The cap is deliberately ONE-SIDED. Clipping the negative side too would
# corrupt genuine, perfectly representable decay: ``exp(-400) = 1.9e-174``
# would be reported as ``exp(-345) = 1.5e-150``, a factor of 1e24 too large,
# growing past 1e150 by exponent -700. Every model would acquire an artificial
# constant tail that distorts residuals, fitted offsets and AICc on
# high-dynamic-range trajectories -- trading an overflow for a silent bias.
# Underflow needs no guard: it is exact in the limit (``exp(-800) -> 0.0``) and
# NumPy's default error state ignores it, so no warning is promoted.
_EXP_CLIP: float = 345.0

# Capping the exponent bounds ``exp``, but not the MODEL: M3a multiplies it by
# ``(A + B t)``, whose magnitude is unbounded in the parameters, so on a long
# grid the product still reaches ~1e160 -- and least-squares then squares it
# inside its own normal equations, overflowing in SciPy's ``trf`` rather than
# in this module. Bounding the returned value closes that off independently of
# the prefactor structure: any model value a real fit produces is of order the
# data (O(1) here), so 1e100 is unreachable in the well-conditioned regime and
# still leaves ~1e108 of headroom for the solver's squared quantities.
_MODEL_CLIP: float = 1.0e100


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """``exp(x)`` with the exponent capped from above only.

    Bit-identical to ``np.exp`` for every ``x <= 345``, including the whole
    decaying range; saturating instead of overflowing to ``inf`` above it.
    """
    result: np.ndarray = np.exp(np.minimum(x, _EXP_CLIP))
    return result


def _bounded(result: np.ndarray) -> np.ndarray:
    """Clamp a model evaluation to a magnitude the optimiser can square safely.

    Symmetric, unlike :func:`_safe_exp`: model values are legitimately negative
    (M3b oscillates), and the bound is about the magnitude the solver must
    handle, not about the direction of decay.

    Non-finite intermediates are mapped to the bound rather than clipped: the
    factor MULTIPLICATION can itself overflow before this function sees the
    product (an unconstrained fit can probe an amplitude like ``1e200``, and
    ``1e200 * exp(345)`` is ``inf`` on arrival), and ``0 * inf`` yields NaN.
    Both mean "this probe is absurdly far from the data", so the honest finite
    encoding is the saturation value the optimiser steps away from — models
    are evaluated under a suppressed overflow errstate for the same reason.
    """
    out: np.ndarray = np.nan_to_num(
        result, nan=_MODEL_CLIP, posinf=_MODEL_CLIP, neginf=-_MODEL_CLIP
    )
    clipped: np.ndarray = np.clip(out, -_MODEL_CLIP, _MODEL_CLIP)
    return clipped


def M0(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A * exp(-alpha t)``; params = (A, alpha)."""
    A, alpha = params
    with np.errstate(over="ignore", invalid="ignore"):
        return _bounded(A * _safe_exp(-alpha * t))


def M1(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A * exp(-alpha t) + C``; params = (A, alpha, C)."""
    A, alpha, C = params
    with np.errstate(over="ignore", invalid="ignore"):
        return _bounded(A * _safe_exp(-alpha * t) + C)


def M2(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A1 exp(-beta1 t) + A2 exp(-beta2 t)``; params = (A1, beta1, A2, beta2)."""
    A1, beta1, A2, beta2 = params
    with np.errstate(over="ignore", invalid="ignore"):
        return _bounded(A1 * _safe_exp(-beta1 * t) + A2 * _safe_exp(-beta2 * t))


def M3a(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``(A + B t) exp(-alpha t)``; params = (A, B, alpha)."""
    A, B, alpha = params
    with np.errstate(over="ignore", invalid="ignore"):
        return _bounded((A + B * t) * _safe_exp(-alpha * t))


def M3b(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """``A exp(-beta t) cos(omega t + phi)``; params = (A, beta, omega, phi)."""
    A, beta, omega, phi = params
    with np.errstate(over="ignore", invalid="ignore"):
        return _bounded(A * _safe_exp(-beta * t) * np.cos(omega * t + phi))


# Seed-rate floor as a DIMENSIONLESS decay depth over the fitted window, not an
# absolute rate (issue #108 class, fitting path). A decay rate is 1/time, so an
# absolute floor is tied to a choice of time unit: on a grid spanning t = 5e10 --
# which is exactly what a slow generator in small rate units produces -- the old
# absolute ``1e-3`` seed sat 1e7 above the true rate, placing the optimiser in a
# region where ``exp(-alpha t)`` has underflowed flat and offers no gradient. The
# fit then returned the floor itself as the "measured" rate, corrupting D5/D17
# while every other diagnostic looked healthy.
#
# ``alpha_floor = ALPHA_SEED_FLOOR_FRAC / t_span`` is the rate that decays by
# ~0.5 % across the whole window: slower than that is not distinguishable from a
# constant on this grid, which is the honest lower end for a SEED. It reproduces
# the historical ``1e-3`` exactly on the ``t_span = 5`` grids used throughout the
# suite, so seeds there are unchanged.
ALPHA_SEED_FLOOR_FRAC: float = 5.0e-3


def _time_span(t: np.ndarray) -> float:
    """Positive extent of a time grid, or ``0.0`` when it has none."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 0.0
    span = float(np.max(t) - np.min(t))
    return span if np.isfinite(span) and span > 0.0 else 0.0


def _alpha_seed_floor(t: np.ndarray) -> float:
    """Smallest seed rate worth proposing on this grid (see the note above)."""
    span = _time_span(t)
    return ALPHA_SEED_FLOOR_FRAC / span if span > 0.0 else 1.0e-3


def initial_guess_m0(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-linear regression on positive samples (FIX-5).

    The positivity floor on the seeded rate scales with the time grid, so the
    seed tracks a pure change of rate units instead of pinning the optimiser to
    an absolute rate that may lie orders of magnitude above the true one.
    """
    y = np.asarray(y, dtype=float)
    floor = _alpha_seed_floor(t)
    mask = y > 0
    if mask.sum() < 2:
        # Fallback rate is likewise grid-relative: ~one e-folding across the
        # window, the natural "no information" scale for this data.
        span = _time_span(t)
        fallback_alpha = 1.0 / span if span > 0.0 else 1.0
        return np.array([y[0] if y.size else 1.0, max(fallback_alpha, floor)])
    log_y = np.log(y[mask])
    A_design = np.vstack([np.ones_like(log_y), t[mask]]).T
    coefs, *_ = np.linalg.lstsq(A_design, log_y, rcond=None)
    A_est = float(np.exp(coefs[0]))
    alpha_est = float(-coefs[1])
    return np.array([A_est, max(alpha_est, floor)])


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
