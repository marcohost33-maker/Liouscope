"""Prony-method seed initialisation for M3b (oscillatory-exponential fit).

The Prony method estimates a complex damping rate ``s = beta + i omega`` and
amplitude from a uniformly-sampled signal by fitting an LP-style polynomial
to the autoregression coefficients of the data, then root-finding.

Used to seed the non-linear M3b fit which is highly sensitive to the
initial omega guess (Patch v2 fix-pack, Spec Teil 13.1).
"""

from __future__ import annotations

import warnings

import numpy as np

# Fallback seed rates are GRID-RELATIVE (sixteenth review round; the #108
# class of defect, same pattern as ``ALPHA_SEED_FLOOR_FRAC`` /
# ``M3A_SLOPE_SEED_FRAC`` in ``fitting.models``). ``beta`` and ``omega`` carry
# rate dimension, so the former absolute fallback ``(1.0, 1.0)`` was tied to a
# unit choice: on a valid non-uniform grid spanning ``t = 1e7`` (where Prony
# falls back by design) the M3b fit was seeded seven orders of magnitude off,
# and least-squares "converged" (``success=True``) onto the seed itself --
# measured ``[A, beta, omega] ~ [0.006, 1, 1]`` for a curve whose true values
# were ``5e-8`` and ``2e-6`` -- corrupting the fitted rate and suppressing the
# M3b/A8 hypothesis purely because of the rate unit. ``frac = 5.0``
# (~5 e-foldings / ~0.8 oscillation periods across the window) reproduces the
# historical ``1.0`` exactly on the canonical ``t_span = 5`` grids, so seeds
# there, and the anchors, are unchanged. The positivity floors on the success
# path scale identically (``5e-6 / t_span`` == the historical ``1e-6`` at
# ``t_span = 5``).
_FALLBACK_RATE_FRAC = 5.0
_RATE_FLOOR_FRAC = 5.0e-6
# Shortest uniform prefix worth a Prony estimate: the Hankel system needs
# N > 2p + 2 = 6 rows for the model order p = 2 used below, so 8 leaves a
# margin of two rows rather than solving a square, interpolating system.
_MIN_UNIFORM_PREFIX = 8


def _time_span(t: np.ndarray) -> float:
    """Positive extent of the time grid, or ``0.0`` when it has none."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 0.0
    span = float(np.max(t) - np.min(t))
    return span if np.isfinite(span) and span > 0.0 else 0.0


def _fallback_rate(t: np.ndarray) -> float:
    """Grid-relative fallback rate for both ``beta`` and ``omega``."""
    span = _time_span(t)
    return _FALLBACK_RATE_FRAC / span if span > 0.0 else 1.0


def _rate_floor(t: np.ndarray) -> float:
    """Grid-relative positivity floor for the estimated rates."""
    span = _time_span(t)
    return _RATE_FLOOR_FRAC / span if span > 0.0 else 1.0e-6



def _uniform_prefix_length(t: np.ndarray) -> int:
    """Number of leading samples of ``t`` that are uniformly spaced.

    Uses the same ``rtol=1e-4`` uniformity convention as the whole-grid test
    in :func:`prony_seed`, so a grid that passes that test yields its own
    length here.
    """
    diffs = np.diff(np.asarray(t, dtype=float))
    if diffs.size == 0:
        return int(np.asarray(t).size)
    same = np.isclose(diffs, diffs[0], rtol=1.0e-4, atol=0.0)
    if bool(np.all(same)):
        return int(diffs.size + 1)
    return int(np.argmin(same)) + 1


def _default_seed(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Conservative decaying-oscillator seed used as a safe fallback."""
    amp = float(np.max(np.abs(y))) if y.size else 1.0
    if not np.isfinite(amp) or amp <= 0.0:
        amp = 1.0
    rate = _fallback_rate(t)
    return amp, rate, rate, 0.0


def prony_seed(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return a Prony seed ``(A, beta, omega, phi)`` for M3b.

    Assumes uniform sampling. If ``t`` is non-uniform the estimate is taken
    from its longest uniform PREFIX when that prefix is long enough for the
    Hankel system (the two-scale relaxation grid always is), and only
    otherwise from a grid-relative decaying-oscillator guess. If the Hankel/least-squares solve or the
    companion-root step fails or produces a non-finite result (near-singular
    or degenerate data), we emit a ``RuntimeWarning`` and return the safe
    default seed instead of propagating the exception or garbage.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 6:
        rate = _fallback_rate(t)
        return float(y[0] if y.size else 1.0), rate, rate, 0.0
    dt = float(t[1] - t[0])
    # EIGHTEENTH REVIEW ROUND (external, PR #127). ``atol=0.0`` is the whole
    # point of this line. NumPy's default ``atol=1e-8`` is an ABSOLUTE time,
    # so on a grid expressed in small time units (the reviewer's case:
    # ``gap=1e8, fast_rate=1e12``, nanosecond-scale steps) it declares wildly
    # varying steps uniform, the two-scale prefix path below never runs, and
    # Prony is applied straight across the discontinuity. Measured effect on
    # an exact M3b curve with dimensionless ``(beta, omega) = (0.0387,
    # 0.0295)``: unscaled recovers both parameters, the 1e12 rate-unit
    # rescaling converges to ``omega/c ~ 5e-11`` -- the M3b/A8 verdict
    # changing on the choice of units alone.
    #
    # No new constant: ``_uniform_prefix_length`` above already spells the
    # same convention ``rtol=1.0e-4, atol=0.0``, and the two tests are
    # documented as sharing it. They now actually do.
    if not np.allclose(np.diff(t), dt, rtol=1e-4, atol=0.0):
        # Seventeenth review round (external, PR #127). Bailing out to a
        # seed derived from the total SPAN is a regression on exactly the
        # grid this branch now sees most often: the two-scale grid of
        # ``default_relaxation_grid`` carries a UNIFORM fine head whose whole
        # purpose is to resolve the fast dynamics, and the span-derived rate
        # lives on the slow gap scale instead. Measured on
        # ``default_relaxation_grid(1e-4, fast_rate=1)`` (80 points, uniform
        # head of 41 at dt = 0.25, total span 1e5): the span seed is
        # ``5e-5`` for both rates, and over 60 random exact curves with
        # ``beta`` in ``[0.05, 2]``, ``omega`` in ``[0.1, 10]`` the M3b fit
        # then converged to a spurious solution in 2 cases (e.g. true
        # ``omega = 1.68`` reported as ``0.037``) where the historical
        # ``(1, 1)`` seed recovered the exact parameters in all 60. On a
        # slower family (``beta``, ``omega`` in ``[1e-4, 1e-1]``) the span
        # seed failed 41 of 60.
        #
        # Estimating from the longest UNIFORM PREFIX fixes both: Prony is
        # valid there by construction, the estimate is data-driven rather
        # than geometric, and the same two families give 0 of 60 and 0 of 60
        # failures. A prefix shorter than ``_MIN_UNIFORM_PREFIX`` carries no
        # usable Hankel system, so the grid-relative fallback stands.
        head = _uniform_prefix_length(t)
        if head >= _MIN_UNIFORM_PREFIX:
            return prony_seed(t[:head], y[:head])
        amp = float(np.max(np.abs(y)))
        rate = _fallback_rate(t)
        return amp, rate, rate, 0.0

    p = 2  # two complex exponentials (oscillatory pair)
    N = y.size
    if 2 * p + 2 >= N:
        rate = _fallback_rate(t)
        return float(y[0]), rate, rate, 0.0

    # Build the Hankel system: y[p+1:N] = -sum_k a_k y[p+1-k:N-k]
    A = np.empty((N - p, p))
    b = -y[p:N]
    for k in range(p):
        A[:, k] = y[p - 1 - k : N - 1 - k]
    # lstsq on a (near-)singular Hankel matrix and np.roots on the resulting
    # companion polynomial can raise (LinAlgError) or return non-finite junk.
    # Guard the whole estimation block and fall back to the safe default seed.
    try:
        coefs, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(coefs)):
            raise np.linalg.LinAlgError("non-finite Prony coefficients")
        poly = np.concatenate(([1.0], coefs))
        roots = np.roots(poly)
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(
            f"Prony seed estimation failed ({exc}); using default seed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _default_seed(t, y)
    # Pick a conjugate pair (or the closest-to-unit-circle root)
    if roots.size == 0:
        return _default_seed(t, y)
    roots = roots[np.argsort(-np.abs(np.angle(roots)))]
    z = roots[0]
    if not np.isfinite(z) or abs(z) < 1.0e-12:
        return _default_seed(t, y)
    # ``log`` on the complex root is well-defined; cast to complex first to
    # avoid a numpy warning when the principal branch crosses the cut.
    s = np.log(complex(z)) / dt
    beta = float(-s.real)
    omega = float(abs(s.imag))
    if not (np.isfinite(beta) and np.isfinite(omega)):
        return _default_seed(t, y)
    # Amplitude / phase via least-squares on cos(omega t + phi) e^{-beta t}
    envelope = np.exp(-beta * t).reshape(-1, 1)
    basis = envelope * np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    rhs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    A_est = float(np.hypot(rhs[0], rhs[1]))
    phi = float(np.arctan2(-rhs[1], rhs[0]))
    if A_est < 1.0e-12:
        A_est = float(np.max(np.abs(y)))
    floor = _rate_floor(t)
    return A_est, max(beta, floor), max(omega, floor), phi
