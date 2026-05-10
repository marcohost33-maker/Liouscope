"""Prony-method seed initialisation for M3b (oscillatory-exponential fit).

The Prony method estimates a complex damping rate ``s = beta + i omega`` and
amplitude from a uniformly-sampled signal by fitting an LP-style polynomial
to the autoregression coefficients of the data, then root-finding.

Used to seed the non-linear M3b fit which is highly sensitive to the
initial omega guess (Patch v2 fix-pack, Spec Teil 13.1).
"""

from __future__ import annotations

import numpy as np


def prony_seed(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return a Prony seed ``(A, beta, omega, phi)`` for M3b.

    Assumes uniform sampling; if ``t`` is non-uniform we fall back to a
    decaying-oscillator guess.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 6:
        return float(y[0] if y.size else 1.0), 1.0, 1.0, 0.0
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-4):
        amp = float(np.max(np.abs(y)))
        return amp, 1.0, 1.0, 0.0

    p = 2  # two complex exponentials (oscillatory pair)
    N = y.size
    if 2 * p + 2 >= N:
        return float(y[0]), 1.0, 1.0, 0.0

    # Build the Hankel system: y[p+1:N] = -sum_k a_k y[p+1-k:N-k]
    A = np.empty((N - p, p))
    b = -y[p:N]
    for k in range(p):
        A[:, k] = y[p - 1 - k : N - 1 - k]
    coefs, *_ = np.linalg.lstsq(A, b, rcond=None)
    poly = np.concatenate(([1.0], coefs))
    roots = np.roots(poly)
    # Pick a conjugate pair (or the closest-to-unit-circle root)
    roots = roots[np.argsort(-np.abs(np.angle(roots)))]
    if roots.size == 0:
        return float(y[0]), 1.0, 1.0, 0.0
    z = roots[0]
    if abs(z) < 1.0e-12 or not np.isfinite(z):
        return float(y[0]), 1.0, 1.0, 0.0
    # ``log`` on the complex root is well-defined; cast to complex first to
    # avoid a numpy warning when the principal branch crosses the cut.
    s = np.log(complex(z)) / dt
    beta = float(-s.real)
    omega = float(abs(s.imag))
    # Amplitude / phase via least-squares on cos(omega t + phi) e^{-beta t}
    envelope = np.exp(-beta * t).reshape(-1, 1)
    basis = envelope * np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    rhs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    A_est = float(np.hypot(rhs[0], rhs[1]))
    phi = float(np.arctan2(-rhs[1], rhs[0]))
    if A_est < 1.0e-12:
        A_est = float(np.max(np.abs(y)))
    return A_est, max(beta, 1.0e-6), max(omega, 1.0e-6), phi
