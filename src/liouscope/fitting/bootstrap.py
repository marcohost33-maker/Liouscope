"""Parametric bootstrap with BCa confidence intervals.

Anchor G: standard iid bootstrap on ODE trajectories violates independence
(neighbouring time samples are heavily correlated). We instead:

1. Fit the model once (GLS+AR(1)).
2. Resimulate ``nu_t ~ N(0, sigma^2)`` and propagate through
   ``eps_t = rho eps_{t-1} + nu_t``.
3. Re-fit the model on the resimulated trajectory.
4. Aggregate fitted parameters and report BCa confidence intervals
   (Efron 1987).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import ndtr, ndtri

from .gls import fit_gls_ar1


def _ar1_resample(
    rng: np.random.Generator,
    n: int,
    rho: float,
    sigma: float,
) -> np.ndarray:
    eps = np.empty(n)
    nu = rng.normal(0.0, sigma, size=n)
    eps[0] = nu[0] / np.sqrt(max(1.0 - rho * rho, 1.0e-12))
    for i in range(1, n):
        eps[i] = rho * eps[i - 1] + nu[i]
    return eps


def parametric_bootstrap(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    *,
    B: int = 1000,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(theta_boot, theta_hat)``.

    ``theta_boot`` has shape ``(B, len(p0))``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    base = fit_gls_ar1(model, t, y, p0, bounds=bounds)
    theta_hat = base.params
    samples = np.empty((B, theta_hat.size))
    for b in range(B):
        eps_b = _ar1_resample(rng, t.size, base.rho_ar1, base.sigma)
        y_b = model(t, theta_hat) + eps_b
        fit_b = fit_gls_ar1(model, t, y_b, theta_hat, bounds=bounds)
        samples[b] = fit_b.params
    return samples, theta_hat


def _jackknife(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    theta_hat: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Leave-one-out jackknife on the time grid."""
    n = t.size
    out = np.empty((n, theta_hat.size))
    for i in range(n):
        t_i = np.delete(t, i)
        y_i = np.delete(y, i)
        fit_i = fit_gls_ar1(model, t_i, y_i, theta_hat, bounds=bounds)
        out[i] = fit_i.params
    return out


def bca_ci(
    samples: np.ndarray,
    theta_hat: np.ndarray,
    *,
    alpha: float = 0.05,
    jackknife_estimates: np.ndarray | None = None,
) -> np.ndarray:
    """Bias-corrected-and-accelerated bootstrap CI.

    Returns ``(p, 2)`` array of ``(lo, hi)`` per parameter at level
    ``1 - alpha``.
    """
    samples = np.asarray(samples, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    B, p = samples.shape
    cis = np.empty((p, 2))
    z_alpha_lo = ndtri(alpha / 2.0)
    z_alpha_hi = ndtri(1.0 - alpha / 2.0)
    for j in range(p):
        boot_j = np.sort(samples[:, j])
        # Bias-correction z0:
        prop_below = float(np.mean(boot_j < theta_hat[j]))
        prop_below = float(np.clip(prop_below, 1.0e-9, 1.0 - 1.0e-9))
        z0 = ndtri(prop_below)
        # Acceleration: from jackknife if provided, else default to 0.
        if jackknife_estimates is None:
            a = 0.0
        else:
            jk = jackknife_estimates[:, j]
            jk_mean = float(jk.mean())
            num = float(np.sum((jk_mean - jk) ** 3))
            den = 6.0 * float(np.sum((jk_mean - jk) ** 2)) ** 1.5
            a = float(num / den) if den != 0.0 else 0.0
        # Adjusted percentiles
        def adjusted(z_alpha: float, z0_: float = z0, a_: float = a) -> float:
            shifted = z0_ + (z0_ + z_alpha) / (1.0 - a_ * (z0_ + z_alpha))
            return float(ndtr(shifted))

        q_lo = adjusted(z_alpha_lo)
        q_hi = adjusted(z_alpha_hi)
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, 0.0, 1.0))
        idx_lo = int(np.clip(round(q_lo * (B - 1)), 0, B - 1))
        idx_hi = int(np.clip(round(q_hi * (B - 1)), 0, B - 1))
        cis[j] = (boot_j[idx_lo], boot_j[idx_hi])
    return cis


__all__ = ["_jackknife", "bca_ci", "parametric_bootstrap"]
