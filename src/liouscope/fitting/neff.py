"""Effective sample size via Geyer 1992 initial-positive-sequence (IPS) estimator.

Anchor H: ODE trajectories are heavily autocorrelated. Standard AICc uses ``n``,
which on n=200, k<=5 yields a correction of only 0.1-0.4 -- a dramatic
under-correction. The IPS estimator gives

    N_eff = n / (1 + 2 * sum_{k>=1} rho_k)

where the autocorrelations ``rho_k`` are summed until the *initial positive
sequence* of consecutive lag-pair sums turns negative (Geyer 1992).

Reference: Geyer, "Practical Markov Chain Monte Carlo", Statistical Science
7(4), 473 (1992).
"""

from __future__ import annotations

import numpy as np


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation up to ``max_lag``."""
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    n = x.size
    var = float(np.dot(x, x) / n)
    if var == 0.0:
        return np.zeros(max_lag + 1)
    result = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            result[lag] = 1.0
        else:
            result[lag] = float(np.dot(x[:-lag], x[lag:]) / (n * var))
    return result


def estimate_neff_geyer(residuals: np.ndarray, *, max_lag: int | None = None) -> float:
    """Return ``N_eff`` for an autocorrelated residual series.

    Implements Geyer's initial-positive-sequence estimator: sums adjacent
    autocorrelation pairs ``Gamma_m = rho_{2m} + rho_{2m+1}`` until the pair
    turns non-positive, then uses ``N_eff = n / (1 + 2 * sum_m Gamma_m)``.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if n < 4:
        return float(n)
    if max_lag is None:
        max_lag = min(n // 2, 200)

    rho = _autocorr(residuals, max_lag)
    tau = 0.0
    m = 0
    while 2 * m + 1 <= max_lag:
        gamma_m = rho[2 * m] + rho[2 * m + 1] if 2 * m > 0 else rho[1]
        # Geyer initial positive sequence rule:
        if 2 * m > 0 and gamma_m <= 0:
            break
        if 2 * m == 0:
            tau += rho[1]
        else:
            tau += gamma_m
        m += 1
    # Add the lag-0 term (1) plus tau twice for symmetric lags.
    denom = max(1.0e-9, 1.0 + 2.0 * tau)
    n_eff = float(n / denom)
    return max(1.0, min(float(n), n_eff))


def ar1_correlation(residuals: np.ndarray) -> float:
    """Return the lag-1 autocorrelation ``rho_1``."""
    x = np.asarray(residuals, dtype=float) - float(np.mean(residuals))
    if x.size < 2:
        return 0.0
    num = float(np.dot(x[:-1], x[1:]))
    den = float(np.dot(x, x))
    if den == 0.0:
        return 0.0
    return float(num / den)
