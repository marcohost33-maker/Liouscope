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

import warnings

import numpy as np

# Below this sample size the lag-1 autocorrelation estimator is materially
# downward-biased even after first-order correction; callers are warned so
# AR(1)-whitened CIs are not silently over-confident (S2 audit 2026-06-04).
_AR1_SMALL_N: int = 40


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
    """Return the (raw, uncorrected) lag-1 autocorrelation ``rho_1``.

    This is the biased plug-in estimator. For AR(1) whitening / CI work use
    :func:`ar1_correlation_corrected`, which applies a small-sample
    bias-correction; the raw estimator is downward-biased by ~``-(1+3 rho)/n``
    (Marriott & Pope 1954; Kendall 1954), so the raw value yields
    over-confident confidence intervals at small ``n``.
    """
    x = np.asarray(residuals, dtype=float) - float(np.mean(residuals))
    if x.size < 2:
        return 0.0
    num = float(np.dot(x[:-1], x[1:]))
    den = float(np.dot(x, x))
    if den == 0.0:
        return 0.0
    return float(num / den)


def ar1_correlation_corrected(
    residuals: np.ndarray, *, warn_small_n: bool = True
) -> float:
    """Return a small-sample bias-corrected lag-1 autocorrelation.

    The raw plug-in estimator ``rho_hat`` (see :func:`ar1_correlation`) is
    downward-biased at small ``n`` (E[rho_hat] - rho ~ -(1 + 3 rho)/n), which
    makes any AR(1)-whitened variance / confidence interval too narrow. We
    apply the closed-form first-order correction

        rho_corr = (rho_hat * (n - 1) + 1) / (n - 3)

    which removes the leading O(1/n) bias term while staying inside a sensible
    range. The corrected value is clipped to ``[-0.999, 0.999]`` so downstream
    whitening (``sqrt(1 - rho^2)``) stays well defined.

    Validity / range of trust
    -------------------------
    This is a *first-order* correction (Kendall / Marriott-Pope family): it
    cancels only the leading ``O(1/n)`` term of the bias expansion. It is
    reliable up to roughly ``rho ~ 0.85``; beyond that a residual bias remains,
    set by the (truncated) order of the series expansion, and grows as ``rho``
    approaches 1. For strongly autocorrelated series (``rho`` near 1) treat the
    returned value as a mildly conservative estimate and prefer the
    Geyer-IPS-based ``N_eff`` (:func:`estimate_neff_geyer`) for variance work.

    A higher-order Kendall variant was tested and is marginally more accurate
    at high ``rho`` / large ``n`` (smaller residual bias near ``rho ~ 0.9``),
    but it was **deliberately not adopted**: the audit pins this first-order
    closed form (item 3 / S2 audit) so the AR(1)-whitening path stays
    bit-stable and auditable across versions. The marginal accuracy gain does
    not justify breaking the frozen audit formula.

    References
    ----------
    Marriott & Pope, "Bias in the estimation of autocorrelations",
    *Biometrika* 41, 390 (1954); Kendall, "Note on bias in the estimation of
    autocorrelation", *Biometrika* 41, 403 (1954); bias-correction survey
    arXiv:2010.05870; Dou et al., "A review of bias-correction methods for the
    first-order autoregressive coefficient", *British Journal of Mathematical
    and Statistical Psychology* (2026).

    Parameters
    ----------
    residuals
        Residual series.
    warn_small_n
        If True (default), emit a :class:`RuntimeWarning` when
        ``n <= 40`` because the residual bias after correction is still
        non-negligible there and the CI may remain mildly optimistic.
    """
    x = np.asarray(residuals, dtype=float)
    n = x.size
    rho_hat = ar1_correlation(x)
    if n < 4:
        # (n - 3) <= 0: correction undefined; fall back to the raw estimate.
        return float(np.clip(rho_hat, -0.999, 0.999))
    rho_corr = (rho_hat * (n - 1) + 1.0) / (n - 3.0)
    if warn_small_n and n <= _AR1_SMALL_N:
        warnings.warn(
            f"AR(1) bias-corrected rho estimated from only n={n} residuals "
            f"(<= {_AR1_SMALL_N}): residual downward bias remains, so "
            "AR(1)-whitened confidence intervals may be mildly over-confident.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(np.clip(rho_corr, -0.999, 0.999))
