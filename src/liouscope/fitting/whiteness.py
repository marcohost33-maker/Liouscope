"""Residual-whiteness gate via the Ljung-Box Q statistic (LIOU-A-013).

After a fit from the M0-M3b hierarchy, the residuals should be *white*
(serially uncorrelated). Structured (non-white) residuals signal an
under-specified model -- leftover autocorrelation the fit failed to capture --
and must block the claim. This sharpens the existing AR(1) treatment
(:mod:`liouscope.fitting.gls`) into an explicit acceptance gate.

The Ljung-Box statistic over the first ``m`` lags is

    Q = N (N + 2) sum_{k=1}^{m} rho_k^2 / (N - k)

with ``rho_k`` the lag-``k`` sample autocorrelation of the residuals
(Ljung, G. M. & Box, G. E. P., "On a measure of lack of fit in time series
models", Biometrika 65(2), 297-303, 1978).

**Degrees of freedom.** For an *observed* (non-fitted) series the reference is
``Q ~ chi^2_m``. When the residuals come from a model with ``p`` estimated
parameters, the standard correction subtracts them: ``Q ~ chi^2_{m - p}`` (for
ARMA(p_ar, q_ma) residuals ``p = p_ar + q_ma``; this is the classical Box-Pierce
/ Ljung-Box adjustment). This module exposes ``n_params`` so the caller chooses
``p`` explicitly. The **default is ``n_params = 0``** (``dof = m``), the
conservative, fail-closed choice: it never *fabricates* extra rejection power by
over-subtracting degrees of freedom. For an M0-M3b fit residual, pass the fitted
parameter count (e.g. ``n_params=2`` for M0). Residuals are declared white iff
``Q <= chi^2_{1-alpha, dof}`` (equivalently ``p_value > alpha``); the
chi-squared critical value comes from :mod:`scipy.stats`, an independent oracle
for the reference distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class WhitenessResult:
    """Outcome of the Ljung-Box whiteness gate.

    Attributes
    ----------
    q_stat
        The Ljung-Box Q statistic.
    dof
        Degrees of freedom ``m - n_params`` of the reference chi-squared.
    p_value
        Right-tail probability ``P(chi^2_dof > Q)``.
    is_white
        ``True`` iff ``p_value > alpha`` (fail to reject whiteness).
    """

    q_stat: float
    dof: int
    p_value: float
    is_white: bool


def _autocorrelations(x: np.ndarray, m: int) -> np.ndarray:
    """Biased sample autocorrelations ``rho_1..rho_m`` of a 1-D series."""
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    # Scale-aware zero-variance guard. A (near-)constant series has centred
    # energy ``denom`` that is only round-off relative to its raw energy
    # ``scale``; without this relative test the float residue of ``x - mean``
    # would be *normalised* into spurious O(1) autocorrelations (a constant
    # would look strongly correlated). Treat it as perfectly white instead.
    scale = float(np.dot(x, x))
    if denom <= 1e-20 * max(scale, 1.0):
        return np.zeros(m, dtype=float)
    out = np.empty(m, dtype=float)
    for k in range(1, m + 1):
        out[k - 1] = float(np.dot(xc[: n - k], xc[k:]) / denom)
    return out


def ljung_box_q(
    residuals: np.ndarray,
    *,
    m: int = 10,
    n_params: int = 0,
) -> WhitenessResult:
    """Compute the Ljung-Box Q statistic and whiteness verdict.

    Parameters
    ----------
    residuals
        1-D array of fit residuals.
    m
        Number of lags to test. Must satisfy ``n_params < m < N``.
    n_params
        Number of fitted model parameters (reduces the chi-squared dof).

    Raises
    ------
    ValueError
        If ``residuals`` is not 1-D, if ``m`` is not a positive integer
        strictly less than the sample size, or if ``m <= n_params`` (which
        would leave a non-positive degrees-of-freedom and an undefined
        reference distribution).
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1:
        raise ValueError(f"residuals must be 1-D, got ndim {residuals.ndim}")
    n = residuals.size
    if m < 1:
        raise ValueError(f"m must be a positive integer, got {m}")
    if m >= n:
        raise ValueError(
            f"m must be strictly less than the sample size N={n}, got m={m}"
        )
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, got {n_params}")
    dof = m - n_params
    if dof < 1:
        raise ValueError(
            f"degrees of freedom m - n_params = {dof} must be >= 1; "
            f"increase m (got {m}) above n_params (got {n_params})."
        )
    rho = _autocorrelations(residuals, m)
    ks = np.arange(1, m + 1)
    q_stat = float(n * (n + 2) * np.sum(rho**2 / (n - ks)))
    p_value = float(stats.chi2.sf(q_stat, dof))
    return WhitenessResult(
        q_stat=q_stat,
        dof=dof,
        p_value=p_value,
        is_white=bool(p_value > 0.05),
    )


def whiteness_gate(
    residuals: np.ndarray,
    *,
    m: int = 10,
    n_params: int = 0,
    alpha: float = 0.05,
) -> WhitenessResult:
    """Whiteness acceptance gate at significance ``alpha``.

    Returns a :class:`WhitenessResult` whose ``is_white`` field uses the
    caller-supplied ``alpha`` (``ljung_box_q`` fixes ``alpha = 0.05``).

    Raises
    ------
    ValueError
        If ``alpha`` is not in the open interval ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    base = ljung_box_q(residuals, m=m, n_params=n_params)
    return WhitenessResult(
        q_stat=base.q_stat,
        dof=base.dof,
        p_value=base.p_value,
        is_white=bool(base.p_value > alpha),
    )


__all__ = ["WhitenessResult", "ljung_box_q", "whiteness_gate"]
