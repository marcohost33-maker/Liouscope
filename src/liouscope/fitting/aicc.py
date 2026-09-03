"""Small-sample-corrected AIC with N_eff (anchor H).

For a model with ``k`` parameters fitted to ``n`` observations with effective
sample size ``N_eff <= n``, the corrected AIC is

    AICc = -2 ln L_max + 2 k + 2 k (k + 1) / (N_eff - k - 1)

When ``N_eff - k - 1 <= 0`` we report ``inf`` to signal that the data does
not support the model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..numerics.norms import scaled_log_sum_squares


def aicc(log_likelihood: float, k: int, n_eff: float) -> float:
    """Return ``AICc`` for given log-likelihood, parameter count, and N_eff."""
    if n_eff - k - 1 <= 0:
        return float("inf")
    correction = 2.0 * k * (k + 1) / (n_eff - k - 1)
    return float(-2.0 * log_likelihood + 2.0 * k + correction)


def gaussian_log_likelihood(
    residuals: np.ndarray,
    *,
    sigma: float | None = None,
) -> float:
    """Gaussian log-likelihood for ``y - y_hat`` without absolute RSS floors.

    When ``sigma`` is omitted, evaluate the profile likelihood at the Gaussian
    MLE ``sigma_hat**2 = RSS / n`` directly in log-RSS space. Exact zero RSS has
    no finite interior MLE for the positive scale parameter and therefore
    returns NaN (model-selection likelihood unavailable) instead of inventing
    an absolute epsilon variance. With an explicitly supplied finite positive
    ``sigma``, zero residuals remain a valid finite likelihood.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if n == 0:
        return float("nan")

    log_rss = scaled_log_sum_squares(residuals)
    if math.isnan(log_rss):
        return float("nan")

    log_2pi = math.log(2.0 * math.pi)
    if sigma is None:
        if log_rss == float("-inf"):
            return float("nan")
        if log_rss == float("inf"):
            return float("-inf")
        return float(-0.5 * n * (log_2pi + 1.0 + log_rss - math.log(n)))

    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return float("nan")
    log_sigma = math.log(sigma)
    if log_rss == float("-inf"):
        half_standardised_rss = 0.0
    elif log_rss == float("inf"):
        return float("-inf")
    else:
        log_standardised_rss = log_rss - 2.0 * log_sigma
        # ROUND-2 REVIEW (PR #147). The standardised RSS never appears on its
        # own: it enters the result as ``-0.5 * RSS``. Its representable range
        # therefore reaches ``2 * float64.max``, and comparing against
        # ``log(float64.max)`` refused an entire octave of perfectly finite
        # likelihoods. Measured: ``sigma = 1`` with a single residual near
        # 1.4e154 has RSS ~1.96e308 and a log-likelihood of ~-9.8e307, and
        # this guard returned ``-inf`` -- which drops the model from AICc
        # selection on an arithmetic accident rather than on the data, the
        # same absolute-scale boundary issue #135 exists to remove.
        #
        # The one-half factor is now applied BEFORE the decision: the bound
        # carries ``+ log(2)`` and, in the octave that only the halved value
        # can represent, the exponential is taken after subtracting
        # ``log(2)``. Below that octave the arithmetic is left exactly as it
        # was (``0.5 * exp(...)``), so no previously computed likelihood
        # changes by even one ulp -- widening a range must not perturb the
        # values already inside it.
        log_max = math.log(np.finfo(float).max)
        if log_standardised_rss > log_max + math.log(2.0):
            return float("-inf")
        if log_standardised_rss > log_max:
            half_standardised_rss = math.exp(log_standardised_rss - math.log(2.0))
        else:
            half_standardised_rss = 0.5 * math.exp(log_standardised_rss)
    return float(
        -0.5 * n * log_2pi - n * log_sigma - half_standardised_rss
    )


def choose_model(aiccs: Mapping[str, float]) -> str:
    """Return the model name with the smallest finite AICc.

    Falls back to ``M0`` when every entry is ``inf`` or ``nan``.
    """
    valid = {name: v for name, v in aiccs.items() if np.isfinite(v)}
    if not valid:
        return "M0"
    return min(valid, key=valid.__getitem__)
