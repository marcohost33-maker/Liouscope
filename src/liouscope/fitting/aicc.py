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
        standardised_rss = 0.0
    elif log_rss == float("inf"):
        return float("-inf")
    else:
        log_standardised_rss = log_rss - 2.0 * log_sigma
        if log_standardised_rss > math.log(np.finfo(float).max):
            return float("-inf")
        standardised_rss = math.exp(log_standardised_rss)
    return float(
        -0.5 * n * log_2pi - n * log_sigma - 0.5 * standardised_rss
    )


def choose_model(aiccs: Mapping[str, float]) -> str:
    """Return the model name with the smallest finite AICc.

    Falls back to ``M0`` when every entry is ``inf`` or ``nan``.
    """
    valid = {name: v for name, v in aiccs.items() if np.isfinite(v)}
    if not valid:
        return "M0"
    return min(valid, key=valid.__getitem__)
