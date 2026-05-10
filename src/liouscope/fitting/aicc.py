"""Small-sample-corrected AIC with N_eff (anchor H).

For a model with ``k`` parameters fitted to ``n`` observations with effective
sample size ``N_eff <= n``, the corrected AIC is

    AICc = -2 ln L_max + 2 k + 2 k (k + 1) / (N_eff - k - 1)

When ``N_eff - k - 1 <= 0`` we report ``inf`` to signal that the data does
not support the model.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


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
    """Gaussian log-likelihood for ``y - y_hat``.

    Uses MLE sigma if ``sigma`` is omitted.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    rss = float(np.dot(residuals, residuals))
    if sigma is None:
        sigma_sq = max(rss / n, 1.0e-30)
    else:
        sigma_sq = sigma * sigma
    return -0.5 * n * (np.log(2.0 * np.pi * sigma_sq) + rss / (n * sigma_sq))


def choose_model(aiccs: Mapping[str, float]) -> str:
    """Return the model name with the smallest finite AICc.

    Falls back to ``M0`` when every entry is ``inf`` or ``nan``.
    """
    valid = {name: v for name, v in aiccs.items() if np.isfinite(v)}
    if not valid:
        return "M0"
    return min(valid, key=valid.__getitem__)
