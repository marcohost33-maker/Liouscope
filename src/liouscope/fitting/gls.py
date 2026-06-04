"""Generalised least squares with AR(1) residual covariance.

The relevant likelihood is

    y_t = f(t; theta) + eps_t,    eps_t = rho eps_{t-1} + nu_t

where ``nu_t`` is iid Gaussian with variance ``sigma^2``. We fit ``theta``
by minimising the AR(1) whitened residual sum of squares, then estimate
``(rho, sigma^2)`` from the residuals of the latest fit. Iterate two
rounds (Cochrane-Orcutt style).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .aicc import gaussian_log_likelihood
from .neff import ar1_correlation


@dataclass(frozen=True, slots=True)
class GLSFitOutput:
    params: np.ndarray
    residuals: np.ndarray
    rho_ar1: float
    sigma: float
    log_likelihood: float
    success: bool


def _whiten(y: np.ndarray, rho: float) -> np.ndarray:
    if y.size < 2:
        y_copy: np.ndarray = y.copy()
        return y_copy
    out = np.empty_like(y)
    out[0] = np.sqrt(max(1.0 - rho * rho, 1.0e-12)) * y[0]
    out[1:] = y[1:] - rho * y[:-1]
    return out


def fit_gls_ar1(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    *,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    n_iters: int = 3,
    max_nfev: int = 2000,
) -> GLSFitOutput:
    """Fit ``y = model(t, theta) + eps`` with AR(1) residuals.

    Parameters
    ----------
    model
        Callable ``f(t, theta) -> y_hat``.
    t, y
        Observation grid and values.
    p0
        Initial parameter vector.
    bounds
        Optional ``(lo, hi)`` arrays for ``least_squares``.
    n_iters
        Number of Cochrane-Orcutt-style iterations.
    max_nfev
        Max function evaluations per inner least-squares solve.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p0, dtype=float).copy()
    rho = 0.0
    success = True
    for _ in range(n_iters):
        def residual(params: np.ndarray, rho_local: float = rho) -> np.ndarray:
            y_hat = model(t, params)
            r = y - y_hat
            return _whiten(r, rho_local)

        ls_kwargs: dict[str, object] = {"max_nfev": max_nfev}
        if bounds is not None:
            ls_kwargs["bounds"] = bounds
        try:
            result = least_squares(residual, p, **ls_kwargs)
            p = result.x
            success = result.success
        except (ValueError, RuntimeError):
            success = False
            break
        y_hat = model(t, p)
        residuals_raw = y - y_hat
        rho = ar1_correlation(residuals_raw)
        rho = float(np.clip(rho, -0.999, 0.999))

    y_hat_final = model(t, p)
    residuals_raw = y - y_hat_final
    whitened = _whiten(residuals_raw, rho)
    n = whitened.size
    sigma = float(np.sqrt(max(np.dot(whitened, whitened) / max(n, 1), 1.0e-30)))
    log_lik = gaussian_log_likelihood(whitened, sigma=sigma)
    return GLSFitOutput(
        params=p,
        residuals=residuals_raw,
        rho_ar1=rho,
        sigma=sigma,
        log_likelihood=log_lik,
        success=success,
    )
