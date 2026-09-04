"""Generalised least squares with AR(1) residual covariance.

The relevant likelihood is

    y_t = f(t; theta) + eps_t,    eps_t = rho eps_{t-1} + nu_t

where ``nu_t`` is iid Gaussian with variance ``sigma^2``. We fit ``theta``
by minimising the AR(1) whitened residual sum of squares, then estimate
``(rho, sigma^2)`` from the residuals of the latest fit. Iterate two
rounds (Cochrane-Orcutt style).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .aicc import gaussian_log_likelihood
from .car1 import (
    estimate_car1_theta,
    is_uniform_grid,
    whiten_car1,
    whiten_car1_log_jacobian,
)
from .neff import _AR1_SMALL_N, ar1_correlation_corrected


@dataclass(frozen=True, slots=True)
class GLSFitOutput:
    """Result of :func:`fit_gls_ar1`.

    ``theta_car1`` disambiguates the two residual parametrisations, and with
    them the meaning of ``sigma``:

    * ``nan`` -- discrete AR(1) on a uniform grid (the historical path).
      ``rho_ar1`` is the fitted lag-1 correlation and ``sigma`` is the
      INNOVATION standard deviation.
    * finite -- continuous-time CAR(1) on a non-uniform grid. ``sigma`` is the
      STATIONARY standard deviation of the residual process (there is no single
      innovation variance when ``dt`` varies), and ``rho_ar1`` is the
      correlation over the MEAN step, reported for continuity of the field's
      meaning only -- the whitening uses the per-step ``exp(-theta dt_k)``.

    Consumers that resample or re-whiten (the parametric bootstrap) MUST branch
    on ``theta_car1``; reading ``sigma`` without it mixes the two conventions.
    """

    params: np.ndarray
    residuals: np.ndarray
    rho_ar1: float
    sigma: float
    log_likelihood: float
    success: bool
    theta_car1: float = float("nan")


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
    # Fail closed on corrupted input at the FIT boundary (eighth-round review):
    # the models saturate non-finite intermediates so that optimiser overflow
    # probes keep a finite, informative residual — but that same saturation
    # would otherwise launder a NaN in caller-supplied data into a "successful"
    # fit with a finite likelihood. Overflow recovery is for probes the
    # optimiser generates itself; data the caller hands in must be measurements.
    for name, arr in (("t", t), ("y", y), ("p0", p)):
        if not np.all(np.isfinite(arr)):
            bad = np.flatnonzero(~np.isfinite(arr)).tolist()
            raise ValueError(
                f"fit_gls_ar1: {name} must be finite; non-finite entries at "
                f"indices {bad}"
            )
    # Which residual model applies is a property of the GRID, decided once.
    # A uniform grid keeps the historical discrete AR(1) path bit-for-bit; a
    # grid whose step varies gets the continuous-time CAR(1) model, whose
    # per-step correlation exp(-theta dt_k) is the only whitening that is
    # actually valid there (see :mod:`liouscope.fitting.car1`).
    uniform = is_uniform_grid(t)
    rho = 0.0
    theta = float("nan")
    success = True
    for _ in range(n_iters):
        def residual(
            params: np.ndarray,
            rho_local: float = rho,
            theta_local: float = theta,
        ) -> np.ndarray:
            r = np.asarray(y - model(t, params), dtype=float)
            if uniform:
                return _whiten(r, rho_local)
            if not np.isfinite(theta_local):
                # First pass (and any pass after a failed estimate): plain
                # least squares, exactly as ``_whiten(r, 0.0)`` is on the
                # uniform path.
                return r
            return whiten_car1(r, t, theta_local)

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
        # S2 audit 2026-06-04: use the small-sample bias-corrected lag-1
        # autocorrelation. The raw plug-in estimator is downward-biased at
        # small n, which makes AR(1)-whitened CIs too narrow. Suppress the
        # per-iteration small-n warning here; we emit it once below so a
        # B-fold bootstrap does not raise B identical warnings.
        if uniform:
            rho = ar1_correlation_corrected(residuals_raw, warn_small_n=False)
        else:
            theta = estimate_car1_theta(t, residuals_raw)

    y_hat_final = model(t, p)
    residuals_raw = y - y_hat_final
    n_resid = residuals_raw.size
    if n_resid <= _AR1_SMALL_N:
        warnings.warn(
            f"GLS AR(1) fit on only n={n_resid} points (<= {_AR1_SMALL_N}): "
            "the bias-corrected rho still carries residual downward bias, so "
            "the reported confidence intervals may be mildly over-confident.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not uniform and not np.isfinite(theta):
        # CAR(1) estimation is fail-closed (NaN on a degenerate series). Fall
        # back to the historical AR(1) treatment rather than shipping
        # unwhitened residuals as if they were white -- the fallback is
        # imperfect on a varying grid, but it is the documented status quo,
        # whereas rho = 0 would silently claim independence.
        rho = ar1_correlation_corrected(residuals_raw, warn_small_n=False)

    if np.isfinite(theta):
        whitened = whiten_car1(residuals_raw, t, theta)
        jac = whiten_car1_log_jacobian(t, theta)
        rho = float(np.exp(-theta * float(np.mean(np.diff(t)))))
    else:
        whitened = _whiten(residuals_raw, rho)
        # Prais-Winsten exact AR(1) likelihood: _whiten keeps observation 0
        # (scaled by sqrt(1-rho^2)), so the transform has log-Jacobian
        # 0.5*log(1-rho^2). Omitting it makes the reported value a hybrid of
        # the exact and conditional likelihoods and biases cross-model AICc
        # (each model fits its own rho) toward under-fitting high-rho models.
        jac = 0.5 * float(np.log(max(1.0 - rho * rho, 1.0e-12)))
    # OPEN FINDING (external review, PR #127), still open on purpose, with the
    # mechanism now measured (2026-09-04). ``estimate_car1_theta`` minimises a
    # CONDITIONAL likelihood that drops the first residual; the lines below
    # report the EXACT stationary one, first residual included and ``sigma``
    # re-estimated from all ``n`` whitened values. So the CAR(1) parameter
    # counted by AICc was not fitted to the likelihood AICc consumes.
    #
    # What the measurement adds to the report is WHERE this bites, and it is
    # not uniform. Comparing ``-2 log L`` at the conditional estimate against
    # its minimum over theta, on exact-OU paths (exact transition density, not
    # Euler) over grids this layer itself builds:
    #
    #     grid span    conditional theta_hat   median deficit   max   > 2
    #     1e5, 1e7     interior                0.002 - 0.032    0.12  0/20
    #     1e2          AT the search floor     2.541            13.2  13/20
    #     1e1          AT the search floor     2.613            36.9  12/20
    #
    # The damaging regime is the one where the conditional search range
    # ``[_THETA_LO_FRAC / span, _THETA_HI_FRAC / min(diff(t))]`` (car1.py:66-67,
    # 215-216) CLAMPS: with span 100 the floor is exactly 1e-5, which is the
    # value the review reported, and 8 of 20 estimates sat on it. Where theta
    # is interior the mismatch is real but an order of magnitude below the
    # conventional AICc-relevant threshold of 2.
    #
    # That opens a THIRD route beside the two the review offered (optimise the
    # exact likelihood, or report a consistently conditional one): repair the
    # search range so the estimate stops clamping. It would leave every
    # interior fit bit for bit unchanged and so would not move the anchors,
    # which is what makes the other two routes expensive. UNVERIFIED as a
    # remedy -- it has not been implemented or shown sufficient; it is recorded
    # because the choice among the three is a modelling decision.
    n = whitened.size
    sigma = float(np.sqrt(max(np.dot(whitened, whitened) / max(n, 1), 1.0e-30)))
    log_lik = gaussian_log_likelihood(whitened, sigma=sigma) + jac
    return GLSFitOutput(
        params=p,
        residuals=residuals_raw,
        rho_ar1=rho,
        sigma=sigma,
        log_likelihood=log_lik,
        success=success,
        theta_car1=theta,
    )
