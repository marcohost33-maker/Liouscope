"""Generalised least squares with AR(1) residual covariance.

The relevant likelihood is

    y_t = f(t; theta) + eps_t,    eps_t = rho eps_{t-1} + nu_t

where ``nu_t`` is iid Gaussian with variance ``sigma^2``. We fit ``theta``
by minimising the AR(1) whitened residual sum of squares, then estimate
``(rho, sigma^2)`` from the residuals of the latest fit. Iterate two
rounds (Cochrane-Orcutt style).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..numerics.norms import scaled_log_sum_squares
from .aicc import gaussian_log_likelihood
from .models import saturation_watch
from .neff import _AR1_SMALL_N, ar1_correlation_corrected


@dataclass(frozen=True, slots=True)
class GLSFitOutput:
    params: np.ndarray
    residuals: np.ndarray
    rho_ar1: float
    sigma: float
    log_likelihood: float
    success: bool
    #: Magnitude guards that fired on the FINAL model evaluation, if any
    #: (``"exponent"`` / ``"magnitude"``). Non-empty implies ``success`` is
    #: False -- see the note at the final evaluation below.
    saturated: tuple[str, ...] = ()
    #: True when the CURVE carried no resolvable variation, so no fit was
    #: attempted at all (issue #123). Distinct from ``saturated``, which
    #: reports a fit that ran and ended on a magnitude plateau; here there was
    #: nothing to fit. Implies ``success`` is False and ``params`` is NaN.
    degenerate: bool = False
    #: True when the residual Gaussian scale has no finite usable MLE for model
    #: selection (issue #135). Distinct from ``degenerate`` above: the curve
    #: may carry variation and the optimiser may have run, but exact-zero RSS
    #: (or an unrepresentable positive MLE scale) cannot support AICc/CI claims.
    likelihood_degenerate: bool = False
    #: True when the profile likelihood IS computable in log space but the
    #: positive MLE scale ``exp(log_sigma)`` is not representable as a float64
    #: (round-1 review of PR #147). ``sigma`` is then NaN while
    #: ``log_likelihood`` stays finite: the fit remains a valid AICc candidate,
    #: and only the scale-dependent evidence -- the parametric bootstrap, hence
    #: the CI -- is withheld. Distinct from ``likelihood_degenerate``, where the
    #: likelihood itself has no finite value.
    scale_unavailable: bool = False


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
    # Fail closed on a curve with NO RESOLVABLE VARIATION (issue #123,
    # round-20 review). Measured on ``t = linspace(0, 5, 64)`` against an
    # identically-zero relative-entropy curve: the fit returned
    # ``success=True`` with the rate parameter equal to its own seed, and the
    # parametric bootstrap around that point produced a BCa interval of width
    # EXACTLY 0.0 -- perfect confidence as the failure mode of an uncertainty
    # pipeline. The optimiser is not at fault: with zero data variation every
    # direction is equally optimal, so "gradient is small" is satisfied at the
    # starting point and the seed comes back wearing the shape of a
    # measurement.
    #
    # The criterion is relative to the curve's OWN scale, never absolute: an
    # absolute floor would reintroduce exactly the rate-unit dependence that
    # #108/#111 removed from the spectral layer. ``ptp(y) <= eps * max|y|``
    # says the variation is at or below the representation resolution of the
    # values it varies between -- for the identically-zero curve, ``0 <= 0``.
    # It is scale-invariant by construction: multiplying ``y`` by any constant
    # multiplies both sides.
    #
    # This is deliberately a statement about the CURVE, not about the grid.
    # The resolution guard of PR #115 asks "was the mode sampled?"; a curve
    # can be flat for reasons no grid can see -- a stationary initial state, a
    # fully decayed one, an observable with no support on the dynamics.
    spread = float(np.ptp(y)) if y.size else 0.0
    y_scale = float(np.max(np.abs(y))) if y.size else 0.0
    if spread <= float(np.finfo(float).eps) * y_scale:
        warnings.warn(
            f"fit_gls_ar1: the curve varies by {spread:.3e} over a scale of "
            f"{y_scale:.3e}, at or below double-precision resolution -- there "
            "is nothing to fit. Returning NaN parameters with success=False "
            "rather than the seed, which is what the optimiser would hand "
            "back unchanged (issue #123).",
            RuntimeWarning,
            stacklevel=2,
        )
        return GLSFitOutput(
            params=np.full(p.shape, float("nan")),
            residuals=np.full(y.shape, float("nan")),
            rho_ar1=0.0,
            sigma=float("nan"),
            log_likelihood=float("nan"),
            success=False,
            degenerate=True,
        )

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
        # S2 audit 2026-06-04: use the small-sample bias-corrected lag-1
        # autocorrelation. The raw plug-in estimator is downward-biased at
        # small n, which makes AR(1)-whitened CIs too narrow. Suppress the
        # per-iteration small-n warning here; we emit it once below so a
        # B-fold bootstrap does not raise B identical warnings.
        rho = ar1_correlation_corrected(residuals_raw, warn_small_n=False)

    # Fail closed on a fit that ENDED inside the model's magnitude guards. The
    # guards keep an out-of-range probe finite so the optimiser can step away
    # from it, but the finite value they return is constant, so its derivatives
    # vanish and ``least_squares`` reports "gradient is small" -- convergence
    # for the wrong reason. Measured (issue #118 finding 9): M0 on
    # ``t in [0, 1e10]`` from ``p0 = [1, -1]`` returned ``success=True`` with p0
    # unchanged and a residual norm of 7.9e100. Probes that merely PASS through
    # the plateau stay untouched; only the reported optimum is judged.
    with saturation_watch() as fired:
        y_hat_final = model(t, p)
    if fired:
        success = False
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
    whitened = _whiten(residuals_raw, rho)
    n = whitened.size
    log_rss = scaled_log_sum_squares(whitened)
    if log_rss == float("-inf") or not math.isfinite(log_rss):
        warnings.warn(
            "fit_gls_ar1: residual Gaussian scale has no finite positive MLE "
            "for model selection; likelihood/AICc/CI evidence is unavailable "
            "(issue #135).",
            RuntimeWarning,
            stacklevel=2,
        )
        return GLSFitOutput(
            params=p,
            residuals=residuals_raw,
            rho_ar1=rho,
            sigma=float("nan"),
            log_likelihood=float("nan"),
            success=False,
            saturated=tuple(sorted(fired)),
            likelihood_degenerate=True,
        )

    log_sigma = 0.5 * (log_rss - math.log(n))
    try:
        sigma = float(math.exp(log_sigma))
    except OverflowError:
        sigma = float("inf")
    scale_unavailable = not math.isfinite(sigma) or sigma <= 0.0
    if scale_unavailable:
        # ROUND-1 REVIEW (PR #147). This branch used to return
        # ``success=False, log_likelihood=nan, likelihood_degenerate=True``,
        # which withheld the whole FIT because a number the likelihood never
        # needs could not be materialised. ``log_rss`` is finite here -- the
        # gate above already refused the case where it is not -- so the profile
        # likelihood is computable directly in log space by
        # ``gaussian_log_likelihood``, which never forms ``sigma``. Only
        # ``exp(log_sigma)`` left the float64 range.
        #
        # Measured on the reviewer's construction: one minimum-subnormal
        # residual (5e-324) in an otherwise-zero series of 64 points gives
        # ``log_rss = -1488.88``, ``log_sigma = -746.52`` and a perfectly
        # finite profile log-likelihood of ``+47686.44`` -- while
        # ``exp(-746.52)`` underflows to ``0.0`` because the smallest
        # subnormal is ``exp(-744.44)``. Marking the fit unsuccessful there
        # makes ``aicc()`` return ``inf`` in ``_fit_one`` and the model drops
        # out of selection, which reintroduces exactly the ABSOLUTE SCALE
        # BOUNDARY into model selection that issue #135 removed: the same
        # curve in different rate units is or is not a candidate.
        #
        # So the fit stays selectable and only the scale-dependent evidence is
        # withheld. ``sigma`` is NaN, never 0.0 or inf: it is consumed by
        # ``_ar1_resample`` as the standard deviation of the innovation, where
        # 0.0 would silently generate a bootstrap of IDENTICAL replicates --
        # a zero-width CI, which is the failure mode of an uncertainty
        # pipeline, not a wide one. ``parametric_bootstrap`` refuses on the
        # dedicated flag rather than on ``success``, so "no interval" and "no
        # estimate" stay distinguishable.
        warnings.warn(
            "fit_gls_ar1: the positive residual MLE scale is not representable "
            f"as float64 (log sigma = {log_sigma:.6g}); the log-space "
            "likelihood and AICc remain available, but bootstrap/CI evidence "
            "is withheld (issue #135, PR #147 review).",
            RuntimeWarning,
            stacklevel=2,
        )

    # Prais-Winsten exact AR(1) likelihood: _whiten keeps observation 0
    # (scaled by sqrt(1-rho^2)), so the transform has log-Jacobian
    # 0.5*log(1-rho^2). The profile likelihood is evaluated directly from
    # log(RSS), not by squaring ``sigma`` or materialising RSS.
    jac = 0.5 * float(np.log(max(1.0 - rho * rho, 1.0e-12)))
    log_lik = gaussian_log_likelihood(whitened) + jac
    return GLSFitOutput(
        params=p,
        residuals=residuals_raw,
        rho_ar1=rho,
        sigma=float("nan") if scale_unavailable else sigma,
        log_likelihood=log_lik,
        success=success,
        saturated=tuple(sorted(fired)),
        scale_unavailable=scale_unavailable,
    )
