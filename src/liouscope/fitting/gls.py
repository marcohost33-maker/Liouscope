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
from scipy.optimize import OptimizeResult, least_squares

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


class _ScaledResidualOverflowError(ArithmeticError):
    """A residual rescaled for the optimiser left float64 while the raw one is finite.

    Private control flow of :func:`fit_gls_ar1` (PR #134). Deliberately NOT a
    ``ValueError``/``RuntimeError``: those mean "the fit failed" there, while
    this means only "the #124 rescaling is not representable for this curve".
    """


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
    # Fail closed on MALFORMED observations, BEFORE anything else inspects them
    # (round-24 review, PR #121). The degeneracy test below reads ``y`` alone;
    # it never touches ``t``, so a mismatched pair reaches it intact and a
    # constant or single-point ``y`` short-circuits out with
    # ``degenerate=True`` -- a plausible, structured answer that says "this
    # curve has no resolvable variation" about a curve that was never supplied.
    # Measured: a 64-point ``t`` against a one-point ``y`` was reported as a
    # flat curve on that grid. Nothing downstream can recover the mismatch from
    # that verdict, because the verdict does not mention the grid.
    #
    # Placed ahead of the finiteness gate deliberately: the index list that
    # gate reports is meaningless for a pair that is not aligned in the first
    # place, and shape is the more fundamental question. The rule is the same
    # one -- data the caller hands in must be measurements.
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "fit_gls_ar1: t and y must be one-dimensional observation arrays; "
            f"got t.ndim={t.ndim} (shape {t.shape}) and y.ndim={y.ndim} "
            f"(shape {y.shape})"
        )
    if t.size != y.size:
        raise ValueError(
            "fit_gls_ar1: t and y must have the same length; got "
            f"len(t)={t.size} and len(y)={y.size}"
        )
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

    # Issue #124: the mathematical least-squares optimum is unchanged when an
    # observable is multiplied by a positive constant, but SciPy's numerical
    # termination is not. In particular, TRF declares success when its scaled
    # gradient falls below ``gtol``; a curve such as ``1e-40*exp(-1.3*t)`` can
    # therefore return the INITIAL RATE as a "converged" estimate after one
    # evaluation. Divide only the residuals presented to the optimiser by the
    # curve's own finite, non-zero scale. This multiplies the objective by one
    # positive constant, so the minimiser is identical, while the numerical
    # problem becomes amplitude-scale invariant. Raw residuals, AR(1) rho,
    # sigma and likelihood below remain in the caller's original data units.
    #
    # The rescaled problem is not always REPRESENTABLE (PR #134 x PR #147).
    # SciPy's finite-difference probes step by ``~1.5e-8 * max(1, |x|)`` in
    # ABSOLUTE parameter units, so the model moves by an amount unrelated to
    # the curve's scale. Measured on the #147 fixture (``max|y| = 5e-324``):
    # the scaled residual at ``p0`` is finite (``max = 1.0``), the first
    # Jacobian probe gives ``1.5e-8 / 5e-324 = inf``, ``least_squares`` raises
    # and the fit was reported unsuccessful -- a curve withheld from model
    # selection because of its absolute amplitude, the boundary #135 removed.
    # The overflow is a property of the rescaling, not of the fit: when a
    # scaled residual leaves float64 while the raw one is finite, the
    # iteration is repeated on the raw residuals, exactly the pre-#124
    # problem, and the loss of the #124 invariance is announced rather than
    # silent. A raw residual that is itself non-finite still fails closed.
    fit_scale = y_scale

    def _residual_fn(
        rho_local: float, scale: float
    ) -> Callable[[np.ndarray], np.ndarray]:
        def residual(params: np.ndarray) -> np.ndarray:
            y_hat = model(t, params)
            r_raw = y - y_hat
            # errstate: the overflow is DETECTED below, so it must not also
            # surface as a numpy RuntimeWarning (an error under this repo's
            # ``filterwarnings = error``) before the detection can act.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                scaled = _whiten(r_raw / scale, rho_local)
            if not np.all(np.isfinite(scaled)) and np.all(np.isfinite(r_raw)):
                raise _ScaledResidualOverflowError
            return scaled

        return residual

    # The residual is not the only place the rescaled problem can leave
    # float64: with a FREE amplitude parameter the Jacobian and the cost carry
    # the same ``1/scale`` factor. Measured through ``_fit_with_model("M0")``
    # on ``s*exp(-1.3 t)``: at s = 1e-150 and 1e-310 the rescaled optimiser
    # overflowed inside SciPy (``dot``/``square``) and the fit was reported
    # unsuccessful (aicc = inf), where the raw problem fits rate 1.3. So every
    # floating-point exception raised while the RESCALED problem is solved,
    # and any non-finite cost/residual/Jacobian it returns, counts as "not
    # representable" and sends the iteration to the raw residuals. A finite
    # rescaled result that merely did not converge is NOT retried: that is a
    # genuine failure, and retrying it raw would re-admit the #124 seed.
    def _solve_rescaled(
        rho_local: float, x0: np.ndarray, **kw: object
    ) -> OptimizeResult:
        events: list[str] = []

        def _record(kind: str, _flag: int) -> None:
            events.append(kind)

        with np.errstate(
            over="call", divide="call", invalid="call", under="ignore", call=_record
        ):
            try:
                res = least_squares(_residual_fn(rho_local, fit_scale), x0, **kw)
            except (ValueError, RuntimeError):
                if events:
                    raise _ScaledResidualOverflowError from None
                raise
        if events or not (
            np.isfinite(res.cost)
            and np.all(np.isfinite(res.fun))
            and np.all(np.isfinite(res.jac))
        ):
            raise _ScaledResidualOverflowError
        return res

    rho = 0.0
    success = True
    for _ in range(n_iters):
        ls_kwargs: dict[str, object] = {"max_nfev": max_nfev}
        if bounds is not None:
            ls_kwargs["bounds"] = bounds
        try:
            try:
                if fit_scale == 1.0:
                    result = least_squares(
                        _residual_fn(rho, fit_scale), p, **ls_kwargs
                    )
                else:
                    result = _solve_rescaled(rho, p, **ls_kwargs)
            except _ScaledResidualOverflowError:
                warnings.warn(
                    "fit_gls_ar1: residuals rescaled by the curve's own scale "
                    f"({fit_scale:.3e}) are not representable as float64 at the "
                    "optimiser's probe points; fitting the unscaled residuals "
                    "instead, so the amplitude-scale invariance of issue #124 "
                    "does not hold for this curve (PR #134).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fit_scale = 1.0
                result = least_squares(_residual_fn(rho, fit_scale), p, **ls_kwargs)
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
