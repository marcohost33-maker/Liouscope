"""Small-sample-corrected Akaike information criterion.

For a model with ``k`` estimated parameters and model-selection sample size
``n``,

    AICc = -2 ln L + 2 k + 2 k (k + 1) / (n - k - 1).

LiouScope's production GLS path passes the number of observed time points as
``n_obs``.  Serial dependence is already represented in the AR(1)/CAR(1)
likelihood; substituting an ESS derived from the same dependence into the AICc
denominator would apply a second, heuristic dependence penalty.  The legacy
third positional argument remains accepted as ``n_eff`` for API
compatibility and for callers that deliberately define a different criterion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..numerics.norms import scaled_log_sum_squares


def aicc(
    log_likelihood: float,
    k: int,
    n_eff: float | None = None,
    *,
    n_obs: float | None = None,
) -> float:
    """Return an AICc score for the supplied model-selection sample size.

    ``n_obs`` is the explicit production spelling.  ``n_eff`` is retained
    as a legacy compatibility alias; supplying both is an error because the
    two quantities express different statistical choices.
    """
    if n_obs is not None and n_eff is not None:
        raise ValueError("aicc: pass exactly one of n_obs or the legacy n_eff")
    if n_obs is None:
        if n_eff is None:
            raise TypeError("aicc: missing model-selection sample size")
        n = float(n_eff)
    else:
        n = float(n_obs)
    if not math.isfinite(n) or n <= 0.0:
        return float("inf")
    if n - k - 1 <= 0:
        return float("inf")
    correction = 2.0 * k * (k + 1) / (n - k - 1)
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
        # ROUND-3 REVIEW (PR #147). The round-2 bound was still a decision
        # made in LOG SPACE, and at this magnitude log space cannot make it.
        # Measured with ``sigma = 1`` and a single residual
        # ``sqrt(float_max) * sqrt(2)``: ``log_rss`` comes out at
        # 710.475860073944 while ``log_max + log(2)`` rounds to
        # 710.4758600739439 -- ONE ulp lower -- so the guard returned
        # ``-inf`` although the mathematically relevant half-RSS is
        # 1.7976931348623155e+308, a perfectly finite 0.9999999999999999 of
        # ``float_max``. Nor can the halved value be recovered by subtracting
        # ``log(2)`` and exponentiating: one ulp of a logarithm near 710 is a
        # factor of ~1e-16 in the value, which straddles the overflow edge
        # exactly where this test is made, and ``math.exp`` raises
        # ``OverflowError`` on the same input (measured).
        #
        # So the halved quantity is MATERIALISED from the residuals instead
        # of inferred from their logarithm. Halving each term BEFORE
        # accumulating keeps every partial sum inside float64 whenever the
        # true half-RSS is inside it, and lets the float64 addition itself
        # decide the boundary -- there is no rounded bound left to be one ulp
        # wrong about. An overflow here is now a measurement rather than a
        # prediction: ``inf`` means the half-RSS genuinely does not fit
        # (verified for 4x, 1e3x float_max, and for 64 residuals whose halved
        # sum overflows), and only then is the fit dropped.
        #
        # The lower octave is untouched, so no previously computed likelihood
        # changes by even one ulp -- the same promise round 2 made.
        log_max = math.log(np.finfo(float).max)
        if log_standardised_rss > log_max:
            scaled = residuals / sigma
            with np.errstate(over="ignore", invalid="ignore"):
                half_standardised_rss = float(np.sum((0.5 * scaled) * scaled))
            # Tested for OVERFLOW specifically, not for non-finiteness.
            # The mutation run reported a blanket isfinite() check BLIND, and
            # measuring why exposed a latent state collapse: for an infinite
            # half-RSS the check is provably equivalent (the subtraction
            # below already yields -inf), and its only remaining effect would
            # be to turn a NaN into -inf -- converting "model-selection
            # likelihood unavailable", which this function returns NaN for by
            # contract, into "definitively dropped". Those are different
            # states. NaN is unreachable here (a NaN residual is caught by
            # the isnan(log_rss) gate above, and every term 0.5*s*s is
            # non-negative), so this is guarding an invariant rather than a
            # path -- which is the reason to state it precisely.
            if half_standardised_rss == float("inf"):
                return float("-inf")
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
