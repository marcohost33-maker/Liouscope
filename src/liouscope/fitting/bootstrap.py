"""Parametric bootstrap with BCa confidence intervals.

Anchor G: standard iid bootstrap on ODE trajectories violates independence
(neighbouring time samples are heavily correlated). We instead:

1. Fit the model once (GLS+AR(1)).
2. Resimulate ``nu_t ~ N(0, sigma^2)`` and propagate through
   ``eps_t = rho eps_{t-1} + nu_t``.
3. Re-fit the model on the resimulated trajectory.
4. Aggregate fitted parameters and report BCa confidence intervals
   (Efron 1987).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import ndtr, ndtri

from .gls import fit_gls_ar1


def _ar1_resample(
    rng: np.random.Generator,
    n: int,
    rho: float,
    sigma: float,
) -> np.ndarray:
    eps = np.empty(n)
    nu = rng.normal(0.0, sigma, size=n)
    eps[0] = nu[0] / np.sqrt(max(1.0 - rho * rho, 1.0e-12))
    for i in range(1, n):
        eps[i] = rho * eps[i - 1] + nu[i]
    return eps


def parametric_bootstrap(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    *,
    B: int = 1000,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(theta_boot, theta_hat)``.

    ``theta_boot`` has shape ``(B, len(p0))``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    base = fit_gls_ar1(model, t, y, p0, bounds=bounds)
    if not base.success:
        # Round-17 review (PR #121). Every replicate is simulated AROUND
        # ``theta_hat``; if the base fit ended on the model's magnitude
        # plateau, that centre is not an estimate and the whole resample is
        # a distribution around a non-result. Raising routes this into the
        # caller's existing handler, which reports the CI as NaN -- "fit
        # uncertainty UNKNOWN" -- rather than as a narrow interval around a
        # failure.
        raise RuntimeError(
            "parametric_bootstrap: the base fit did not converge"
            + (f" (saturated: {', '.join(base.saturated)})" if base.saturated else "")
            + (" (the curve carries no resolvable variation, issue #123)"
               if base.degenerate else "")
            + "; a bootstrap around a non-estimate has no meaning"
        )
    theta_hat = base.params
    samples = np.empty((B, theta_hat.size))
    failed = 0
    for b in range(B):
        eps_b = _ar1_resample(rng, t.size, base.rho_ar1, base.sigma)
        y_b = model(t, theta_hat) + eps_b
        fit_b = fit_gls_ar1(model, t, y_b, theta_hat, bounds=bounds)
        failed += not fit_b.success
        samples[b] = fit_b.params
    if failed:
        # Round-20 review (PR #121). The previous round RETAINED failed
        # replicates on the argument that keeping them can only widen the
        # interval, hence err conservatively. That argument is wrong, and
        # measurably so: a failed ``least_squares`` returns its unchanged
        # STARTING value, which here is ``theta_hat`` itself, so every failure
        # deposits mass exactly at the centre of the distribution. Measured on
        # 400 replicates with 40 % failures, the BCa width fell to 0.93x and
        # 0.87x of the interval computed without them -- NARROWER, in the
        # dangerous direction, and narrower by more the more often the fit
        # failed.
        #
        # Dropping them is not the alternative: that biases the endpoints in a
        # direction nobody has characterised (a fit fails on the replicates
        # that are hardest, which is not a random subset). Both routes need a
        # failure-handling rule validated against something; there is none.
        # Until there is, the honest output is no interval at all. The caller
        # in ``compute_relaxation_layer`` already routes RuntimeError to
        # ``bca_ci_beta = (nan, nan)`` -- "fit uncertainty UNKNOWN" -- which is
        # the statement that survives review.
        raise RuntimeError(
            f"parametric_bootstrap: {failed} of {B} replicates did not "
            "converge. A failed fit returns its unchanged starting value, so "
            "retaining them narrows the BCa interval instead of widening it, "
            "and dropping them biases the endpoints; neither has a validated "
            "rule. No confidence interval is reported from this resample"
        )
    return samples, theta_hat


def _jackknife(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    theta_hat: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Leave-one-out jackknife on the time grid.

    Raises
    ------
    RuntimeError
        If any leave-one-out fit fails to converge. Round-22 review (PR #121):
        the round-20 guard on the BOOTSTRAP branch does not protect the
        interval when every replicate converges but a jackknife fit does not.
        The acceleration ``a`` in :func:`bca_ci` is a third moment of exactly
        these estimates, and a failed ``least_squares`` returns its unchanged
        STARTING value -- here ``theta_hat`` itself. One such value pulls the
        jackknife distribution towards its own centre, which shifts ``a``, and
        the BCa quantiles with it: the endpoints move without any data having
        said so, and the interval that comes back is finite, narrow and
        unwarranted.

        The same two routes as in ``parametric_bootstrap`` are available and
        the same objection applies to both: retaining a non-fit deposits mass
        at the centre, dropping it biases the endpoints in a direction nobody
        has characterised (a leave-one-out fit fails on the points that matter
        most). Neither has a validated rule, so no interval is reported. The
        caller in ``compute_relaxation_layer`` already routes RuntimeError to
        ``bca_ci_beta = (nan, nan)`` -- "fit uncertainty UNKNOWN".
    """
    n = t.size
    out = np.empty((n, theta_hat.size))
    failed = 0
    for i in range(n):
        t_i = np.delete(t, i)
        y_i = np.delete(y, i)
        fit_i = fit_gls_ar1(model, t_i, y_i, theta_hat, bounds=bounds)
        failed += not fit_i.success
        out[i] = fit_i.params
    if failed:
        raise RuntimeError(
            f"_jackknife: {failed} of {n} leave-one-out fits did not "
            "converge. A failed fit returns its unchanged starting value, so "
            "the BCa acceleration would be computed from estimates that are "
            "not estimates; no confidence interval is reported from this "
            "jackknife"
        )
    return out


def bca_ci(
    samples: np.ndarray,
    theta_hat: np.ndarray,
    *,
    alpha: float = 0.05,
    jackknife_estimates: np.ndarray | None = None,
) -> np.ndarray:
    """Bias-corrected-and-accelerated bootstrap CI.

    Returns ``(p, 2)`` array of ``(lo, hi)`` per parameter at level
    ``1 - alpha``.
    """
    samples = np.asarray(samples, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)
    B, p = samples.shape
    cis = np.empty((p, 2))
    z_alpha_lo = ndtri(alpha / 2.0)
    z_alpha_hi = ndtri(1.0 - alpha / 2.0)
    for j in range(p):
        boot_j = np.sort(samples[:, j])
        # Bias-correction z0 with half-correction for ties (Efron 1987, S3
        # audit 2026-06-04). Using a strict ``<`` only would send the
        # proportion to 0 (hence z0 -> -inf, clamped) whenever many bootstrap
        # replicates land exactly on theta_hat (common for clipped / bounded
        # parameters or discrete-valued statistics). Counting ties at half
        # weight is the standard continuity correction.
        prop_below = float(
            np.mean(boot_j < theta_hat[j]) + 0.5 * np.mean(boot_j == theta_hat[j])
        )
        prop_below = float(np.clip(prop_below, 1.0e-9, 1.0 - 1.0e-9))
        z0 = ndtri(prop_below)
        # Acceleration: from jackknife if provided, else default to 0.
        if jackknife_estimates is None:
            a = 0.0
        else:
            jk = jackknife_estimates[:, j]
            jk_mean = float(jk.mean())
            num = float(np.sum((jk_mean - jk) ** 3))
            den = 6.0 * float(np.sum((jk_mean - jk) ** 2)) ** 1.5
            a = float(num / den) if den != 0.0 else 0.0
        # Adjusted percentiles
        def adjusted(z_alpha: float, z0_: float = z0, a_: float = a) -> float:
            shifted = z0_ + (z0_ + z_alpha) / (1.0 - a_ * (z0_ + z_alpha))
            return float(ndtr(shifted))

        q_lo = adjusted(z_alpha_lo)
        q_hi = adjusted(z_alpha_hi)
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, 0.0, 1.0))
        # Linear quantile interpolation instead of nearest-rank (S4 audit
        # 2026-06-04). Nearest-rank ``boot_j[round(q*(B-1))]`` is granular for
        # small B (the CI endpoints can only take the B sampled values, so
        # they jump in discrete steps). ``np.quantile`` with the default
        # linear method interpolates between order statistics -- the standard
        # continuous-percentile estimator.
        lo = float(np.quantile(boot_j, q_lo, method="linear"))
        hi = float(np.quantile(boot_j, q_hi, method="linear"))
        cis[j] = (lo, hi)
    return cis


__all__ = ["_jackknife", "bca_ci", "parametric_bootstrap"]
