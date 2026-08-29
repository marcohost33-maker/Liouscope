"""Continuous-time AR(1) residual model (CAR(1) / Ornstein-Uhlenbeck).

Why this module exists
----------------------
The GLS layer (:mod:`liouscope.fitting.gls`) whitens fit residuals with a
discrete AR(1) model ``r_k - rho r_{k-1}`` using a SINGLE ``rho``. That form is
tied to a constant sampling interval: on a grid whose step varies, the lag-1
correlation between neighbouring samples varies with it, so one ``rho``
whitens part of the series and mis-whitens the rest.

That is a property of the DISCRETE parametrisation, not of the noise process.
The stationary continuous-time analogue -- the Ornstein-Uhlenbeck process,
equivalently CAR(1) -- has

    Corr(eps(t), eps(t + d)) = exp(-theta * d)     for any d >= 0,

so on an ARBITRARY grid ``t_0 < t_1 < ... < t_{n-1}`` the exact one-step
transition is

    eps_k = a_k eps_{k-1} + nu_k,
    a_k   = exp(-theta * dt_k),   Var(nu_k) = s^2 (1 - a_k^2),   dt_k = t_k - t_{k-1}

with ``s^2`` the stationary variance. Whitening with the per-step ``a_k`` and
rescaling by ``sqrt(1 - a_k^2)`` therefore yields residuals that are white AND
homoskedastic on any grid. Uniform spacing is the special case
``a_k = rho = exp(-theta dt)``.

Measured consequence (40 runs, exact OU transition density with theta = 0.7,
on the two-scale grid this layer builds): median ``|lag-1 autocorrelation|`` of
the whitened residuals is 0.3738 with one constant ``rho`` -- taken at the
median step, i.e. that scheme's best case -- against 0.0728 with the per-step
``a_k``. The positive control on a uniform grid gives 0.0728 for BOTH schemes,
which is what shows the contrast comes from the grid and not from the
comparison.

Scope
-----
This module is deliberately narrow: grid classification, theta estimation,
whitening (+ its log-Jacobian), the exact effective sample size, and CAR(1)
resampling for the parametric bootstrap. It changes nothing on a uniform grid:
callers keep the historical AR(1) path there bit-for-bit, and reach for CAR(1)
only when the grid actually varies its step.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

# Relative tolerance on the spread of ``diff(t)`` below which a grid counts as
# uniform. ``linspace`` output varies at the 1e-16 level, and any grid uniform
# to 1e-6 has ``a_k`` varying by the same 1e-6, far below the sampling error of
# the correlation estimate itself -- so the two paths are numerically
# indistinguishable there and the historical one is kept.
GRID_UNIFORM_RTOL: float = 1.0e-6

# Floor on ``1 - a_k^2``. Mirrors the ``max(1 - rho*rho, 1e-12)`` guard in
# ``gls._whiten`` so the two paths clip identically as correlation -> 1.
_VAR_FLOOR: float = 1.0e-12

# Bounds of the theta search, expressed in the grid's own scales: from
# "essentially constant across the whole window" to "already decorrelated
# within the smallest step". Outside this range theta is not identifiable from
# this grid at all, so the bounds are a statement about the data, not a taste
# parameter.
_THETA_LO_FRAC: float = 1.0e-3   # / (t[-1] - t[0])
_THETA_HI_FRAC: float = 1.0e3    # / min(diff(t))


def is_uniform_grid(t: np.ndarray, *, rtol: float = GRID_UNIFORM_RTOL) -> bool:
    """Whether ``t`` is uniformly spaced to within ``rtol``.

    Fails CLOSED to ``True``: a grid that is not strictly increasing, or that
    carries non-finite steps, is not a grid this module can model, and
    returning ``True`` routes such input down the historical AR(1) path
    unchanged rather than into a new code path on the strength of a malformed
    input. Grids of fewer than three points have at most one step and are
    uniform by construction.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 3:
        return True
    d = np.diff(t)
    if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        return True
    return bool(np.allclose(d, d[0], rtol=rtol, atol=0.0))


def car1_rho(t: np.ndarray, theta: float) -> np.ndarray:
    """Per-step correlations ``a_k = exp(-theta * dt_k)``; length ``len(t) - 1``."""
    dt = np.diff(np.asarray(t, dtype=float))
    return np.exp(-abs(float(theta)) * dt)


def whiten_car1(residuals: np.ndarray, t: np.ndarray, theta: float) -> np.ndarray:
    """CAR(1)-whitened residuals, scaled to CONSTANT variance.

    ``w_0 = r_0`` and ``w_k = (r_k - a_k r_{k-1}) / sqrt(1 - a_k^2)``, so every
    ``w_k`` has the stationary variance ``s^2``. The rescaling is not cosmetic:
    without it the innovation variance ``s^2 (1 - a_k^2)`` itself depends on
    ``dt_k``, the whitened series is heteroskedastic, and the GLS weights are
    wrong in exactly the way the constant-``rho`` whitening was.

    Note the normalisation differs from :func:`liouscope.fitting.gls._whiten`,
    which leaves the innovations unscaled and instead scales observation 0.
    Both are valid Prais-Winsten transforms and, on a uniform grid, give the
    SAME log-likelihood once each is paired with its own log-Jacobian (the
    common factor cancels between the sigma estimate and the Jacobian);
    ``tests/test_car1.py`` pins that identity. The constant-variance convention
    is the one that generalises, because there is no single innovation variance
    when ``dt`` varies.
    """
    r = np.asarray(residuals, dtype=float)
    if r.size < 2:
        return r.copy()
    a = car1_rho(t, theta)
    var = np.maximum(1.0 - a * a, _VAR_FLOOR)
    out = np.empty_like(r)
    out[0] = r[0]
    out[1:] = (r[1:] - a * r[:-1]) / np.sqrt(var)
    return out


def whiten_car1_log_jacobian(t: np.ndarray, theta: float) -> float:
    """``log|det d w / d r|`` for :func:`whiten_car1`.

    The transform is lower triangular with diagonal
    ``(1, 1/sqrt(1 - a_1^2), ...)``, so the log-determinant is
    ``-0.5 * sum_k log(1 - a_k^2)``. It must be added to the Gaussian
    log-likelihood of ``w``; omitting it makes the reported value a hybrid of
    the exact and conditional likelihoods and biases the cross-model AICc
    comparison (each model estimates its own theta).
    """
    a = car1_rho(t, theta)
    var = np.maximum(1.0 - a * a, _VAR_FLOOR)
    return float(-0.5 * np.sum(np.log(var)))


def _profile_nll(
    theta: float, dt: np.ndarray, r0: np.ndarray, r1: np.ndarray
) -> float:
    """Negative log-likelihood of the CAR(1) step, profiled over ``s^2``.

    ``-2 log L = (n-1) log s2_hat + sum_k log(1 - a_k^2) + const`` after
    substituting the closed-form maximiser
    ``s2_hat = mean_k[(r_k - a_k r_{k-1})^2 / (1 - a_k^2)]``.
    """
    a = np.exp(-theta * dt)
    var = np.maximum(1.0 - a * a, _VAR_FLOOR)
    innov = r1 - a * r0
    s2 = float(np.mean(innov * innov / var))
    if not np.isfinite(s2) or s2 <= 0.0:
        return float("inf")
    return float(dt.size * np.log(s2) + np.sum(np.log(var)))


def estimate_car1_theta(t: np.ndarray, residuals: np.ndarray) -> float:
    """Conditional-MLE relaxation rate ``theta`` of a CAR(1) residual series.

    Maximises the exact one-step Gaussian likelihood over ``theta`` with the
    stationary variance profiled out (see :func:`_profile_nll`). A coarse
    logarithmic sweep locates the basin before the bounded refinement: the
    search interval spans the whole range of steps present in the grid -- on a
    two-scale grid that is many decades -- and a bare local solver seeded at
    one end can stop at a boundary.

    No small-sample bias correction
    -------------------------------
    The uniform AR(1) path applies the first-order correction
    ``(rho (n-1) + 1) / (n - 3)`` (:func:`liouscope.fitting.neff.
    ar1_correlation_corrected`). It is deliberately NOT carried over here, and
    the reason is measured rather than aesthetic. Transporting it into the
    theta domain requires anchoring it at some single step -- and on a grid
    whose steps span six decades every anchor dominates the estimate. Anchored
    at the mean step of the default two-scale grid, the correction floors the
    implied correlation at ``1/(n-3)`` and the function returned the SAME
    constant ``1.7158e-05`` for every input tested, including OU paths
    generated with true theta of 1e-4, 1e-2, 1 and 10: the estimator had no
    discrimination left at all. Returning the MLE keeps it (measured relative
    error 0.05 / 0.05 / 0.25 at true theta 1e-7 / 1e-6 / 1e-5, and the estimate
    tracks the truth up to the point where ``dt_min`` makes theta
    unidentifiable).

    The residual downward bias of the correlation at small ``n`` is therefore
    NOT corrected on this path -- the same optimism the uniform path warns
    about remains, and the existing small-``n`` ``RuntimeWarning`` from
    :func:`liouscope.fitting.gls.fit_gls_ar1` is what discloses it.

    Returns NaN when the grid or the residuals cannot support an estimate
    (fewer than three points, a non-increasing grid, non-finite input, or a
    degenerate zero-variance series) -- fail-closed, so the caller keeps the
    AR(1) path instead of whitening with a fabricated rate.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if t.size < 3 or r.size != t.size:
        return float("nan")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(r))):
        return float("nan")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        return float("nan")
    centred = r - float(np.mean(r))
    if float(np.dot(centred, centred)) <= 0.0:
        return float("nan")

    r0, r1 = r[:-1], r[1:]
    # Search range in theta: from "essentially constant across the whole
    # window" to "already decorrelated within the smallest step". Anything
    # outside is unidentifiable from this grid by construction.
    span = float(t[-1] - t[0])
    dt_min = float(np.min(dt))
    if span <= 0.0 or dt_min <= 0.0:
        return float("nan")
    lo = float(np.log(_THETA_LO_FRAC / span))
    hi = float(np.log(_THETA_HI_FRAC / dt_min))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return float("nan")

    sweep = np.linspace(lo, hi, 241)
    vals = np.array(
        [_profile_nll(float(np.exp(u)), dt, r0, r1) for u in sweep], dtype=float
    )
    if not np.any(np.isfinite(vals)):
        return float("nan")
    j = int(np.argmin(np.where(np.isfinite(vals), vals, np.inf)))
    left = float(sweep[max(j - 1, 0)])
    right = float(sweep[min(j + 1, sweep.size - 1)])
    if right <= left:
        theta = float(np.exp(sweep[j]))
    else:
        res = minimize_scalar(
            lambda u: _profile_nll(float(np.exp(u)), dt, r0, r1),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1.0e-10},
        )
        theta = (
            float(np.exp(float(res.x))) if res.success else float(np.exp(sweep[j]))
        )

    if not np.isfinite(theta) or theta <= 0.0:
        return float("nan")
    return theta


def neff_car1(t: np.ndarray, theta: float) -> float:
    """Exact effective sample size of a CAR(1) series observed at times ``t``.

    For the mean of a stationary series with correlation matrix ``C`` the
    variance-inflation factor is ``1^T C 1 / n``, hence

        N_eff = n^2 / sum_{j,k} exp(-theta |t_j - t_k|).

    On a uniform grid this is algebraically identical to the familiar
    ``n / (1 + 2 sum_k (1 - k/n) rho^k)`` -- the same double sum, not an
    approximation of it -- so it is a strict generalisation of what the uniform
    path estimates, not a competing quantity.

    This replaces :func:`liouscope.fitting.neff.estimate_neff_geyer` on a
    non-uniform grid because the Geyer initial-positive-sequence estimator sums
    autocorrelations indexed by LAG, and on a varying grid a fixed lag is not a
    fixed time separation: lag 1 mixes the fine and the coarse region, so the
    sequence it sums is not the autocorrelation function of anything. Being
    model-based is the price; the GLS layer already assumes this exact residual
    model, so it is not a new assumption.

    Measured, not assumed
    ---------------------
    A replacement has to earn the swap, so both estimators were run against the
    closed-form ESS on a UNIFORM grid, where the two models coincide and the
    truth is known (n = 80, 200 replicates per point, exact OU paths):

        rho      exact    car1 (ratio)    Geyer (ratio)
        0.00    80.000    77.35 (0.97)    73.75 (0.92)
        0.30    43.435    43.12 (0.99)    44.76 (1.03)
        0.60    20.480    21.47 (1.05)    23.44 (1.14)
        0.85     7.024     7.39 (1.05)    10.67 (1.52)
        0.95     2.698     3.22 (1.19)     6.23 (2.31)
        0.99     1.285     1.45 (1.13)     5.05 (3.93)

    So the CAR(1) route is the closer of the two exactly where the correction
    matters -- Geyer is 2-4x optimistic at high correlation, which is the
    regime an ODE trajectory lives in. The uniform path nevertheless KEEPS
    Geyer: switching it would move ``N_eff`` for every existing fit and with it
    every AICc and every anchor, which is a separate decision with its own
    evidence burden, not a side effect of a grid fix.

    Returns NaN for a non-finite theta or a malformed grid.
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    if n == 0 or not np.all(np.isfinite(t)):
        return float("nan")
    if not np.isfinite(theta) or theta < 0.0:
        return float("nan")
    if n < 2:
        return float(n)
    sep = np.abs(t[:, None] - t[None, :])
    total = float(np.sum(np.exp(-abs(float(theta)) * sep)))
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return float(np.clip(n * n / total, 1.0, float(n)))


def car1_resample(
    rng: np.random.Generator,
    t: np.ndarray,
    theta: float,
    sigma_stationary: float,
) -> np.ndarray:
    """Draw a stationary CAR(1) path on ``t`` with stationary sd ``sigma``.

    Uses the EXACT transition density ``eps_k = a_k eps_{k-1} +
    sqrt(s^2 (1 - a_k^2)) z_k``, not an Euler step: on a two-scale grid the
    steps span many decades, where an Euler discretisation is meaningless on
    the coarse half.
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    eps = np.empty(n)
    s = float(sigma_stationary)
    if n == 0:
        return eps
    eps[0] = rng.normal(0.0, s)
    if n == 1:
        return eps
    a = car1_rho(t, theta)
    sd = s * np.sqrt(np.maximum(1.0 - a * a, 0.0))
    z = rng.normal(size=a.size)
    for k in range(a.size):
        eps[k + 1] = a[k] * eps[k] + sd[k] * z[k]
    return eps


__all__ = [
    "GRID_UNIFORM_RTOL",
    "car1_resample",
    "car1_rho",
    "estimate_car1_theta",
    "is_uniform_grid",
    "neff_car1",
    "whiten_car1",
    "whiten_car1_log_jacobian",
]
