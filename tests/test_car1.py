"""Contract tests for the CAR(1) residual model (:mod:`liouscope.fitting.car1`).

Context (PR #115). The relaxation layer used to DISCLOSE that it cannot resolve
widely separated timescales, on the argument that a non-uniform grid is
impossible because the GLS layer "whitens with a single AR(1) coefficient,
which presumes a constant sample interval". These tests pin the counter-claim
that replaced it: the constant-interval requirement belongs to the DISCRETE
parametrisation, not to the noise process, and the continuous-time form
``a_k = exp(-theta dt_k)`` is valid on any grid.

Every claim below is checked against an INDEPENDENT oracle rather than against
the implementation's own output:

* the AR(1) <-> CAR(1) likelihood identity on a uniform grid (two different
  Prais-Winsten normalisations must agree once each carries its own Jacobian);
* the closed-form effective sample size ``n^2 / (n + 2 sum_k (n-k) rho^k)``;
* exact OU paths with a KNOWN theta, generated from the exact transition
  density, with the estimate required to MOVE with the truth rather than
  merely land near it.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from liouscope.diagnostics.relaxation import _fit_with_model
from liouscope.fitting.aicc import aicc, gaussian_log_likelihood
from liouscope.fitting.bootstrap import parametric_bootstrap
from liouscope.fitting.car1 import (
    car1_resample,
    car1_rho,
    estimate_car1_theta,
    is_uniform_grid,
    neff_car1,
    whiten_car1,
    whiten_car1_log_jacobian,
)
from liouscope.fitting.gls import _whiten, fit_gls_ar1
from liouscope.fitting.models import M2
from liouscope.fitting.neff import estimate_neff_geyer


def _ess_closed_form(n: int, rho: float) -> float:
    """Exact ESS of the mean of a stationary AR(1) series.

    ``n^2 / (1^T C 1)`` with ``C_jk = rho^|j-k|``, written out as the standard
    single sum. Independent of the implementation under test.
    """
    k = np.arange(1, n)
    return float(n * n / (n + 2.0 * np.sum((n - k) * rho**k)))


def _lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    den = float(x @ x)
    return float(x[1:] @ x[:-1] / den) if den > 0.0 else 0.0


def _two_scale_grid(n: int = 80, t_split: float = 10.0, t_max: float = 2.0e7):
    """The shape the relaxation layer builds: fine head, coarse tail."""
    early = np.linspace(0.0, t_split, n // 2, endpoint=False)
    late = np.linspace(t_split, t_max, n - n // 2)
    return np.concatenate([early, late])


# ---------------------------------------------------------------------------
# Grid classification: the switch must be conservative
# ---------------------------------------------------------------------------


def test_uniform_grids_are_recognised():
    assert is_uniform_grid(np.linspace(0.0, 10.0, 80))
    assert is_uniform_grid(np.linspace(0.0, 1.0e7, 5))
    assert is_uniform_grid(np.array([0.0, 1.0]))  # one step is uniform
    assert is_uniform_grid(np.array([3.0]))


def test_two_scale_and_log_grids_are_not_uniform():
    assert not is_uniform_grid(_two_scale_grid())
    assert not is_uniform_grid(np.geomspace(1.0, 1.0e6, 40))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.0, 1.0, 1.0, 2.0]),          # repeated point: step 0
        np.array([0.0, 2.0, 1.0, 3.0]),          # not increasing
        np.array([0.0, 1.0, np.nan, 3.0]),       # non-finite
        np.array([0.0, 1.0, np.inf]),
    ],
)
def test_malformed_grids_fail_closed_to_the_historical_path(bad):
    """A grid this module cannot model must route to AR(1), not into CAR(1).

    Returning ``True`` here is not a claim that the grid IS uniform; it is the
    fail-closed answer to "may I take the new code path?".
    """
    assert is_uniform_grid(bad)


# ---------------------------------------------------------------------------
# The likelihood identity: on a uniform grid the two forms must agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.7, 0.95])
@pytest.mark.parametrize("dt", [1.0, 0.037, 250.0])
def test_car1_reproduces_the_ar1_log_likelihood_on_a_uniform_grid(rho, dt):
    """Two normalisations, one likelihood -- this is the correctness anchor.

    ``gls._whiten`` scales observation 0 and leaves the innovations; CAR(1)
    keeps observation 0 and scales the innovations to constant variance. The
    two differ by a constant factor that must cancel between the sigma estimate
    and the log-Jacobian. If it does not, the CAR(1) path would silently shift
    every cross-model AICc on a non-uniform grid, and no test on the grid alone
    could see it.
    """
    rng = np.random.default_rng(4242)
    n = 64
    t = np.arange(n) * dt
    r = rng.normal(size=n)

    w_ar = _whiten(r, rho)
    sig_ar = float(np.sqrt(max(np.dot(w_ar, w_ar) / n, 1.0e-30)))
    ll_ar = gaussian_log_likelihood(w_ar, sigma=sig_ar) + 0.5 * float(
        np.log(max(1.0 - rho * rho, 1.0e-12))
    )

    theta = -np.log(rho) / dt if rho > 0.0 else 1.0e30
    w_car = whiten_car1(r, t, theta)
    sig_car = float(np.sqrt(max(np.dot(w_car, w_car) / n, 1.0e-30)))
    ll_car = gaussian_log_likelihood(w_car, sigma=sig_car) + whiten_car1_log_jacobian(
        t, theta
    )

    assert ll_car == pytest.approx(ll_ar, rel=1.0e-10, abs=1.0e-9)


def test_car1_rho_reduces_to_a_single_constant_on_a_uniform_grid():
    t = np.linspace(0.0, 7.0, 50)
    a = car1_rho(t, 0.4)
    np.testing.assert_allclose(a, a[0], rtol=1.0e-12)


# ---------------------------------------------------------------------------
# Effective sample size against the closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.1, 0.3, 0.6, 0.85, 0.99])
def test_neff_car1_equals_the_closed_form_ess_on_a_uniform_grid(rho):
    """Not "close to" -- algebraically the same double sum, so machine exact."""
    n = 80
    t = np.arange(n, dtype=float)
    theta = -np.log(rho)
    assert neff_car1(t, theta) == pytest.approx(_ess_closed_form(n, rho), rel=1.0e-12)


def test_neff_car1_tracks_the_truth_better_than_geyer_at_high_correlation():
    """The swap has to be earned, so it is measured against the closed form.

    Geyer's IPS estimator is the historical route and stays in place on uniform
    grids; on a NON-uniform grid it sums autocorrelations indexed by lag, where
    a fixed lag is not a fixed time separation. This test checks the swap is
    not merely different but closer to the truth in the regime that matters --
    strong autocorrelation, which is where an ODE trajectory lives.
    """
    n, rho = 80, 0.95
    t = np.arange(n, dtype=float)
    theta = -np.log(rho)
    exact = _ess_closed_form(n, rho)
    rng = np.random.default_rng(20260829)
    car, geyer = [], []
    for _ in range(60):
        eps = car1_resample(rng, t, theta, 1.0)
        car.append(neff_car1(t, estimate_car1_theta(t, eps)))
        geyer.append(estimate_neff_geyer(eps))
    err_car = abs(float(np.median(car)) - exact) / exact
    err_geyer = abs(float(np.median(geyer)) - exact) / exact
    assert err_car < err_geyer, (
        f"CAR(1) N_eff not closer to the closed form: "
        f"car={np.median(car):.3f} geyer={np.median(geyer):.3f} exact={exact:.3f}"
    )
    assert err_car < 0.35


def test_neff_car1_is_fail_closed_on_unusable_input():
    t = np.linspace(0.0, 5.0, 10)
    assert np.isnan(neff_car1(t, float("nan")))
    assert np.isnan(neff_car1(t, -1.0))
    assert np.isnan(neff_car1(np.array([0.0, np.nan, 2.0]), 1.0))


# ---------------------------------------------------------------------------
# theta estimation: it must MOVE with the truth, not return a constant
# ---------------------------------------------------------------------------


THETA_TRUTHS = [1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-2, 1.0]


def _estimate_theta_series(t: np.ndarray, seed: int = 909) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for th in THETA_TRUTHS:
        est = [estimate_car1_theta(t, car1_resample(rng, t, th, 1.0)) for _ in range(5)]
        out.append(float(np.nanmedian(est)))
    return out


def test_theta_estimate_moves_with_the_truth_across_six_decades():
    """Blindness looks exactly like a perfect fit, so movement is the assertion.

    A regression here is not hypothetical: an earlier version of this estimator
    round-tripped theta through the correlation at the MEAN step and clipped it,
    and on the two-scale grid that clip returned the SAME constant 1.7158e-05
    for every input -- including OU paths built with true theta 1e-4, 1e-2, 1
    and 10. Every one of those looked like a plausible number. The only thing
    that distinguishes an estimate from a constant is that it moves, so that is
    what is asserted first, before any accuracy claim.
    """
    t = _two_scale_grid()
    got = _estimate_theta_series(t)
    # Strictly increasing in the truth: a constant, a clip or a saturating
    # ceiling all fail this, whatever value they saturate at.
    assert all(b > a for a, b in itertools.pairwise(got)), got
    # And it spans the same order of magnitude range as the truth.
    assert got[-1] / got[0] > 1.0e5, got


@pytest.mark.parametrize("theta_true", [1.0e-6, 1.0e-4, 1.0e-2, 1.0])
def test_theta_estimate_is_accurate_on_the_two_scale_grid(theta_true):
    t = _two_scale_grid()
    rng = np.random.default_rng(31337)
    est = [estimate_car1_theta(t, car1_resample(rng, t, theta_true, 1.0)) for _ in range(7)]
    med = float(np.nanmedian(est))
    assert abs(med - theta_true) / theta_true < 0.5, (med, theta_true)


@pytest.mark.parametrize(
    "t, r",
    [
        (np.array([0.0, 1.0]), np.array([1.0, 2.0])),                 # < 3 points
        (np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])),       # zero step
        (np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])),       # decreasing
        (np.linspace(0.0, 1.0, 5), np.full(5, 2.0)),                  # zero variance
        (np.linspace(0.0, 1.0, 5), np.array([1.0, np.nan, 1.0, 2.0, 3.0])),
    ],
)
def test_theta_estimate_fails_closed(t, r):
    """NaN, not a plausible number: the caller then keeps the AR(1) path."""
    assert np.isnan(estimate_car1_theta(t, r))


# ---------------------------------------------------------------------------
# The claim the disclosure denied: whitening IS valid on a varying grid
# ---------------------------------------------------------------------------


def test_per_step_whitening_beats_a_single_rho_on_a_non_uniform_grid():
    """The measurement that overturned "this is a disclosure, not a repair".

    Both schemes are applied to the SAME OU paths on the SAME non-uniform grid,
    so the only difference is the whitening. The constant-rho scheme is given
    its best case -- rho taken at the median step, not an arbitrary one.
    """
    t = _two_scale_grid()
    theta = 0.7
    rng = np.random.default_rng(20260829)
    rho_const = float(np.exp(-theta * float(np.median(np.diff(t)))))
    const, per_step = [], []
    for _ in range(40):
        eps = car1_resample(rng, t, theta, 0.01)
        const.append(abs(_lag1(eps[1:] - rho_const * eps[:-1])))
        per_step.append(abs(_lag1(whiten_car1(eps, t, theta)[1:])))
    med_const = float(np.median(const))
    med_step = float(np.median(per_step))
    assert med_step < 0.5 * med_const, (med_step, med_const)
    assert med_step < 0.2, med_step


def test_uniform_grid_positive_control_both_schemes_whiten():
    """If the fine-grid case also failed, the test above would measure noise.

    On a uniform grid CAR(1) IS AR(1), so neither scheme may look broken --
    otherwise the contrast above would be an artefact of the comparison rather
    than of the grid.
    """
    n = 80
    dt = 1.0
    t = np.arange(n) * dt
    theta = 0.7
    rng = np.random.default_rng(20260829)
    rho_const = float(np.exp(-theta * dt))
    const, per_step = [], []
    for _ in range(40):
        eps = car1_resample(rng, t, theta, 0.01)
        const.append(abs(_lag1(eps[1:] - rho_const * eps[:-1])))
        per_step.append(abs(_lag1(whiten_car1(eps, t, theta)[1:])))
    assert float(np.median(const)) < 0.2
    assert float(np.median(per_step)) < 0.2


# ---------------------------------------------------------------------------
# Resampling: the bootstrap's surrogate noise must carry the right correlation
# ---------------------------------------------------------------------------


def test_car1_resample_reproduces_the_target_correlation_on_both_scales():
    """Checked at TWO separations, because one would not detect a fixed rho.

    A resampler that ignored ``dt`` and used a single coefficient would match
    at whichever separation that coefficient was tuned to and be wrong at the
    other; asserting both is what makes the test discriminating.
    """
    t = _two_scale_grid()
    theta = 0.05
    dt = np.diff(t)
    rng = np.random.default_rng(5150)
    paths = np.array([car1_resample(rng, t, theta, 1.0) for _ in range(4000)])
    # Fine head: correlation across one small step.
    emp_fine = float(np.mean(paths[:, 1] * paths[:, 0]) / np.var(paths[:, 0]))
    assert emp_fine == pytest.approx(float(np.exp(-theta * dt[0])), abs=0.03)
    # Coarse tail: the same theta over a step 6 decades larger must decorrelate.
    emp_coarse = float(
        np.mean(paths[:, -1] * paths[:, -2]) / np.var(paths[:, -2])
    )
    assert abs(emp_coarse) < 0.05
    # Stationarity: the variance must not drift between the two regimes.
    assert float(np.std(paths[:, 5])) == pytest.approx(1.0, rel=0.06)
    assert float(np.std(paths[:, -1])) == pytest.approx(1.0, rel=0.06)


# ---------------------------------------------------------------------------
# The bootstrap must resample under the model the fit actually used
# ---------------------------------------------------------------------------


def test_bootstrap_spread_matches_an_independent_monte_carlo_on_a_two_scale_grid():
    """Calibration against an oracle the bootstrap cannot see.

    The reference is NOT another bootstrap but the empirical sampling
    distribution: 25 independent datasets from the same truth, each refitted.
    A parametric bootstrap that resampled with the AR(1) branch would draw its
    surrogate noise with the correlation at the MEAN step -- which on this grid
    is exactly zero (``rho_ar1 = 0``, measured) -- so the fine head, where the
    fast rate is actually determined, would get white noise instead of
    correlated noise and the interval would come out too narrow.

    Measured on this fixture: the CAR(1) resampler reaches 0.80 of the
    Monte-Carlo spread, the AR(1) one 0.46. The band below admits the former
    and excludes the latter.
    """
    n = 80
    t = np.concatenate(
        [
            np.linspace(0.0, 10.0, n // 2, endpoint=False),
            np.linspace(10.0, 2.0e7, n - n // 2),
        ]
    )
    truth = np.array([0.5, 1.0e-6, 0.5, 1.0])
    theta, sigma = 1.0, 0.02
    # Detuned seed: never hand a fit its own answer, or an immobile parameter
    # reads as a perfect one.
    seed = truth * np.array([1.3, 2.0, 0.7, 0.4])

    rng = np.random.default_rng(11)
    gold = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(25):
            y = M2(t, truth) + car1_resample(rng, t, theta, sigma)
            gold.append(abs(fit_gls_ar1(M2, t, y, seed).params[3]))
    sd_gold = float(np.std(gold))
    # The Monte-Carlo reference itself must be sane, or the ratio below is a
    # comparison against noise.
    assert float(np.median(gold)) == pytest.approx(1.0, rel=0.1), gold
    assert sd_gold > 0.0

    y0 = M2(t, truth) + car1_resample(np.random.default_rng(5), t, theta, sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, _ = parametric_bootstrap(
            M2, t, y0, seed, B=40, rng=np.random.default_rng(6)
        )
    ratio = float(np.std(np.abs(samples[:, 3]))) / sd_gold
    assert 0.6 < ratio < 1.6, ratio


# ---------------------------------------------------------------------------
# Round-18 review (external, PR #127): the fitted CAR(1) rate is a parameter
# ---------------------------------------------------------------------------


def _decaying_series(t: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(20260903)
    return np.exp(-1.0 * t) + 1.0e-3 * rng.standard_normal(t.size)


def test_aicc_counts_the_fitted_car1_rate_on_a_non_uniform_grid():
    """``theta`` is estimated from THIS data set, so it belongs in ``k``.

    It is re-fitted for every candidate model and enters that model's maximised
    likelihood through the whitening. Because the small-sample correction
    ``2k(k+1)/(N_eff-k-1)`` is nonlinear in ``k``, leaving it out is not a
    constant offset: it under-penalises the higher-dimensional candidates
    exactly when ``N_eff`` is small, which can move the selected relaxation
    model and with it the reported A-class.
    """
    t = np.concatenate(
        [np.linspace(0.0, 0.1, 40, endpoint=False), np.linspace(0.1, 8.0, 40)]
    )
    assert not is_uniform_grid(t), "fixture must exercise the CAR(1) path"
    y = _decaying_series(t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, _ = _fit_with_model("M1", t, y)

    assert np.isfinite(fit.residual_theta_car1), "fixture must fit a CAR(1) rate"
    p = int(np.asarray(fit.params).size)
    assert fit.aicc == pytest.approx(
        aicc(fit.log_likelihood, p + 1, fit.n_eff), rel=0.0, abs=0.0
    )
    # DISCRIMINATION: the pre-fix count must be a DIFFERENT number here, or the
    # assertion above would pass without the repair.
    assert aicc(fit.log_likelihood, p, fit.n_eff) != aicc(
        fit.log_likelihood, p + 1, fit.n_eff
    )


def test_aicc_parameter_count_is_unchanged_on_a_uniform_grid():
    """Negative control: the historical discrete-AR(1) path must not move.

    ``rho`` is estimated there too, but counting it would re-rank every
    existing uniform-grid result; that convention change is deliberately NOT
    part of this repair (see the comment in ``_fit_with_model``).
    """
    t = np.linspace(0.0, 8.0, 80)
    assert is_uniform_grid(t)
    y = _decaying_series(t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, _ = _fit_with_model("M1", t, y)

    assert not np.isfinite(fit.residual_theta_car1), "uniform grid fits no theta"
    p = int(np.asarray(fit.params).size)
    assert fit.aicc == pytest.approx(
        aicc(fit.log_likelihood, p, fit.n_eff), rel=0.0, abs=0.0
    )
