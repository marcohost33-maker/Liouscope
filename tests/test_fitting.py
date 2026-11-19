"""Tests for the fitting pipeline."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope.fitting.aicc import aicc, choose_model, gaussian_log_likelihood
from liouscope.fitting.bootstrap import bca_ci, parametric_bootstrap
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M0
from liouscope.fitting.neff import (
    ar1_correlation,
    ar1_correlation_corrected,
    estimate_neff_geyer,
)
from liouscope.fitting.prony import prony_seed


def test_aicc_inf_when_neff_too_small():
    assert np.isinf(aicc(-10.0, 5, 4.0))


def test_choose_model_picks_smallest_finite():
    aiccs = {"M0": 100.0, "M1": 105.0, "M3a": float("inf")}
    assert choose_model(aiccs) == "M0"


def test_neff_geyer_below_n_for_correlated(rng):
    n = 300
    rho = 0.85
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    n_eff = estimate_neff_geyer(x)
    assert n_eff < n
    assert n_eff > 1


def test_ar1_correlation_matches_known(rng):
    n = 1000
    rho = 0.6
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    rho_hat = ar1_correlation(x)
    assert abs(rho_hat - rho) < 0.1


def _mc_ar1_mean_bias(n, rho_true, estimator, reps=4000, seed=12345):
    """Monte-Carlo mean bias of an AR(1) lag-1 estimator at sample size n."""
    rng = np.random.default_rng(seed)
    ests = np.empty(reps)
    for r in range(reps):
        eps = rng.standard_normal(n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho_true * x[i - 1] + eps[i]
        ests[r] = estimator(x)
    return float(np.mean(ests) - rho_true)


@pytest.mark.parametrize("n", [20, 40])
def test_ar1_bias_correction_reduces_bias(n):
    """FAILS-BEFORE (S2): raw rho_hat is materially downward-biased at small n.

    The audit measured raw bias of about -0.16 @ n=20 and -0.08 @ n=40 for
    rho=0.5. The corrected estimator must shrink the magnitude of that bias
    substantially. (Mean over 4000 AR(1) replicates, fixed seed.)
    """
    rho_true = 0.5
    raw_bias = _mc_ar1_mean_bias(n, rho_true, ar1_correlation)
    corr_bias = _mc_ar1_mean_bias(
        n,
        rho_true,
        lambda x: ar1_correlation_corrected(x, warn_small_n=False),
    )
    # Raw is downward-biased (the bug we are fixing).
    assert raw_bias < -0.05, f"expected downward raw bias, got {raw_bias:+.4f}"
    # Correction removes most of it: at least a 1.5x reduction in magnitude.
    assert abs(corr_bias) < abs(raw_bias) / 1.5, (
        f"n={n}: raw_bias={raw_bias:+.4f} corr_bias={corr_bias:+.4f} "
        "(correction did not reduce bias enough)"
    )


def test_ar1_corrected_warns_at_small_n():
    """The corrected estimator warns when n <= 40 (over-confident CI risk)."""
    rng = np.random.default_rng(0)
    n = 30
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + eps[i]
    with pytest.warns(RuntimeWarning, match="over-confident"):
        ar1_correlation_corrected(x)


def test_ar1_corrected_no_warn_large_n():
    """No small-n warning for comfortably large series; value near truth."""
    rng = np.random.default_rng(0)
    n = 2000
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + eps[i]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        rho_c = ar1_correlation_corrected(x)
    assert abs(rho_c - 0.6) < 0.1


def test_gls_ar1_recovers_known_decay(rng):
    t = np.linspace(0, 5, 100)
    A_true, alpha_true = 1.5, 0.7
    y = A_true * np.exp(-alpha_true * t) + 0.01 * rng.standard_normal(t.size)
    fit = fit_gls_ar1(M0, t, y, np.array([1.0, 0.5]))
    assert fit.success
    assert abs(fit.params[0] - A_true) < 0.05
    assert abs(fit.params[1] - alpha_true) < 0.05


def test_gls_ar1_log_likelihood_is_exact_prais_winsten(rng):
    # _whiten keeps observation 0 (Prais-Winsten), so the reported log-likelihood
    # must include the transform's log-Jacobian 0.5*log(1-rho^2). Regression
    # against the pre-fix hybrid value (conditional likelihood on an exact
    # transform) that biased cross-model AICc toward under-fitting high-rho fits.
    t = np.linspace(0, 6, 140)
    rho_true = 0.6
    nu = 0.05 * rng.standard_normal(t.size)
    eps = np.empty(t.size)
    eps[0] = nu[0] / np.sqrt(1.0 - rho_true**2)
    for i in range(1, t.size):
        eps[i] = rho_true * eps[i - 1] + nu[i]
    y = 1.2 * np.exp(-0.7 * t) + eps
    fit = fit_gls_ar1(M0, t, y, np.array([1.0, 0.5]))

    rho = fit.rho_ar1
    whitened = np.empty_like(fit.residuals)
    whitened[0] = np.sqrt(max(1.0 - rho * rho, 1.0e-12)) * fit.residuals[0]
    whitened[1:] = fit.residuals[1:] - rho * fit.residuals[:-1]
    n = whitened.size
    sig = fit.sigma
    exact = (
        -0.5 * n * np.log(2.0 * np.pi * sig**2)
        - 0.5 * float(np.dot(whitened, whitened)) / sig**2
        + 0.5 * np.log(1.0 - rho * rho)
    )
    assert fit.log_likelihood == pytest.approx(exact, rel=1e-9, abs=1e-9)


def test_bootstrap_returns_correct_shape(rng):
    t = np.linspace(0, 5, 80)
    y = 1.0 * np.exp(-0.5 * t) + 0.02 * rng.standard_normal(t.size)
    samples, theta_hat = parametric_bootstrap(M0, t, y, np.array([1.0, 0.5]), B=40)
    assert samples.shape == (40, 2)
    assert theta_hat.shape == (2,)


def test_bca_ci_contains_truth(rng):
    t = np.linspace(0, 5, 100)
    alpha_true = 0.6
    y = 1.0 * np.exp(-alpha_true * t) + 0.02 * rng.standard_normal(t.size)
    samples, theta_hat = parametric_bootstrap(M0, t, y, np.array([1.0, 0.5]), B=100)
    cis = bca_ci(samples, theta_hat)
    assert cis.shape == (2, 2)
    lo, hi = cis[1]
    # Confidence interval for alpha should contain the truth at 95% level.
    assert lo <= alpha_true + 0.1
    assert hi >= alpha_true - 0.1


def test_bca_z0_half_correction_for_ties():
    """FAILS-BEFORE (S3): ties at theta_hat sent prop_below to 0 (z0 -> -inf).

    With many bootstrap replicates exactly equal to theta_hat, a strict ``<``
    count yields proportion 0, so z0 is clamped to the -1e-9 floor (z0 ~ -6)
    and the CI is wildly skewed. The Efron-1987 half-correction counts ties at
    half weight, giving a finite, sensible z0 and a non-degenerate interval.
    """
    # 3 of 5 replicates land exactly on theta_hat=1.0; none strictly below.
    samples = np.array([[1.0], [1.0], [1.0], [2.0], [3.0]])
    theta_hat = np.array([1.0])
    cis = bca_ci(samples, theta_hat, alpha=0.05)
    lo, hi = cis[0]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    # Old strict-< behaviour clamped z0 ~ -6, collapsing the interval onto the
    # minimum (lo == hi == 1.0). The half-correction must avoid that collapse.
    assert hi > 1.0, f"interval collapsed to the minimum (hi={hi}); ties bug"


def test_bca_quantile_is_interpolated_not_nearest_rank():
    """FAILS-BEFORE (S4): nearest-rank made CI endpoints granular for small B.

    With z0=0 and a=0 (symmetric, no acceleration), the BCa percentiles are
    the plain 2.5%/97.5% quantiles. On a small, evenly spaced sample the
    interpolated endpoints must NOT coincide with raw order statistics, which
    is what nearest-rank would have returned.
    """
    # No bias / no acceleration: samples symmetric about their median, mean
    # equal to theta_hat so prop_below = 0.5 -> z0 = 0.
    boot = np.linspace(0.0, 100.0, 9)  # 0,12.5,...,100 ; median 50
    samples = boot.reshape(-1, 1)
    theta_hat = np.array([50.0])
    cis = bca_ci(samples, theta_hat, alpha=0.10)  # 5% / 95% percentiles
    lo, hi = cis[0]
    expected_lo = float(np.quantile(boot, 0.05, method="linear"))
    expected_hi = float(np.quantile(boot, 0.95, method="linear"))
    np.testing.assert_allclose([lo, hi], [expected_lo, expected_hi], atol=1e-9)
    # And these interpolated values are strictly between order statistics
    # (i.e. not equal to any raw sample) -> proves interpolation happened.
    assert not np.any(np.isclose(lo, boot)), "lo fell on a raw order statistic"
    assert not np.any(np.isclose(hi, boot)), "hi fell on a raw order statistic"


def test_prony_seed_returns_finite(rng):
    t = np.linspace(0, 5, 60)
    y = np.exp(-0.4 * t) * np.cos(2.0 * t)
    seed = prony_seed(t, y)
    assert all(np.isfinite(seed))
    A_est, beta_est, omega_est, _ = seed
    assert beta_est > 0
    assert omega_est > 0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_prony_seed_non_finite_input_falls_back(bad_value):
    """FAILS-BEFORE: non-finite signals made lstsq raise LinAlgError.

    The pre-hardening code propagated ``numpy.linalg.LinAlgError`` ("SVD did
    not converge") on all-NaN / all-inf input. The guarded version must warn
    and return a finite default seed instead of raising.
    """
    t = np.linspace(0.0, 1.0, 12)
    y = np.full(12, bad_value, dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seed = prony_seed(t, y)
    assert all(np.isfinite(v) for v in seed)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_prony_seed_constant_signal_is_finite():
    """A flat (zero-oscillation) signal has a (near-)singular Hankel matrix."""
    t = np.linspace(0.0, 1.0, 16)
    y = np.full(16, 3.0)
    seed = prony_seed(t, y)
    assert all(np.isfinite(v) for v in seed)
    assert seed[0] > 0  # amplitude must be positive


def test_prony_seed_all_zero_signal_has_positive_amplitude():
    """Default seed must not return a zero amplitude (degenerate fit start)."""
    t = np.linspace(0.0, 1.0, 12)
    y = np.zeros(12)
    seed = prony_seed(t, y)
    assert all(np.isfinite(v) for v in seed)
    assert seed[0] > 0


def test_prony_seed_short_signal_falls_back():
    """Fewer than 6 samples -> immediate finite fallback (no Hankel solve)."""
    t = np.linspace(0.0, 1.0, 4)
    y = np.exp(-t)
    seed = prony_seed(t, y)
    assert len(seed) == 4
    assert all(np.isfinite(v) for v in seed)


def test_prony_seed_non_uniform_sampling_falls_back():
    """Non-uniform ``t`` (Prony assumes uniform) -> decaying-oscillator seed.

    The fallback rates are GRID-RELATIVE (sixteenth review round): ``beta``
    and ``omega`` carry rate dimension, so the former absolute ``(1, 1)``
    seeded the M3b fit orders of magnitude off on valid grids far from unit
    span, and least-squares "converged" onto the seed itself.
    """
    t = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1])
    y = np.exp(-0.5 * t) * np.cos(2.0 * t)
    seed = prony_seed(t, y)
    assert all(np.isfinite(v) for v in seed)
    assert seed[1] == pytest.approx(5.0 / 2.1)  # beta = FRAC / t_span
    assert seed[2] == pytest.approx(5.0 / 2.1)


def test_prony_seed_fallback_reproduces_historical_at_span_five():
    """frac = 5.0 was chosen to keep the canonical t_span = 5 seeds unchanged."""
    t = np.concatenate([np.linspace(0.0, 4.0, 4), [5.0]])  # non-uniform, span 5
    y = np.exp(-t)
    seed = prony_seed(t, y)
    assert seed[1] == pytest.approx(1.0)
    assert seed[2] == pytest.approx(1.0)


def test_prony_seed_fallback_is_unit_invariant():
    """A pure rescale t -> t/c must rescale the fallback rates by c exactly."""
    t = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1])
    y = np.exp(-0.5 * t) * np.cos(2.0 * t)
    base = prony_seed(t, y)
    for c in (1.0e-6, 1.0e6):
        scaled = prony_seed(t / c, y)
        assert scaled[1] == pytest.approx(base[1] * c)
        assert scaled[2] == pytest.approx(base[2] * c)


def test_prony_seed_too_few_for_model_order_falls_back():
    """N <= 2p+2 (here N=6, p=2) leaves no rows for the Hankel solve."""
    t = np.linspace(0.0, 1.0, 6)
    y = np.exp(-t)
    seed = prony_seed(t, y)
    assert all(np.isfinite(v) for v in seed)
    assert seed[1] == pytest.approx(5.0) and seed[2] == pytest.approx(5.0)


def test_gaussian_log_likelihood_returns_finite(rng):
    res = rng.standard_normal(50)
    ll = gaussian_log_likelihood(res)
    assert np.isfinite(ll)


def test_m3b_fit_is_unit_invariant_on_a_non_uniform_grid():
    """Round-16: the grid-relative Prony fallback keeps the fitted M3b rate
    unit-invariant where the absolute (1, 1) seed made least-squares
    "converge" onto the seed itself (measured: beta = 1 for a true 5e-8)."""
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M3b

    for c in (1.0, 1.0e-6, 1.0e6):
        t = (np.linspace(0.0, 1.0, 80) ** 1.5) * 10.0 / c
        y = np.exp(-0.05 * c * t) * np.cos(2.0 * c * t)
        fit = fit_gls_ar1(M3b, t, y, np.asarray(prony_seed(t, y)))
        assert fit.success
        assert fit.params[1] == pytest.approx(0.05 * c, rel=1e-3)
        assert abs(fit.params[2]) == pytest.approx(2.0 * c, rel=1e-3)


# --------------------------------------------------------------------------
# PR #127 external review, finding 4: on the two-scale relaxation grid the
# span-derived fallback seed sits on the SLOW gap scale, and the M3b fit
# converges onto a spurious solution that the historical (1, 1) seed did not
# produce. The grid carries a uniform fine head precisely to resolve the fast
# dynamics; the seed must be estimated there.
# --------------------------------------------------------------------------

# The measured regression case. On ``default_relaxation_grid(1e-4,
# fast_rate=1.0)`` (80 points, uniform head of 41 at dt = 0.25, span 1e5) the
# span seed 5e-5 drove the M3b fit to |omega| = 0.0366 for this exact curve,
# while the historical (1, 1) seed recovered it. Over 60 random curves from
# the same family the span seed failed twice and (1, 1) never; over 60 slower
# curves (beta, omega in [1e-4, 1e-1]) the span seed failed 41 times.
_REGRESSION_BETA = 0.09044
_REGRESSION_OMEGA = 1.679
_REGRESSION_PHI = 0.138


def _two_scale_grid():
    from liouscope.diagnostics.relaxation import default_relaxation_grid

    t = default_relaxation_grid(1.0e-4, fast_rate=1.0)
    assert not np.allclose(np.diff(t), t[1] - t[0]), "fixture must be non-uniform"
    return t


def test_prony_seed_uses_the_uniform_head_of_a_two_scale_grid():
    """A non-uniform grid with a usable uniform prefix is estimated, not guessed."""
    from liouscope.fitting.prony import _FALLBACK_RATE_FRAC

    t = _two_scale_grid()
    y = np.exp(-_REGRESSION_BETA * t) * np.cos(_REGRESSION_OMEGA * t + _REGRESSION_PHI)
    _amp, beta, omega, _phi = prony_seed(t, y)

    span_rate = _FALLBACK_RATE_FRAC / float(t[-1] - t[0])
    assert beta != pytest.approx(span_rate), "still seeding off the total span"
    assert omega != pytest.approx(span_rate)
    # The head resolves the fast scale, so the seeded omega must be within an
    # order of magnitude of the truth rather than four below it.
    assert 0.1 * _REGRESSION_OMEGA < omega < 10.0 * _REGRESSION_OMEGA


def test_m3b_recovers_the_case_the_span_seed_lost():
    """End to end: the seed change must move the FIT, not just the seed."""
    from liouscope.fitting.models import M3b

    t = _two_scale_grid()
    y = np.exp(-_REGRESSION_BETA * t) * np.cos(_REGRESSION_OMEGA * t + _REGRESSION_PHI)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_gls_ar1(M3b, t, y, np.asarray(prony_seed(t, y)))
    assert fit.params[1] == pytest.approx(_REGRESSION_BETA, rel=1.0e-2)
    assert abs(fit.params[2]) == pytest.approx(_REGRESSION_OMEGA, rel=1.0e-2)


def test_prony_seed_head_estimate_does_not_disturb_a_uniform_grid():
    """Over-correction control: the uniform path is untouched."""
    t = np.linspace(0.0, 5.0, 60)
    y = np.exp(-0.4 * t) * np.cos(2.0 * t)
    _amp, beta, omega, _phi = prony_seed(t, y)
    assert beta == pytest.approx(0.4, rel=1.0e-6)
    assert omega == pytest.approx(2.0, rel=1.0e-6)


def test_prony_seed_keeps_the_grid_relative_fallback_on_a_short_prefix():
    """No uniform prefix worth a Hankel system -> documented fallback stands."""
    t = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1])
    y = np.exp(-0.5 * t) * np.cos(2.0 * t)
    _amp, beta, omega, _phi = prony_seed(t, y)
    assert beta == pytest.approx(5.0 / 2.1)
    assert omega == pytest.approx(5.0 / 2.1)
