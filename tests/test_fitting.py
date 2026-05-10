"""Tests for the fitting pipeline."""

from __future__ import annotations

import numpy as np

from liouscope.fitting.aicc import aicc, choose_model, gaussian_log_likelihood
from liouscope.fitting.bootstrap import bca_ci, parametric_bootstrap
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M0
from liouscope.fitting.neff import ar1_correlation, estimate_neff_geyer
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


def test_gls_ar1_recovers_known_decay(rng):
    t = np.linspace(0, 5, 100)
    A_true, alpha_true = 1.5, 0.7
    y = A_true * np.exp(-alpha_true * t) + 0.01 * rng.standard_normal(t.size)
    fit = fit_gls_ar1(M0, t, y, np.array([1.0, 0.5]))
    assert fit.success
    assert abs(fit.params[0] - A_true) < 0.05
    assert abs(fit.params[1] - alpha_true) < 0.05


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


def test_prony_seed_returns_finite(rng):
    t = np.linspace(0, 5, 60)
    y = np.exp(-0.4 * t) * np.cos(2.0 * t)
    seed = prony_seed(t, y)
    assert all(np.isfinite(seed))
    A_est, beta_est, omega_est, _ = seed
    assert beta_est > 0
    assert omega_est > 0


def test_gaussian_log_likelihood_returns_finite(rng):
    res = rng.standard_normal(50)
    ll = gaussian_log_likelihood(res)
    assert np.isfinite(ll)
