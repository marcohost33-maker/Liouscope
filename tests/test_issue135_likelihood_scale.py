"""Issue #135: scale-safe Gaussian likelihood and degenerate RSS semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from liouscope.diagnostics import relaxation as relaxation_mod
from liouscope.fitting.aicc import aicc, choose_model, gaussian_log_likelihood
from liouscope.fitting.bootstrap import parametric_bootstrap
from liouscope.fitting.gls import GLSFitOutput, fit_gls_ar1
from liouscope.fitting.models import M0
from liouscope.numerics.norms import scaled_log_sum_squares


def test_scaled_log_rss_spans_underflow_and_overflow_regimes():
    tiny = np.array([1.0e-320, -2.0e-320])
    huge = np.array([1.0e308, -1.0e308])
    log_tiny = scaled_log_sum_squares(tiny)
    log_huge = scaled_log_sum_squares(huge)
    assert np.isfinite(log_tiny)
    assert np.isfinite(log_huge)
    assert log_tiny == pytest.approx(math.log(5.0) + 2.0 * math.log(1.0e-320), rel=2e-5)
    assert log_huge == pytest.approx(math.log(2.0) + 2.0 * math.log(1.0e308), rel=1e-14)
    assert scaled_log_sum_squares(np.zeros(4)) == float("-inf")


def test_profile_loglikelihood_is_scale_covariant_and_delta_invariant():
    x = np.linspace(0.2, 1.2, 64)
    r0 = 0.7 + x + 0.03 * np.sin(5.0 * x)
    r1 = 0.9 + 1.1 * x - 0.02 * np.cos(3.0 * x)
    reference_delta = None
    reference_winner = None
    for scale in (1.0e-150, 1.0e-40, 1.0, 1.0e40, 1.0e150):
        ll0 = gaussian_log_likelihood(scale * r0)
        ll1 = gaussian_log_likelihood(scale * r1)
        assert np.isfinite(ll0) and np.isfinite(ll1)
        delta = ll1 - ll0
        scores = {
            "M0": aicc(ll0, k=2, n_eff=64.0),
            "M1": aicc(ll1, k=3, n_eff=64.0),
        }
        winner = choose_model(scores)
        if reference_delta is None:
            reference_delta = delta
            reference_winner = winner
        else:
            assert delta == pytest.approx(reference_delta, rel=1e-11, abs=1e-10)
            assert winner == reference_winner


def test_profile_loglikelihood_handles_true_rss_above_float_range():
    ll = gaussian_log_likelihood(np.array([1.0e308, -1.0e308]))
    assert np.isfinite(ll)


def test_zero_rss_unknown_sigma_is_unavailable_but_known_sigma_is_valid():
    residuals = np.zeros(8)
    assert np.isnan(gaussian_log_likelihood(residuals))
    ll = gaussian_log_likelihood(residuals, sigma=2.0)
    expected = -0.5 * residuals.size * math.log(2.0 * math.pi * 4.0)
    assert ll == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_gls_exact_fit_marks_likelihood_degenerate_and_bootstrap_refuses():
    t = np.linspace(0.0, 4.0, 48)
    p0 = np.array([1.25, 0.6])
    y = M0(t, p0)
    with pytest.warns(RuntimeWarning, match="likelihood/AICc/CI evidence is unavailable"):
        fit = fit_gls_ar1(M0, t, y, p0, n_iters=1)
    assert not fit.success
    assert fit.likelihood_degenerate
    assert not fit.degenerate
    assert np.isnan(fit.sigma)
    assert np.isnan(fit.log_likelihood)
    # The repository treats unexpected warnings as errors. Here the warning is
    # part of the intended public contract: the base fit first reports why its
    # likelihood evidence is unusable, then bootstrap refuses the non-estimate.
    with pytest.warns(RuntimeWarning, match="likelihood/AICc/CI evidence is unavailable"):
        with pytest.raises(RuntimeError, match="likelihood scale is degenerate"):
            parametric_bootstrap(M0, t, y, p0, B=4)


def test_ordinary_noisy_gls_positive_control_is_not_likelihood_degenerate(rng):
    t = np.linspace(0.0, 4.0, 80)
    p = np.array([1.25, 0.6])
    y = M0(t, p) + 1.0e-3 * rng.standard_normal(t.size)
    fit = fit_gls_ar1(M0, t, y, p, n_iters=1)
    assert fit.success
    assert not fit.likelihood_degenerate
    assert np.isfinite(fit.sigma) and fit.sigma > 0.0
    assert np.isfinite(fit.log_likelihood)


def test_likelihood_degenerate_state_reaches_fitresult_and_is_nonselectable(monkeypatch):
    t = np.linspace(0.0, 1.0, 16)
    y = np.exp(-t)
    fake = GLSFitOutput(
        params=np.array([1.0, 1.0]),
        residuals=np.zeros_like(y),
        rho_ar1=0.0,
        sigma=float("nan"),
        log_likelihood=float("nan"),
        success=False,
        likelihood_degenerate=True,
    )
    monkeypatch.setattr(relaxation_mod, "fit_gls_ar1", lambda *args, **kwargs: fake)
    fit_result, _ = relaxation_mod._fit_with_model("M0", t, y)
    assert not fit_result.success
    assert fit_result.likelihood_degenerate
    assert np.isinf(fit_result.aicc)
