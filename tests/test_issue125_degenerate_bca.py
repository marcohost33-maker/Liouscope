"""Regression tests for issue #125: degenerate BCa uncertainty claims."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.fitting.bootstrap import bca_ci


def test_degenerate_bootstrap_distribution_is_unavailable_not_zero_width() -> None:
    samples = np.ones((200, 1), dtype=float)
    theta_hat = np.array([1.0])

    with pytest.warns(RuntimeWarning, match="degenerate"):
        ci = bca_ci(samples, theta_hat)

    assert np.isnan(ci[0, 0])
    assert np.isnan(ci[0, 1])


def test_degenerate_parameter_does_not_withhold_an_independent_parameter() -> None:
    x = np.linspace(-1.0, 1.0, 200)
    samples = np.column_stack([np.ones_like(x), x])
    theta_hat = np.array([1.0, 0.0])

    with pytest.warns(RuntimeWarning, match="parameter 0"):
        ci = bca_ci(samples, theta_hat)

    assert np.all(np.isnan(ci[0]))
    assert np.all(np.isfinite(ci[1]))
    assert ci[1, 0] < ci[1, 1]


def test_ordinary_nondegenerate_distribution_keeps_a_finite_interval() -> None:
    rng = np.random.default_rng(125)
    samples = rng.normal(loc=0.7, scale=0.03, size=(1000, 1))
    theta_hat = np.array([0.7])

    ci = bca_ci(samples, theta_hat)

    assert np.all(np.isfinite(ci))
    assert ci[0, 0] < ci[0, 1]
    assert ci[0, 0] < theta_hat[0] < ci[0, 1]


def test_zero_width_after_adjusted_quantiles_is_not_reported_as_certainty() -> None:
    # Non-identical distribution, but >97.5% of its mass sits on one value.
    # The BCa adjusted endpoints can therefore collapse to that value. The
    # claim boundary, not merely the raw distribution, must catch the collapse.
    samples = np.concatenate([np.ones(199), np.array([2.0])]).reshape(-1, 1)
    theta_hat = np.array([1.0])

    with pytest.warns(RuntimeWarning, match="zero width"):
        ci = bca_ci(samples, theta_hat)

    assert np.all(np.isnan(ci[0]))
