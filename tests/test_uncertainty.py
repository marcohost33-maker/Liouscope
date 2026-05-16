"""Smoke tests for diagnostics/uncertainty.py."""
from __future__ import annotations

import math

import pytest

from liouscope._types import RelaxationResult
from liouscope.diagnostics.uncertainty import compute_uncertainty_layer


def _mock_relaxation(lo: float = 0.95, hi: float = 1.05) -> RelaxationResult:
    """Build a minimal RelaxationResult with a chosen BCa interval."""
    return RelaxationResult(
        beta_D=1.0,
        ci_beta=(0.92, 1.08),
        bca_ci_beta=(lo, hi),
        d2b_beta=1.0,
        bootstrap_B=200,
        n_eff=1024.0,
    )


def test_uncertainty_u0_from_finite_ci():
    rel = _mock_relaxation(lo=0.9, hi=1.1)
    res = compute_uncertainty_layer(rel)
    assert res.fit_uncertainty == pytest.approx(0.1, abs=1.0e-9), (
        "U0 should be the half-width of the BCa interval."
    )


def test_uncertainty_u0_nan_on_non_finite_ci():
    rel = _mock_relaxation(lo=float("-inf"), hi=float("inf"))
    res = compute_uncertainty_layer(rel)
    assert math.isnan(res.fit_uncertainty)


def test_uncertainty_solver_default_and_override():
    rel = _mock_relaxation()
    default = compute_uncertainty_layer(rel)
    override = compute_uncertainty_layer(rel, solver_residual=1.0e-6)
    assert default.solver_uncertainty == pytest.approx(1.0e-10)
    assert override.solver_uncertainty == pytest.approx(1.0e-6)


def test_uncertainty_size_residual_optional():
    rel = _mock_relaxation()
    none_case = compute_uncertainty_layer(rel)
    with_size = compute_uncertainty_layer(rel, size_residual=3.0e-3)
    assert none_case.size_uncertainty is None
    assert with_size.size_uncertainty == pytest.approx(3.0e-3)


def test_uncertainty_bootstrap_passthrough():
    rel = _mock_relaxation()
    res = compute_uncertainty_layer(rel, bootstrap_B=512)
    assert res.bootstrap_B == 512
