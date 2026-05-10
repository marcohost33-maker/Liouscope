"""Tests for resolvent layer D11b, D12, D13."""

from __future__ import annotations

from liouscope import build_liouvillian
from liouscope.diagnostics.resolvent import (
    compute_resolvent_layer,
    pseudospectral_radius_diag,
    resolvent_peak,
    resolvent_peak_curve,
    ridge_fwhm,
)


def test_resolvent_peak_positive(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    peak = resolvent_peak(L)
    assert peak > 0


def test_resolvent_curve_shapes(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    omegas, norms = resolvent_peak_curve(L, n_omega=40)
    assert omegas.shape == norms.shape == (40,)


def test_ridge_fwhm_nonneg(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    omegas, norms = resolvent_peak_curve(L, n_omega=80)
    assert ridge_fwhm(omegas, norms) >= 0


def test_pseudospectral_radius_nonneg(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    r = pseudospectral_radius_diag(L, eps=1.0e-2)
    assert r >= 0


def test_compute_resolvent_layer_returns_result(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    res = compute_resolvent_layer(L)
    assert res.resolvent_peak > 0
    assert res.ridge_fwhm >= 0
    assert res.pseudospectral_radius >= 0
