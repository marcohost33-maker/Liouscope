"""Tests for resolvent layer D11b, D12, D13."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

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


def test_resolvent_peak_curve_large_matches_dense(rng):
    # n > 128 routes resolvent_peak_curve through the SuperLU power-iteration
    # path of resolvent_norm. The profile must still match a dense
    # inverse + SVD reference evaluated on the same (omegas, shift) grid.
    n = 150
    L = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.05
    omegas, norms = resolvent_peak_curve(L, sigma=1.0e-2, n_omega=21)

    eigvals = sla.eigvals(L)
    shift = 1.0e-2 + float(np.max(np.real(eigvals)))
    eye = np.eye(n, dtype=complex)
    ref = np.array(
        [sla.svdvals(sla.inv((shift + 1j * w) * eye - L))[0] for w in omegas]
    )
    # The peak (the D11b diagnostic) is bang-on; the low-norm tail edges sit in
    # the clustered-singular-value regime where the power iteration is a slight
    # lower bound (documented ~1e-3), so the profile uses a looser tolerance.
    assert np.max(ref) == pytest.approx(np.max(norms), rel=1e-6)
    assert np.allclose(norms, ref, rtol=5e-3)
