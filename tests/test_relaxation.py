"""Tests for relaxation layer D5-D7b and the fit hierarchy."""

from __future__ import annotations

import numpy as np

from liouscope import build_liouvillian
from liouscope.diagnostics.relaxation import (
    compute_relaxation_layer,
    entanglement_asymmetry,
    fidelity,
    relative_entropy,
    von_neumann_entropy,
)


def test_d5_vne_zero_for_pure_state(pauli):
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert von_neumann_entropy(rho) < 1e-9


def test_d5_vne_log2_for_maximally_mixed():
    rho = np.eye(2, dtype=complex) / 2
    assert abs(von_neumann_entropy(rho) - np.log(2)) < 1e-9


def test_d6_relative_entropy_zero_at_steady_state(pauli):
    rho = np.eye(2, dtype=complex) / 2
    assert relative_entropy(rho, rho) < 1e-9


def test_d6_relative_entropy_positive_off_steady_state():
    rho_initial = np.array([[1, 0], [0, 0]], dtype=complex)
    pi = np.eye(2, dtype=complex) / 2
    assert relative_entropy(rho_initial, pi) > 0


def test_d7_fidelity_one_for_identical():
    rho = np.diag([0.7, 0.3]).astype(complex)
    f = fidelity(rho, rho)
    assert abs(f - 1.0) < 1e-9


def test_d7b_entanglement_asymmetry_two_qubit_smoke():
    rho = np.diag([0.25, 0.25, 0.25, 0.25]).astype(complex)
    val = entanglement_asymmetry(rho)
    assert val >= 0 or np.isnan(val)


def test_compute_relaxation_layer_returns_result(pauli):
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    res = compute_relaxation_layer(
        L,
        rho_initial=rho0,
        t_grid=np.linspace(0, 8, 60),
        bootstrap_B=30,
    )
    assert res.aicc_model in ("M0", "M1", "M2", "M3a", "M3b")
    assert np.isfinite(res.beta_D)
