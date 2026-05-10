"""Tests for Mpemba layer D19-D20."""

from __future__ import annotations

import numpy as np

from liouscope import build_liouvillian
from liouscope.diagnostics.mpemba import compute_mpemba_layer, expansion_alpha, overlap_c1


def test_overlap_c1_zero_when_initial_is_steady(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = 0.5 * np.eye(2, dtype=complex)
    c1 = overlap_c1(L, rho0)
    assert c1 < 1e-9


def test_overlap_c1_nonzero_for_general_state(pauli):
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [pauli["Z"]], [0.3])
    rho0 = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
    c1 = overlap_c1(L, rho0)
    assert c1 >= 0


def test_expansion_alpha_finite(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
    alpha = expansion_alpha(L, rho0)
    assert np.isfinite(alpha)


def test_compute_mpemba_layer_returns_result(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
    res = compute_mpemba_layer(L, rho0)
    assert res.overlap_c1 >= 0
    assert np.isfinite(res.expansion_alpha)
