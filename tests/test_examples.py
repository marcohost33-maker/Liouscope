"""Smoke tests for the V1-V5 validation system constructors."""

from __future__ import annotations

import numpy as np

from liouscope import diagnose, steady_state
from liouscope.examples import (
    all_systems,
    v1_qutrit,
    v2_dephasing_qubit,
    v3_amplitude_damped_qubit,
    v4_thermal_two_level,
    v5_jaynes_cummings,
)


def test_v1_constructor():
    sys = v1_qutrit()
    assert sys.L.shape == (9, 9)
    assert sys.rho_initial.shape == (3, 3)


def test_v2_constructor():
    sys = v2_dephasing_qubit()
    assert sys.L.shape == (4, 4)


def test_v3_constructor():
    sys = v3_amplitude_damped_qubit(gamma=0.4)
    assert sys.L.shape == (4, 4)


def test_v4_constructor():
    sys = v4_thermal_two_level(beta=1.0, omega=1.0)
    assert sys.L.shape == (4, 4)


def test_v5_constructor():
    sys = v5_jaynes_cummings(g=0.2, kappa=1.0, n_max=2)
    d = 2 * 3
    assert sys.L.shape == (d * d, d * d)


def test_all_systems_returns_five():
    sys_list = all_systems()
    assert len(sys_list) == 5


def test_v3_steady_state_is_ground_state(pauli):
    sys = v3_amplitude_damped_qubit(gamma=0.4)
    rho_ss = steady_state(sys.L)
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    np.testing.assert_allclose(rho_ss, expected, atol=1e-9)


def test_v4_detailed_balance_kms_equals_gns():
    sys = v4_thermal_two_level(beta=1.0, omega=1.0)
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    # In detailed balance, KMS and GNS gaps should agree closely.
    assert abs(report.spectral.kms_gap - report.spectral.gns_gap) < 0.05
