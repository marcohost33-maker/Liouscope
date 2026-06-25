"""Analytic gap-invariant oracles, reproducing the pack beleg-values (LIOU-F-019).

Scope note (Reality-Anchor): the *general* analytic oracles ``gap = gamma/2``
(amplitude damping) and ``gap = (g_up + g_down)/2`` (thermal qubit) already exist
in ``tests/test_qutip_spectral_oracle.py`` (PR#51, gamma=0.4 and g={0.2,0.5}).
The additive value here is reproducing the **exact pack parametrisation** of the
F-019 mini-oracle with our own run -- the eigen-reproduction Equalita left open:

* amplitude damping, several gamma -> ``|gap - gamma/2|`` at machine precision;
* thermal qubit with the pack parameters ``g_down=0.9, g_up=0.2,
  H = 0.5*omega*sigma_z, omega=1.3`` -> ``gap = (0.9+0.2)/2 = 0.55``.

The dephasing case ``gap = 2*gamma`` is deliberately NOT re-added (already canon
LIOU-RUN-002).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import build_liouvillian
from liouscope.diagnostics.spectral import compute_spectral_layer

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_RAISE = 0.5 * (_X - 1j * _Y)  # |1><0|, raises 0 -> 1
_LOWER = _RAISE.conj().T  # |0><1|, lowers 1 -> 0


@pytest.mark.parametrize("gamma", [0.2, 0.4, 0.9, 1.7])
def test_amplitude_damping_gap_is_gamma_over_two(gamma):
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [gamma])
    layer = compute_spectral_layer(L)
    assert layer.gap == pytest.approx(gamma / 2, abs=1e-12)
    assert layer.oscillating_gap == 0.0  # purely dissipative => no oscillation


def test_thermal_qubit_gap_pack_parameters():
    # Pack beleg: g_down=0.9, g_up=0.2, H = 0.5*omega*sigma_z, omega=1.3.
    g_down, g_up, omega = 0.9, 0.2, 1.3
    H = 0.5 * omega * _Z
    L = build_liouvillian(H, [_LOWER, _RAISE], [g_down, g_up])
    layer = compute_spectral_layer(L)
    expected = (g_up + g_down) / 2  # = 0.55
    assert layer.gap == pytest.approx(expected, abs=1e-9)
    # Detailed balance => symmetrised gaps coincide with the standard gap.
    assert layer.gns_gap == pytest.approx(layer.gap, abs=1e-6)
    assert layer.kms_gap == pytest.approx(layer.gap, abs=1e-6)


@qutip_required
def test_thermal_qubit_gap_pack_parameters_matches_qutip():
    import qutip

    g_down, g_up, omega = 0.9, 0.2, 1.3
    H = 0.5 * omega * _Z
    L = build_liouvillian(H, [_LOWER, _RAISE], [g_down, g_up])
    c_ops = [np.sqrt(g_down) * qutip.Qobj(_LOWER), np.sqrt(g_up) * qutip.Qobj(_RAISE)]
    qt = qutip.liouvillian(qutip.Qobj(H), c_ops).eigenenergies()
    nonzero = qt[np.abs(qt) > 1e-9]
    qt_gap = float(-np.max(np.real(nonzero)))
    layer = compute_spectral_layer(L)
    assert layer.gap == pytest.approx(qt_gap, abs=1e-9)
    assert layer.gap == pytest.approx(0.55, abs=1e-9)
