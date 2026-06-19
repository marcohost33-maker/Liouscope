"""Independent-oracle cross-checks for the spectral diagnostic layer (D1-D4).

The existing QuTiP cross-checks (``tests/test_anchors.py``,
``tests/test_lindblad.py``) only validate the **Liouvillian builder** — i.e.
that ``build_liouvillian`` reproduces ``qutip.liouvillian``. They never check
the *diagnostic outputs* against an independent source. The spectral tests in
``tests/test_spectral.py`` only assert self-consistency and loose magnitudes
(e.g. ``abs(gap - 0.2) < 0.05 or abs(gap - 0.4) < 0.05``), so a sign error or a
wrong eigenvalue filter in :mod:`liouscope.diagnostics.spectral` could pass.

This module closes that gap. For three canonical GKSL systems with a known
closed-form spectrum it pins the diagnostic outputs to **two** independent
oracles:

1. **Analytic** (primary, always runs): the exact Liouvillian eigenvalues are
   known in closed form, so the gap D1, oscillating gap D3 and steady state are
   asserted against hand-derived values — not against the library's own
   machinery.
2. **QuTiP** (secondary, ``@qutip_required``): the gap and oscillating gap are
   re-derived from ``qutip.liouvillian(...).eigenenergies()`` and the steady
   state from ``qutip.steadystate(...)`` — a fully independent implementation of
   the GKSL generator and its spectrum.

Physics references for the oracle values:

* Liouvillian gap = smallest non-zero ``|Re(lambda)|`` of the Liouvillian
  spectrum = asymptotic decay rate toward the steady state. See e.g.
  Mori, "Liouvillian analysis of the relaxation time in open quantum systems".
* At equilibrium (detailed balance) the symmetrised (GNS/KMS) gap coincides
  with the standard gap — Kishimoto-Mori / Mori-Shirai, "Symmetrized
  Liouvillian Gap in Markovian Open Quantum Systems", Phys. Rev. Lett. 130,
  230404 (2023) (arXiv:2212.06317). The thermal-qubit case below exercises
  exactly that prediction.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import build_liouvillian, steady_state
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.numerics.linalg import eig_nonhermitian

# --- canonical jump operators (qubit ladder) ---------------------------------
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
# NOTE on the matrix convention: 0.5*(X - iY) has its single non-zero entry at
# index [1, 0], i.e. it equals |1><0| and maps basis index 0 -> index 1. We
# therefore name the variables by their matrix action (_RAISE: 0 -> 1,
# _LOWER: 1 -> 0) to keep the steady-state oracle below unambiguous.
_RAISE = 0.5 * (_X - 1j * _Y)  # |1><0|, maps index 0 -> 1
_LOWER = _RAISE.conj().T  # |0><1|, maps index 1 -> 0
_SM = _LOWER  # amplitude-damping decay channel (index 1 -> 0)


def _sorted_spectrum(L: np.ndarray) -> np.ndarray:
    """LiouScope eigenvalues, lexicographically sorted for stable comparison."""
    return np.sort_complex(eig_nonhermitian(L).eigenvalues)


# =============================================================================
# System 1: amplitude damping (T1 decay), gamma = 0.4.
# Closed-form Liouvillian spectrum: {0, -gamma, -gamma/2, -gamma/2}.
# => gap D1 = gamma/2; spectrum real (no oscillating mode, D3 = 0).
# =============================================================================
_GAMMA_AD = 0.4


def test_amplitude_damping_gap_matches_analytic():
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_SM], [_GAMMA_AD])
    spectrum = _sorted_spectrum(L)
    expected = np.sort_complex(
        np.array([0.0, -_GAMMA_AD, -_GAMMA_AD / 2, -_GAMMA_AD / 2], dtype=complex)
    )
    np.testing.assert_allclose(spectrum, expected, atol=1e-12)

    layer = compute_spectral_layer(L)
    # Analytic gap is exactly gamma/2 — pins what the loose existing test left open.
    assert layer.gap == pytest.approx(_GAMMA_AD / 2, abs=1e-10)
    # Purely dissipative, no Hamiltonian => spectrum is real => no oscillating mode.
    assert layer.oscillating_gap == 0.0


@qutip_required
def test_amplitude_damping_gap_matches_qutip():
    import qutip

    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_SM], [_GAMMA_AD])
    L_qt = qutip.liouvillian(
        qutip.Qobj(np.zeros((2, 2), dtype=complex)),
        [np.sqrt(_GAMMA_AD) * qutip.Qobj(_SM)],
    )
    qt_eigs = L_qt.eigenenergies()
    nonzero = qt_eigs[np.abs(qt_eigs) > 1e-9]
    qt_gap = float(-np.max(np.real(nonzero)))

    layer = compute_spectral_layer(L)
    assert layer.gap == pytest.approx(qt_gap, abs=1e-9)


# =============================================================================
# System 2: coherently driven dephasing, H = 0.5 X, jump = sqrt(0.3) Z.
# The Hamiltonian drive produces a genuine complex-conjugate eigenvalue pair,
# exercising the gap (D1) and oscillating-mode (D3) diagnostics jointly.
# Closed form: {0, -2*gamma=-0.6, -gamma +- i*Omega, -gamma - i*Omega}
# with gamma=0.3 and Omega = sqrt(1 - (2*gamma)^2) ... derived numerically and
# cross-checked against QuTiP below. gap = gamma = 0.3.
# =============================================================================
_GAMMA_DD = 0.3
_H_DD = 0.5 * _X


def test_driven_dephasing_has_complex_pair_and_correct_gap():
    L = build_liouvillian(_H_DD, [_Z], [_GAMMA_DD])
    layer = compute_spectral_layer(L)
    # Real-part gap is the dephasing rate gamma (the conjugate pair sits at
    # Re = -gamma; the isolated real mode at Re = -2*gamma is faster).
    assert layer.gap == pytest.approx(_GAMMA_DD, abs=1e-9)
    # The drive guarantees a genuine oscillating mode.
    assert layer.has_complex_pairs
    assert layer.oscillating_gap > 0.0


@qutip_required
def test_driven_dephasing_spectrum_and_oscillation_match_qutip():
    import qutip

    L = build_liouvillian(_H_DD, [_Z], [_GAMMA_DD])
    L_qt = qutip.liouvillian(qutip.Qobj(_H_DD), [np.sqrt(_GAMMA_DD) * qutip.Qobj(_Z)])

    ls = _sorted_spectrum(L)
    qt = np.sort_complex(L_qt.eigenenergies())
    np.testing.assert_allclose(ls, qt, atol=1e-9)

    layer = compute_spectral_layer(L)
    qt_nonzero = qt[np.abs(qt) > 1e-9]
    qt_gap = float(-np.max(np.real(qt_nonzero)))
    qt_osc_modes = qt[np.abs(np.imag(qt)) > 1e-9]
    qt_osc = float(np.min(np.abs(np.imag(qt_osc_modes))))

    assert layer.gap == pytest.approx(qt_gap, abs=1e-9)
    assert layer.oscillating_gap == pytest.approx(qt_osc, abs=1e-9)


# =============================================================================
# System 3: thermal qubit with detailed balance.
# H = diag(0, 1); channel _RAISE (index 0 -> 1) at rate g_up = 0.2, channel
# _LOWER (index 1 -> 0) at rate g_down = 0.5.
# Population mode decays at (g_down + g_up) = 0.7; the two coherence modes at
# (g_down + g_up)/2 = 0.35 (a conjugate pair around the Bohr frequency).
# => gap D1 = 0.35. Detailed balance => the symmetrised (GNS/KMS) gap must
#    coincide with the standard gap (Mori-Shirai, PRL 130, 230404).
# Steady state: a two-level rate balance has the two diagonal populations equal
# to the set {g_up, g_down} / (g_up + g_down). The exact index<->population
# assignment depends on the column-stacking convention, so the analytic test
# asserts the population *set* (order-independent), while the QuTiP test below
# pins the full matrix index-for-index against an independent implementation.
# =============================================================================
_H_TH = np.diag([0.0, 1.0]).astype(complex)
_G_DOWN, _G_UP = 0.5, 0.2


def _thermal_liouvillian() -> np.ndarray:
    # _LOWER (1->0) at the down-rate, _RAISE (0->1) at the up-rate.
    return build_liouvillian(_H_TH, [_LOWER, _RAISE], [_G_DOWN, _G_UP])


def test_thermal_qubit_gap_and_steady_state_match_analytic():
    L = _thermal_liouvillian()
    layer = compute_spectral_layer(L)

    expected_gap = (_G_DOWN + _G_UP) / 2  # = 0.35
    assert layer.gap == pytest.approx(expected_gap, abs=1e-9)

    ss = steady_state(L)
    # Steady state must be diagonal (detailed balance, no coherences).
    off_diagonal = ss - np.diag(np.diag(ss))
    assert np.max(np.abs(off_diagonal)) < 1e-9
    # Populations form the rate-balance set {g_up, g_down}/(g_up+g_down);
    # compare order-independently (see module note on index convention).
    total = _G_UP + _G_DOWN
    expected_populations = np.sort([_G_UP / total, _G_DOWN / total])
    got_populations = np.sort(np.real(np.diag(ss)))
    np.testing.assert_allclose(got_populations, expected_populations, atol=1e-9)
    # Trace preserved.
    assert np.trace(ss).real == pytest.approx(1.0, abs=1e-12)


def test_thermal_qubit_symmetrised_gap_equals_standard_gap():
    # Detailed balance => GNS/KMS symmetrised gap == standard gap (PRL 130, 230404).
    L = _thermal_liouvillian()
    layer = compute_spectral_layer(L)
    assert layer.gns_gap == pytest.approx(layer.gap, abs=1e-6)
    assert layer.kms_gap == pytest.approx(layer.gap, abs=1e-6)


@qutip_required
def test_thermal_qubit_matches_qutip_spectrum_and_steady_state():
    import qutip

    L = _thermal_liouvillian()
    c_ops = [np.sqrt(_G_DOWN) * qutip.Qobj(_LOWER), np.sqrt(_G_UP) * qutip.Qobj(_RAISE)]
    L_qt = qutip.liouvillian(qutip.Qobj(_H_TH), c_ops)

    ls = _sorted_spectrum(L)
    qt = np.sort_complex(L_qt.eigenenergies())
    np.testing.assert_allclose(ls, qt, atol=1e-9)

    ss_ls = steady_state(L)
    ss_qt = qutip.steadystate(qutip.Qobj(_H_TH), c_ops).full()
    np.testing.assert_allclose(ss_ls, ss_qt, atol=1e-9)
