"""Independent-oracle tests for the trace distance D_tr (LIOU-F-018).

Closed-form oracles (Nielsen & Chuang):

* ``D_tr(rho, rho) = 0`` (identity of indiscernibles).
* Orthogonal pure states ``|0>, |1>`` are perfectly distinguishable:
  ``D_tr = 1``.
* For commuting (diagonal) states ``diag(p, 1-p)`` and ``diag(q, 1-q)``,
  ``D_tr = |p - q|`` (the classical total-variation distance).
* Fuchs-van de Graaf: ``1 - F <= D_tr <= sqrt(1 - F^2)`` against the package's
  own Uhlmann fidelity (an independent diagnostic).
* Contractivity under a CPTP (dephasing) map: ``D_tr(Phi rho, Phi sigma) <=
  D_tr(rho, sigma)`` (data-processing inequality).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import build_liouvillian, diagnose
from liouscope.diagnostics.relaxation import fidelity, trace_distance
from liouscope.numerics.kronecker import unvec, vec

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_ket0 = np.array([[1, 0], [0, 0]], dtype=complex)
_ket1 = np.array([[0, 0], [0, 1]], dtype=complex)


def test_identical_states_zero_distance():
    rho = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)
    assert trace_distance(rho, rho) == pytest.approx(0.0, abs=1e-14)


def test_orthogonal_pure_states_distance_one():
    assert trace_distance(_ket0, _ket1) == pytest.approx(1.0, abs=1e-12)


def test_diagonal_states_equal_total_variation():
    for p, q in [(0.9, 0.1), (0.5, 0.5), (0.75, 0.2)]:
        rho = np.diag([p, 1 - p]).astype(complex)
        sigma = np.diag([q, 1 - q]).astype(complex)
        assert trace_distance(rho, sigma) == pytest.approx(abs(p - q), abs=1e-12)


def test_symmetry():
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = _random_density(rng)
        b = _random_density(rng)
        assert trace_distance(a, b) == pytest.approx(trace_distance(b, a), abs=1e-12)


def test_fuchs_van_de_graaf_bounds():
    # 1 - F <= D_tr <= sqrt(1 - F^2) for the Uhlmann fidelity F in [0, 1].
    rng = np.random.default_rng(11)
    for _ in range(20):
        rho = _random_density(rng)
        sigma = _random_density(rng)
        d_tr = trace_distance(rho, sigma)
        f = fidelity(rho, sigma)
        assert d_tr >= 1.0 - f - 1e-9
        assert d_tr <= np.sqrt(max(1.0 - f * f, 0.0)) + 1e-9


def test_contractive_under_dephasing_cptp_map():
    rng = np.random.default_rng(3)
    gamma = 0.5
    dt = 0.4
    import scipy.linalg as sla

    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_Z], [gamma])
    phi = sla.expm(dt * L)
    for _ in range(10):
        rho = _random_density(rng)
        sigma = _random_density(rng)
        before = trace_distance(rho, sigma)
        rho_t = unvec(phi @ vec(rho), d=2)
        sigma_t = unvec(phi @ vec(sigma), d=2)
        after = trace_distance(rho_t, sigma_t)
        assert after <= before + 1e-9  # data-processing inequality


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal shapes"):
        trace_distance(np.eye(2, dtype=complex), np.eye(3, dtype=complex))


def test_relaxation_layer_trace_distance_curve_decays_to_zero():
    # The trajectory distance to rho_ss must contract to ~0 (it is a CPTP
    # relaxation) and be present as an additive curve on the result.
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.array([[0, 1], [0, 0]], dtype=complex)],  # |0><1|, amplitude damping
        [1.0],
    )
    report = diagnose(L, bootstrap_B=50)
    curve = report.relaxation.trace_distance_curve
    assert curve is not None
    assert curve.shape == report.relaxation.fidelity_curve.shape
    assert np.all(curve >= -1e-12)
    assert curve[-1] < curve[0]
    assert curve[-1] == pytest.approx(0.0, abs=5e-2)


@qutip_required
def test_trace_distance_matches_qutip_tracedist():
    import qutip

    rng = np.random.default_rng(99)
    for _ in range(8):
        rho = _random_density(rng)
        sigma = _random_density(rng)
        ours = trace_distance(rho, sigma)
        theirs = float(qutip.tracedist(qutip.Qobj(rho), qutip.Qobj(sigma)))
        assert ours == pytest.approx(theirs, abs=1e-9)


def _random_density(rng: np.random.Generator, d: int = 2) -> np.ndarray:
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    m = a @ a.conj().T
    return m / np.trace(m)
