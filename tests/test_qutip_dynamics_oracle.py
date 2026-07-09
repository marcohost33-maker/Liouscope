"""Independent-oracle cross-checks for the *dynamics* code path.

``tests/test_qutip_spectral_oracle.py`` pins the spectral diagnostics (gap,
oscillating gap, steady state) against analytic values and QuTiP. What it does
**not** cover is the time-domain propagation that the relaxation layer (D5-D7)
is built on: :func:`liouscope.diagnostics.relaxation._evolve` computes
``rho(t) = unvec(expm(L t) vec(rho_0))``, and every fitted relaxation rate
downstream inherits its correctness. A vectorisation-convention error, a
transposed Kronecker factor or a sign slip in ``build_liouvillian`` that
happens to preserve the *spectrum* would slip past the spectral oracles but
corrupt every trajectory.

This module closes that gap (007acc portfolio audit 2026-07-07: differential
coverage for "dynamics" and "trace preservation") with two oracle classes:

1. **Analytic / invariant** (primary, always runs): closed-form GKSL solutions
   for amplitude damping and pure dephasing, plus the CPTP invariants (unit
   trace, Hermiticity, positivity) that any physical evolution must satisfy at
   every time point.
2. **QuTiP** (secondary, ``@qutip_required``): the same trajectories re-derived
   by ``qutip.mesolve`` (an independent adaptive-ODE integrator, not a matrix
   exponential) and the full propagator re-derived by ``qutip.propagator``.

Tolerances: the analytic and propagator comparisons are exact-arithmetic
identities and use tight tolerances (1e-10). ``qutip.mesolve`` integrates an
ODE, so those comparisons run with tightened solver tolerances and assert at
1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import build_liouvillian
from liouscope.diagnostics.relaxation import _evolve

# --- canonical qubit operators ------------------------------------------------
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_LOWER = 0.5 * (_X + 1j * _Y)  # |0><1|, maps index 1 -> 0 (decay channel)

_T_GRID = np.linspace(0.0, 4.0, 9)

# Systems under test. The driven-dephasing case has no simple closed form for
# the full trajectory (drive and dephasing do not commute), which is exactly
# why it is valuable as a *differential* case against mesolve.
_GAMMA_AD = 0.4
_GAMMA_DEPH = 0.3
_OMEGA = 0.5


def _amplitude_damping() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    H = np.zeros((2, 2), dtype=complex)
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)  # excited state |1><1|
    return H, rho0, [_LOWER], [_GAMMA_AD]


def _driven_dephasing() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    H = 0.5 * _OMEGA * _X
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    return H, rho0, [_Z], [_GAMMA_DEPH]


def _trajectory(
    system: tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    H, rho0, jumps, rates = system
    L = build_liouvillian(H, jump_ops=jumps, rates=rates)
    return L, _evolve(L, rho0, _T_GRID)


# =============================================================================
# 1. Analytic oracles (always run)
# =============================================================================


def test_amplitude_damping_populations_match_closed_form():
    """T1 decay: rho_11(t) = e^{-gamma t}, rho_00(t) = 1 - e^{-gamma t}."""
    _, traj = _trajectory(_amplitude_damping())
    expected_excited = np.exp(-_GAMMA_AD * _T_GRID)
    np.testing.assert_allclose(traj[:, 1, 1].real, expected_excited, atol=1e-10)
    np.testing.assert_allclose(traj[:, 0, 0].real, 1.0 - expected_excited, atol=1e-10)
    # No coherence is ever generated from a diagonal initial state here.
    np.testing.assert_allclose(traj[:, 0, 1], 0.0, atol=1e-12)


def test_pure_dephasing_coherence_matches_closed_form():
    """Undriven dephasing (H = 0): rho_01(t) = rho_01(0) e^{-2 gamma t}.

    For L_z = sqrt(gamma) sigma_z the off-diagonal GKSL equation is
    d(rho_01)/dt = -2 gamma rho_01 while populations are frozen.
    """
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_Z], [_GAMMA_DEPH])
    traj = _evolve(L, rho0, _T_GRID)
    expected = 0.5 * np.exp(-2.0 * _GAMMA_DEPH * _T_GRID)
    np.testing.assert_allclose(traj[:, 0, 1], expected, atol=1e-10)
    np.testing.assert_allclose(traj[:, 0, 0].real, 0.5, atol=1e-10)


@pytest.mark.parametrize(
    "system", [_amplitude_damping(), _driven_dephasing()], ids=["amp-damp", "driven-deph"]
)
def test_evolution_preserves_cptp_invariants(system):
    """Every propagated state must stay a density matrix (TP + Hermitian + PSD)."""
    _, traj = _trajectory(system)
    for k in range(_T_GRID.size):
        rho_t = traj[k]
        assert abs(np.trace(rho_t) - 1.0) < 1e-10, f"trace violated at t={_T_GRID[k]}"
        np.testing.assert_allclose(rho_t, rho_t.conj().T, atol=1e-10)
        min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (rho_t + rho_t.conj().T))))
        assert min_eig > -1e-10, f"positivity violated at t={_T_GRID[k]}: {min_eig}"


# =============================================================================
# 2. QuTiP differential oracles
# =============================================================================


@qutip_required
@pytest.mark.parametrize(
    "system", [_amplitude_damping(), _driven_dephasing()], ids=["amp-damp", "driven-deph"]
)
def test_trajectory_matches_qutip_mesolve(system):
    """expm-propagated trajectory == independent adaptive-ODE integration."""
    import qutip

    H, rho0, jumps, rates = system
    _, traj = _trajectory(system)

    c_ops = [np.sqrt(g) * qutip.Qobj(j) for j, g in zip(jumps, rates)]
    result = qutip.mesolve(
        qutip.Qobj(H),
        qutip.Qobj(rho0),
        _T_GRID,
        c_ops=c_ops,
        options={"atol": 1e-12, "rtol": 1e-10},
    )
    for k, state in enumerate(result.states):
        np.testing.assert_allclose(
            traj[k],
            state.full(),
            atol=1e-6,
            err_msg=f"trajectory diverges from mesolve at t={_T_GRID[k]}",
        )


@qutip_required
@pytest.mark.parametrize(
    "system", [_amplitude_damping(), _driven_dephasing()], ids=["amp-damp", "driven-deph"]
)
def test_propagator_matches_qutip_expm(system):
    """expm(L t) == exp of QuTiP's independently built Liouvillian, per time point.

    ``qutip.propagator`` integrates an ODE (solver-limited to ~1e-7), so the
    exact comparison uses ``(qutip.liouvillian(...) * t).expm()`` instead: the
    generator is built by QuTiP from H and the collapse operators, so any
    vectorisation-convention or rate-normalisation mismatch in
    ``build_liouvillian`` shows up here at full precision.
    """
    import qutip
    import scipy.linalg as sla

    H, _, jumps, rates = system
    L = build_liouvillian(H, jump_ops=jumps, rates=rates)
    c_ops = [np.sqrt(g) * qutip.Qobj(j) for j, g in zip(jumps, rates)]
    L_qt = qutip.liouvillian(qutip.Qobj(H), c_ops)

    for t in _T_GRID[1:]:
        prop_ls = sla.expm(L * t)
        prop_qt = (L_qt * t).expm().full()
        np.testing.assert_allclose(
            prop_ls, prop_qt, atol=1e-10, err_msg=f"propagator mismatch at t={t}"
        )
