"""Five canonical validation systems V1-V5 (Spec Teil 15).

Each function returns ``(L, rho_initial, system_description)`` so they can
be plugged directly into :func:`liouscope.diagnose`.

V1 -- dissipative qutrit (6 transitions)
V2 -- pure-dephasing qubit
V3 -- amplitude-damped qubit (3 gamma regimes)
V4 -- thermal two-level (detailed balance)
V5 -- Jaynes-Cummings exceptional-point sweep
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.hamiltonian import _pauli
from .core.lindblad import build_liouvillian


@dataclass(frozen=True, slots=True)
class System:
    """Compact bundle for a validation system."""

    name: str
    L: np.ndarray
    rho_initial: np.ndarray
    rho_steady: np.ndarray | None = None
    description: str = ""


def _qutrit_jumps() -> list[np.ndarray]:
    """Six raising / lowering operators between three levels (0, 1, 2)."""
    sigmas: list[np.ndarray] = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        op = np.zeros((3, 3), dtype=complex)
        op[j, i] = 1.0
        sigmas.append(op)
        op_up = op.conj().T
        sigmas.append(op_up)
    return sigmas


def v1_qutrit() -> System:
    """Dissipative qutrit with six transitions (Spec V1)."""
    H = np.diag([0.0, 1.0, 2.5]).astype(complex)
    jumps = _qutrit_jumps()
    rates = [0.3, 0.05, 0.4, 0.07, 0.2, 0.04]
    L = build_liouvillian(H, jumps, rates)
    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[2, 2] = 1.0
    return System(
        name="V1-qutrit",
        L=L,
        rho_initial=rho0,
        description="Dissipative qutrit, six transitions (sec. 15 V1)",
    )


def v2_dephasing_qubit() -> System:
    """Pure-dephasing qubit; gap reliable (A1, F1)."""
    _, sx, _, sz = _pauli()
    H = 0.5 * sx
    L = build_liouvillian(H, [sz], [0.25])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    return System(
        name="V2-dephasing",
        L=L,
        rho_initial=rho0,
        description="Pure dephasing qubit; A1 / F1 regime",
    )


def v3_amplitude_damped_qubit(gamma: float = 0.5) -> System:
    """Amplitude-damped qubit (V3); rho_ss = |0><0|."""
    _, sx, sy, _ = _pauli()
    sm = 0.5 * (sx + 1j * sy)  # |0><1| in computational basis
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [sm], [gamma])
    # Excited initial state
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return System(
        name="V3-amplitude-damped",
        L=L,
        rho_initial=rho0,
        description=f"Amplitude-damped qubit, gamma={gamma}",
    )


def v4_thermal_two_level(beta: float = 1.0, omega: float = 1.0) -> System:
    """Thermal two-level detailed-balance system (V4)."""
    _, _, _, sz = _pauli()
    sx_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sx_plus = sx_minus.conj().T
    # sx_minus = [[0,1],[0,0]] is the lowering operator already.
    # sx_plus = [[0,0],[1,0]] raises.
    # ``nan <= 0.0`` is False, so the sign test alone would let a NaN beta or
    # omega through into silently-NaN thermal rates.
    if not np.isfinite(beta * omega) or beta * omega <= 0.0:
        raise ValueError(
            f"beta*omega must be finite and > 0 for a thermal occupation (got "
            f"beta={beta}, omega={omega}); the infinite-temperature limit "
            "beta=0 diverges."
        )
    n_th = 1.0 / (np.exp(beta * omega) - 1.0)
    g_down = (1.0 + n_th) * 0.5
    g_up = n_th * 0.5
    H = 0.5 * omega * sz
    L = build_liouvillian(H, [sx_minus, sx_plus], [g_down, g_up])
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    return System(
        name="V4-thermal",
        L=L,
        rho_initial=rho0,
        description=f"Thermal two-level, beta={beta}, omega={omega}",
    )


def v5_jaynes_cummings(g: float = 0.25, kappa: float = 1.0, n_max: int = 3) -> System:
    """Truncated Jaynes-Cummings model with cavity loss (V5).

    The exceptional point lives at ``g = kappa / 4`` in the rotating frame.
    """
    if n_max < 2:
        raise ValueError("Need n_max >= 2 for a meaningful JC truncation")
    d = 2 * (n_max + 1)
    # Cavity annihilation
    a = np.diag(np.sqrt(np.arange(1, n_max + 1, dtype=complex)), k=1)
    a_dag = a.conj().T
    # Qubit operators in cavity-then-qubit ordering: |n, e/g>
    sx_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sx_plus = sx_minus.conj().T
    eye_cav = np.eye(n_max + 1, dtype=complex)
    eye_qubit = np.eye(2, dtype=complex)

    a_full = np.kron(a, eye_qubit)
    a_dag_full = np.kron(a_dag, eye_qubit)
    sm_full = np.kron(eye_cav, sx_minus)
    sp_full = np.kron(eye_cav, sx_plus)

    H = g * (a_dag_full @ sm_full + a_full @ sp_full)
    L = build_liouvillian(H, [a_full], [kappa])
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[1, 1] = 1.0  # |0, e>
    return System(
        name="V5-JC",
        L=L,
        rho_initial=rho0,
        description=f"Jaynes-Cummings g={g}, kappa={kappa}, n_max={n_max}",
    )


def all_systems() -> list[System]:
    """Return the canonical V1-V5 bundle for benchmarking."""
    return [
        v1_qutrit(),
        v2_dephasing_qubit(),
        v3_amplitude_damped_qubit(),
        v4_thermal_two_level(),
        v5_jaynes_cummings(),
    ]


__all__ = [
    "System",
    "all_systems",
    "v1_qutrit",
    "v2_dephasing_qubit",
    "v3_amplitude_damped_qubit",
    "v4_thermal_two_level",
    "v5_jaynes_cummings",
]
