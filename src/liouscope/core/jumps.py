"""Three canonical dissipator families: bulk, boundary, engineered."""

from __future__ import annotations

import numpy as np

from .hamiltonian import _pauli, single_site_operator


def bulk_dephasing_jumps(n_qubits: int) -> list[np.ndarray]:
    """Dephasing on every qubit: ``L_k = sigma_z_k`` for all sites."""
    _, _, _, sz = _pauli()
    return [single_site_operator(sz, site, n_qubits) for site in range(n_qubits)]


def bulk_amplitude_damping_jumps(n_qubits: int) -> list[np.ndarray]:
    """Amplitude damping on every qubit via ``sigma_-`` (|0><1|).

    Uses the convention where ``|0>`` is the computational-basis ground state
    and ``sigma_- = (sigma_x + i sigma_y) / 2 = |0><1|``.
    """
    _, sx, sy, _ = _pauli()
    sm = 0.5 * (sx + 1j * sy)
    return [single_site_operator(sm, site, n_qubits) for site in range(n_qubits)]


def boundary_dephasing_jumps(n_qubits: int) -> list[np.ndarray]:
    """Dephasing on the first and last qubit only."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    _, _, _, sz = _pauli()
    if n_qubits == 1:
        return [single_site_operator(sz, 0, 1)]
    return [
        single_site_operator(sz, 0, n_qubits),
        single_site_operator(sz, n_qubits - 1, n_qubits),
    ]


def engineered_target_jumps(target_psi: np.ndarray) -> list[np.ndarray]:
    """Jumps that drive the system towards ``target_psi``.

    Returns TWO rank-one operators: ``L = |psi><e_0|`` steering population
    into the target state and ``L_perp = |e_0><psi|`` draining it out of the
    reference basis state. ``target_psi`` is normalised internally; a zero
    vector raises :class:`ValueError`.
    """
    target: np.ndarray = np.asarray(target_psi).reshape(-1, 1).astype(complex)
    norm = float(np.linalg.norm(target))
    if norm <= 1.0e-12:
        raise ValueError("target_psi must be a non-zero state vector")
    if not np.isclose(norm, 1.0, atol=1.0e-9):
        target = np.asarray(target / norm, dtype=complex)
    d = target.size
    e0 = np.zeros((d, 1), dtype=complex)
    e0[0, 0] = 1.0
    L = target @ e0.conj().T
    L_perp = e0 @ target.conj().T
    return [L, L_perp]
