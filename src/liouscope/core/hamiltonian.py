"""Benchmark Hamiltonians: Ising, XY, Heisenberg-XXZ, Bose-Hubbard.

All operators act on an ``n``-qubit Hilbert space ``(C^2)^{otimes n}``
unless otherwise noted (Bose-Hubbard uses truncated bosonic modes).
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .lattice import Lattice


@cache
def _pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return I2, sx, sy, sz


def single_site_operator(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Embed ``op`` (single-qubit) into a string of identities at position ``site``."""
    if not 0 <= site < n:
        raise ValueError(f"site {site} out of range for n={n}")
    I2 = _pauli()[0]
    result = np.array([[1.0 + 0j]])
    for i in range(n):
        result = np.kron(result, op if i == site else I2)
    return result


def two_site_operator(op_a: np.ndarray, op_b: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    """Embed ``op_a (x) op_b`` on sites ``(i, j)`` with ``i < j``."""
    if not (0 <= i < j < n):
        raise ValueError(f"need 0 <= i < j < n, got i={i}, j={j}, n={n}")
    I2 = _pauli()[0]
    result = np.array([[1.0 + 0j]])
    for k in range(n):
        if k == i:
            result = np.kron(result, op_a)
        elif k == j:
            result = np.kron(result, op_b)
        else:
            result = np.kron(result, I2)
    return result


def ising_hamiltonian(lat: Lattice, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    """Transverse-field Ising: ``H = -J sum Z Z - h sum X``."""
    _, sx, _, sz = _pauli()
    n = lat.n_sites
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in lat.pairs:
        H -= J * two_site_operator(sz, sz, i, j, n)
    for site in range(n):
        H -= h * single_site_operator(sx, site, n)
    return H


def xy_hamiltonian(lat: Lattice, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    """XY model: ``H = -J sum (XX + YY) - h sum Z``."""
    _, sx, sy, sz = _pauli()
    n = lat.n_sites
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in lat.pairs:
        H -= J * (
            two_site_operator(sx, sx, i, j, n) + two_site_operator(sy, sy, i, j, n)
        )
    for site in range(n):
        H -= h * single_site_operator(sz, site, n)
    return H


def heisenberg_xxz_hamiltonian(
    lat: Lattice,
    J: float = 1.0,
    Delta: float = 1.0,
    h: float = 0.0,
) -> np.ndarray:
    """Heisenberg-XXZ: ``H = -J sum (XX + YY + Delta ZZ) - h sum Z``."""
    _, sx, sy, sz = _pauli()
    n = lat.n_sites
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    for i, j in lat.pairs:
        H -= J * (
            two_site_operator(sx, sx, i, j, n)
            + two_site_operator(sy, sy, i, j, n)
            + Delta * two_site_operator(sz, sz, i, j, n)
        )
    if h != 0.0:
        for site in range(n):
            H -= h * single_site_operator(sz, site, n)
    return H


def bose_hubbard_hamiltonian(
    n_sites: int,
    n_max: int = 2,
    t: float = 1.0,
    U: float = 1.0,
) -> np.ndarray:
    """Truncated 1D Bose-Hubbard with local cutoff ``n_max`` bosons per site.

    Local dimension is ``n_max + 1``. The full Hilbert space has dimension
    ``(n_max + 1)^{n_sites}``; keep ``n_sites`` small.
    """
    if n_sites < 2:
        raise ValueError("Bose-Hubbard requires at least 2 sites")
    d_loc = n_max + 1
    a = np.diag(np.sqrt(np.arange(1, d_loc, dtype=complex)), k=1)
    a_dag = a.conj().T
    num = a_dag @ a
    eye_loc = np.eye(d_loc, dtype=complex)

    def embed(op: np.ndarray, site: int) -> np.ndarray:
        result = np.array([[1.0 + 0j]])
        for k in range(n_sites):
            result = np.kron(result, op if k == site else eye_loc)
        return result

    dim = d_loc**n_sites
    H = np.zeros((dim, dim), dtype=complex)
    for site in range(n_sites - 1):
        hop = embed(a_dag, site) @ embed(a, site + 1)
        H -= t * (hop + hop.conj().T)
    for site in range(n_sites):
        n_op = embed(num, site)
        H += 0.5 * U * (n_op @ (n_op - np.eye(dim, dtype=complex)))
    return H
