"""Resolvent computations.

Uses :class:`scipy.sparse.linalg.splu` (LU factorisation, SuperLU backend) to
solve ``(z I - L) x = b``. BiCGSTAB does not converge reliably on
Liouvillian-resolvent systems (anchor E / patch E10).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def resolvent_apply_superlu(L: np.ndarray, z: complex, b: np.ndarray) -> np.ndarray:
    """Solve ``(z I - L) x = b`` via SuperLU.

    Falls back to dense ``scipy.linalg.solve`` for small dense matrices,
    using LU with partial pivoting (``zgetrf``).
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= 256:
        return sla.solve(A, b, assume_a="gen")
    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    return lu.solve(np.asarray(b, dtype=complex))


def resolvent_norm(L: np.ndarray, z: complex) -> float:
    """Return ``|| (z I - L)^{-1} ||_2`` via SVD on the resolvent.

    For small matrices computes the explicit inverse and its 2-norm.
    For larger systems uses a power-iteration on the resolvent via SuperLU.
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= 128:
        Ainv = sla.solve(A, eye)
        return float(sla.svdvals(Ainv)[0])

    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)
    sigma_prev = 0.0
    for _ in range(80):
        y = lu.solve(x)
        # Apply (A^{-1})^H
        z_vec = lu.solve(y.conj()).conj()
        sigma = float(np.linalg.norm(z_vec))
        if sigma == 0.0:
            break
        x = z_vec / sigma
        if abs(sigma - sigma_prev) < 1.0e-9 * max(1.0, sigma):
            break
        sigma_prev = sigma
    return float(np.sqrt(sigma_prev))
