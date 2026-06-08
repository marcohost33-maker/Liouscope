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
        solved: np.ndarray = sla.solve(A, b, assume_a="gen")
        return solved
    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    sp_solved: np.ndarray = lu.solve(np.asarray(b, dtype=complex))
    return sp_solved


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
    x: np.ndarray = np.asarray(
        rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=complex
    )
    x /= np.linalg.norm(x)
    sigma = 0.0
    sigma_prev = 0.0
    # Konvergenz-Guard 200 Iter / 1e-12 (robustere Schranke aus PR#39 uebernommen,
    # Trefethen & Embree, Spectra and Pseudospectra) -> Best-of-both bei nahezu
    # degenerierten Singulaerwerten; bleibt weit unter der 1e-6-Testtoleranz.
    for _ in range(200):
        y = lu.solve(x)
        # Apply M^H = (A^{-1})^H = (A^H)^{-1} via the LU's conjugate-transpose
        # solve. NOTE: lu.solve(y.conj()).conj() computes conj(A)^{-1} y, which
        # equals (A^H)^{-1} y only for symmetric A and is wrong for the
        # non-normal Liouvillians this kernel targets -- use trans="H" instead.
        z_vec = lu.solve(y, trans="H")
        sigma = float(np.linalg.norm(z_vec))
        if sigma == 0.0:
            break
        x = z_vec / sigma
        if abs(sigma - sigma_prev) < 1.0e-12 * max(1.0, sigma):
            break
        sigma_prev = sigma
    return float(np.sqrt(sigma))
