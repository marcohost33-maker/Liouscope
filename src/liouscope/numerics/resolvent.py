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
    # Inverse power iteration on B = (A^{-1})^H A^{-1} (Hermitian PSD); its
    # dominant eigenvalue is sigma_max(A^{-1})^2 = ||(zI-L)^{-1}||_2^2 (the
    # standard pseudospectra/resolvent-norm estimator, Trefethen & Embree,
    # *Spectra and Pseudospectra*, 2005). The adjoint solve uses SuperLU's
    # ``trans="H"`` (A^H x = b) -- NOT ``lu.solve(y.conj()).conj()`` which
    # applies conj(A^{-1}) and is wrong for non-normal L (the case of interest).
    sigma = 0.0
    for _ in range(200):
        y = lu.solve(x)  # A^{-1} x
        z_vec = lu.solve(y, trans="H")  # (A^{-1})^H y  == B x
        sigma_new = float(np.linalg.norm(z_vec))
        if sigma_new == 0.0:
            break
        x = z_vec / sigma_new
        if abs(sigma_new - sigma) < 1.0e-12 * max(1.0, sigma_new):
            sigma = sigma_new
            break
        sigma = sigma_new
    return float(np.sqrt(sigma))
