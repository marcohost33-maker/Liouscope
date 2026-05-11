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

_DENSE_THRESHOLD = 256
_NORM_DENSE_THRESHOLD = 128


def resolvent_apply_superlu(L: np.ndarray, z: complex, b: np.ndarray) -> np.ndarray:
    """Solve ``(z I - L) x = b`` via SuperLU.

    Falls back to dense ``scipy.linalg.solve`` for small matrices using LU
    with partial pivoting (``zgetrf``).
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= _DENSE_THRESHOLD:
        return sla.solve(A, b, assume_a="gen")
    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    return lu.solve(np.asarray(b, dtype=complex))


def resolvent_norm(L: np.ndarray, z: complex) -> float:
    """Return ``|| (z I - L)^{-1} ||_2``.

    Dense path: explicit inverse plus top singular value (correct to machine
    precision). Sparse path: matrix-free ARPACK ``svds`` against a SuperLU-
    factorised ``A^{-1}`` linear operator.
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= _NORM_DENSE_THRESHOLD:
        Ainv = sla.solve(A, eye)
        return float(sla.svdvals(Ainv)[0])

    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)

    def matvec(v: np.ndarray) -> np.ndarray:
        return lu.solve(np.asarray(v, dtype=complex))

    def rmatvec(v: np.ndarray) -> np.ndarray:
        # (A^{-1})^H v == A^{-H} v == solve(A^H, v).
        # scipy's SuperLU supports trans='H' for the Hermitian transpose.
        return lu.solve(np.asarray(v, dtype=complex), trans="H")

    op = spla.LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=complex)
    # ARPACK svds is the standard way to grab the leading singular value of
    # a matrix-free operator. ``k=1`` is the minimum it supports.
    try:
        svs = spla.svds(op, k=1, which="LM", return_singular_vectors=False)
        return float(np.max(np.asarray(svs)))
    except (spla.ArpackError, spla.ArpackNoConvergence):
        # Fallback: deterministic power iteration on (A^{-H} A^{-1}).
        return _power_iteration_norm(matvec, rmatvec, n)


def _power_iteration_norm(matvec, rmatvec, n: int, max_iters: int = 100) -> float:
    """Power iteration on ``A^{-H} A^{-1}``; returns ``sqrt(top eigenvalue)``.

    Used as a fallback when ARPACK fails to converge. Deterministic seed.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)
    last = 0.0
    for _ in range(max_iters):
        y = matvec(x)
        w = rmatvec(y)
        sigma_sq = float(np.linalg.norm(w))
        if sigma_sq <= 0.0:
            return 0.0
        x = w / sigma_sq
        if abs(sigma_sq - last) <= 1.0e-9 * max(1.0, sigma_sq):
            last = sigma_sq
            break
        last = sigma_sq
    return float(np.sqrt(last))
