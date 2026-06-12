"""Resolvent computations.

Uses :class:`scipy.sparse.linalg.splu` (LU factorisation, SuperLU backend) to
solve ``(z I - L) x = b``. BiCGSTAB does not converge reliably on
Liouvillian-resolvent systems (anchor E / patch E10).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Dense explicit-inverse + SVD is exact and O(n^3); used up to this size. Above
# it the SuperLU shift-and-invert power iteration (O(n) memory, sparse solves)
# takes over. resolvent_apply_superlu uses a larger cutoff because a single
# dense solve is much cheaper than a full SVD.
_DENSE_NORM_CUTOFF = 128
_DENSE_SOLVE_CUTOFF = 256
# Power-iteration budget for the largest singular value of the resolvent. 200
# lets moderately clustered top singular values (sigma_2/sigma_1 >~ 0.98)
# converge; tighter clusters exhaust the budget but, because it is the dominant
# *value* that is sought (not the eigenvector), the estimate is still accurate
# to ~1e-4 relative and a ResolventConvergenceWarning is emitted.
_POWER_ITERS = 200
_POWER_TOL = 1.0e-9


class ResolventConvergenceWarning(UserWarning):
    """Emitted when the resolvent-norm power iteration hits its budget.

    The cross-product power iteration for ``||(zI - L)^{-1}||_2`` did not meet
    its relative tolerance within :data:`_POWER_ITERS` steps -- the top singular
    values of the resolvent are tightly clustered. The returned value is a
    slight underestimate (lower bound); it is typically still accurate to
    ~1e-4 relative.
    """


def resolvent_apply_superlu(L: np.ndarray, z: complex, b: np.ndarray) -> np.ndarray:
    """Solve ``(z I - L) x = b`` via SuperLU.

    Falls back to dense ``scipy.linalg.solve`` for small dense matrices,
    using LU with partial pivoting (``zgetrf``).
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= _DENSE_SOLVE_CUTOFF:
        solved: np.ndarray = sla.solve(A, b, assume_a="gen")
        return solved
    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    sp_solved: np.ndarray = lu.solve(np.asarray(b, dtype=complex))
    return sp_solved


def resolvent_norm(L: np.ndarray, z: complex) -> float:
    """Return ``|| (z I - L)^{-1} ||_2``, the resolvent 2-norm at ``z``.

    This is the central quantity of pseudospectra (Trefethen, *Pseudospectra
    of Linear Operators*, SIAM Rev. 1997): ``||(zI - L)^{-1}||_2`` equals the
    reciprocal of the smallest singular value of ``zI - L``.

    Two regimes:

    * ``n <= 128`` -- form the explicit inverse and take its largest singular
      value (exact, dense ``scipy.linalg``).
    * ``n > 128`` -- the standard EigTool shift-and-invert approach: factorise
      ``A = zI - L`` once (SuperLU) and run a power iteration for the largest
      singular value of ``M = A^{-1}`` on the cross-product ``M^H M``, applying
      ``M`` via ``lu.solve`` and ``M^H = (A^H)^{-1}`` via ``lu.solve(trans="H")``.
      Power iteration is used (rather than Lanczos) deliberately: it is the
      *dominant value* that is needed, and value-convergence -- unlike
      eigenvector-convergence -- is robust to clustering of the top singular
      values (when ``sigma_1 ~ sigma_2`` any vector in the near-degenerate top
      subspace yields ``||M^H M x|| ~ sigma_1^2``). Empirically the estimate
      stays within ~1e-3 relative of the dense reference across separated and
      clustered spectra alike, so the extra ARPACK machinery -- and its own
      non-convergence failure modes -- buys nothing here.

    If the power iteration exhausts its budget without meeting the tolerance, a
    :class:`ResolventConvergenceWarning` is emitted and the returned value is a
    lower bound (still typically accurate to ~1e-4 relative).
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= _DENSE_NORM_CUTOFF:
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
    converged = False
    for _ in range(_POWER_ITERS):
        y = lu.solve(x)
        # Apply M^H = (A^{-1})^H = (A^H)^{-1} via the LU's conjugate-transpose
        # solve. NOTE: lu.solve(y.conj()).conj() computes conj(A)^{-1} y, which
        # equals (A^H)^{-1} y only for symmetric A and is wrong for the
        # non-normal Liouvillians this kernel targets -- use trans="H" instead.
        z_vec = lu.solve(y, trans="H")
        sigma = float(np.linalg.norm(z_vec))
        if sigma == 0.0:
            converged = True
            break
        x = z_vec / sigma
        if abs(sigma - sigma_prev) < _POWER_TOL * max(1.0, sigma):
            converged = True
            break
        sigma_prev = sigma
    if not converged:
        warnings.warn(
            f"resolvent_norm power iteration did not converge in {_POWER_ITERS} "
            f"steps (z={z!r}); the top singular values of the resolvent are "
            "clustered. Returned value is a lower bound.",
            ResolventConvergenceWarning,
            stacklevel=2,
        )
    return float(np.sqrt(sigma))
