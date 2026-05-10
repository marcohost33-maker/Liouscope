"""ARPACK shift-invert spectrum and steady state for large sparse Liouvillians."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..numerics.kronecker import unvec


def sparse_steady_state(
    L_sparse: sp.spmatrix,
    *,
    tol: float = 1.0e-9,
    sigma_shift: complex = 1.0e-8 + 0.0j,
) -> np.ndarray:
    """Return the steady state via shift-invert ARPACK near zero.

    The smallest-magnitude eigenvalue corresponds to the steady state.
    ``sigma_shift`` is a tiny offset to keep the SuperLU factorisation of
    ``sigma*I - L`` non-singular when ``L`` has an exact zero eigenvalue.
    """
    L = sp.csc_matrix(L_sparse, dtype=complex)
    n2 = L.shape[0]
    d = int(round(np.sqrt(n2)))
    vals, vecs = spla.eigs(L, k=1, sigma=sigma_shift, which="LM", tol=tol)
    rho = unvec(vecs[:, 0], d=d)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < tol:
        raise RuntimeError("Sparse steady state has near-zero trace")
    return rho / tr


def _safe_shift_select(L_sparse: sp.spmatrix, sigma: complex) -> complex:
    """Avoid ARPACK shift collisions with exact eigenvalues.

    If ``sigma`` happens to lie within ``1e-10`` of an eigenvalue, ARPACK's
    LU factorisation becomes singular. We nudge by a tiny random offset.
    """
    try:
        # Test factorisation
        n = L_sparse.shape[0]
        eye = sp.identity(n, dtype=complex)
        spla.splu((sigma * eye - L_sparse).tocsc())
        return sigma
    except RuntimeError:
        rng = np.random.default_rng(0)
        return sigma + (1.0e-6) * (rng.standard_normal() + 1j * rng.standard_normal())


def sparse_spectrum(
    L_sparse: sp.spmatrix,
    k: int = 8,
    *,
    sigma: complex = 1.0e-8 + 0.0j,
    tol: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``k`` eigenvalues of ``L`` near ``sigma`` via shift-invert ARPACK.

    The default ``sigma=1e-8`` is a tiny offset from zero to keep the
    SuperLU factorisation of ``sigma I - L`` non-singular when ``L`` has
    an exact zero eigenvalue (the GKSL steady state).
    """
    L = sp.csc_matrix(L_sparse, dtype=complex)
    safe_sigma = _safe_shift_select(L, sigma)
    vals, vecs = spla.eigs(L, k=k, sigma=safe_sigma, which="LM", tol=tol)
    order = np.argsort(-np.real(vals))
    return vals[order], vecs[:, order]
