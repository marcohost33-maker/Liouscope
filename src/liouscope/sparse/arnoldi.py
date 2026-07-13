"""ARPACK shift-invert spectrum and steady state for large sparse Liouvillians."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..core.lindblad import DegenerateSteadyStateError
from ..numerics.kronecker import unvec


def sparse_steady_state(
    L_sparse: sp.spmatrix,
    *,
    tol: float = 1.0e-9,
    sigma_shift: complex | None = None,
    allow_degenerate: bool = False,
) -> np.ndarray:
    """Return the steady state via shift-invert ARPACK near zero.

    The smallest-magnitude eigenvalue corresponds to the steady state.
    ``sigma_shift`` is a tiny offset to keep the SuperLU factorisation of
    ``sigma*I - L`` non-singular when ``L`` has an exact zero eigenvalue.
    When ``None`` (default) it is chosen *relative to the spectral scale* of
    ``L`` as ``1e-8 * scale``, where ``scale = sqrt(||L||_1 ||L||_inf)`` is a
    cheap upper bound on the largest singular value.

    Degeneracy guard (parity with the dense :func:`~liouscope.core.lindblad.
    steady_state`, S1 audit 2026-06-04): the two smallest-magnitude
    eigenvalues are computed, and if BOTH lie within the guard threshold of
    zero the steady state is not unique -- an ARPACK vector would be an
    arbitrary (typically not even positive semi-definite) point in the
    steady-state manifold. That fails closed with
    :class:`DegenerateSteadyStateError` unless ``allow_degenerate=True``,
    which returns one trace-normalised representative together with a
    ``RuntimeWarning``.

    Tolerance semantics (scale-relative, issue #97 item 5): the guard
    threshold is ``max(tol * scale, |sigma_shift|)`` -- relative to the
    spectral scale of ``L`` so the diagnosis is invariant under a change of
    rate units ``L -> c L``, and never finer than the ARPACK convergence
    tolerance ``tol`` (itself relative), so eigenvalues cannot slip past the
    guard on numerical noise. The previous absolute threshold (rate units)
    misdiagnosed small-scale unique systems as degenerate.
    """
    L = sp.csc_matrix(L_sparse, dtype=complex)
    n2 = L.shape[0]
    d = int(round(np.sqrt(n2)))
    if L.shape[0] != L.shape[1] or d * d != n2:
        raise ValueError(
            f"L superoperator must be square with square-d dimension, got {L.shape}"
        )
    # Spectral scale: sqrt(||L||_1 ||L||_inf) >= s_max is a cheap, sparse-
    # friendly upper bound on the largest singular value (within sqrt(n2)).
    abs_L = abs(L)
    norm_1 = float(abs_L.sum(axis=0).max()) if L.nnz else 0.0
    norm_inf = float(abs_L.sum(axis=1).max()) if L.nnz else 0.0
    scale = float(np.sqrt(norm_1 * norm_inf))
    if sigma_shift is None:
        sigma_shift = (1.0e-8 * scale if scale > 0.0 else 1.0e-8) + 0.0j
    # k=2 so the guard can see a second null mode (ARPACK needs k < n2).
    k = 2 if n2 > 2 else 1
    vals, vecs = spla.eigs(L, k=k, sigma=sigma_shift, which="LM", tol=tol)
    order = np.argsort(np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    null_dim = int(np.sum(np.abs(vals) <= max(tol * scale, abs(sigma_shift))))
    if null_dim > 1:
        if not allow_degenerate:
            raise DegenerateSteadyStateError(null_dim)
        warnings.warn(
            f"Sparse Liouvillian null space has dimension >= {null_dim} > 1: "
            "the steady state is not unique. Returning one arbitrary "
            "trace-normalised representative (allow_degenerate=True).",
            RuntimeWarning,
            stacklevel=2,
        )
    rho = unvec(vecs[:, 0], d=d)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < tol:
        raise RuntimeError("Sparse steady state has near-zero trace")
    rho_normed: np.ndarray = rho / tr
    return rho_normed


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
