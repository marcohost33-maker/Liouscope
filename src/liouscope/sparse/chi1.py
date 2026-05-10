"""chi_1 lower bound used in the sparse-pipeline reliability gate.

For a non-Hermitian Liouvillian the standard Bauer-Fike eigenvalue bound has
constant ``chi_1 = ||V||_2 ||V^{-1}||_2`` (condition of the eigenvector matrix).
A small ``chi_1`` lower-bound certificate is required to trust ARPACK results
when the spectrum is highly non-normal.

We estimate ``chi_1`` via the leading right and left eigenvectors of a few
slowest modes (cheap proxy) plus the Petermann factor of the slowest mode.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def chi1_lower_bound(
    L_sparse: sp.spmatrix,
    k_modes: int = 4,
    *,
    sigma: complex = 1.0e-8 + 0.0j,
    tol: float = 1.0e-9,
) -> float:
    """Lower bound on ``chi_1`` from the ``k_modes`` slowest eigenmodes.

    Returns ``max_j K_j^{1/2}`` where ``K_j`` is the Petermann factor of
    mode ``j``. This is a strict lower bound on the eigenvector condition.
    """
    L = sp.csc_matrix(L_sparse, dtype=complex)
    # Right eigenvectors
    vals_r, vecs_r = spla.eigs(L, k=k_modes, sigma=sigma, which="LM", tol=tol)
    # Left eigenvectors via L^H
    vals_l, vecs_l = spla.eigs(L.conj().T, k=k_modes, sigma=sigma, which="LM", tol=tol)
    # Match by eigenvalue (left eigenvals are conjugates)
    chi = 0.0
    used = set()
    for j in range(vals_r.size):
        lam = vals_r[j]
        # Pick best left match
        best = -1
        best_dist = float("inf")
        for i in range(vals_l.size):
            if i in used:
                continue
            dist = float(abs(vals_l[i].conjugate() - lam))
            if dist < best_dist:
                best_dist = dist
                best = i
        if best < 0:
            continue
        used.add(best)
        r = vecs_r[:, j]
        l_vec = vecs_l[:, best]
        denom = abs(np.vdot(l_vec, r))
        if denom < 1.0e-30:
            K = float("inf")
        else:
            K = float(np.linalg.norm(r) * np.linalg.norm(l_vec) / denom)
        chi = max(chi, np.sqrt(K))
    return float(chi)
