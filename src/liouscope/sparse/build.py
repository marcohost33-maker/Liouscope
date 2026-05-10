"""Sparse Liouvillian construction.

Uses ``scipy.sparse`` to assemble the column-stacked superoperator from
sparse jump operators, never materialising the dense ``d^2 x d^2`` array
unless the user explicitly requests it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import scipy.sparse as sp


def build_sparse_liouvillian(
    H: np.ndarray | sp.spmatrix,
    jump_ops: Sequence[np.ndarray | sp.spmatrix] | None = None,
    rates: Sequence[float] | None = None,
    *,
    order: Literal["F"] = "F",
) -> sp.csr_matrix:
    """Sparse GKSL superoperator in column-stacking convention.

    Mirrors :func:`liouscope.core.lindblad.build_liouvillian` but stores the
    result as ``scipy.sparse.csr_matrix``.
    """
    if order != "F":
        raise ValueError("build_sparse_liouvillian only supports order='F'")

    H_sp = sp.csr_matrix(H, dtype=complex)
    d = H_sp.shape[0]
    if H_sp.shape != (d, d):
        raise ValueError(f"H must be square, got {H_sp.shape}")
    if jump_ops is None:
        jump_ops = []
    sparse_jumps = [sp.csr_matrix(L, dtype=complex) for L in jump_ops]
    if rates is None:
        rates = [1.0] * len(sparse_jumps)
    rates = list(rates)
    if len(rates) != len(sparse_jumps):
        raise ValueError("len(rates) != len(jump_ops)")

    eye = sp.identity(d, dtype=complex, format="csr")
    # Coherent part
    L_super = -1j * (sp.kron(eye, H_sp, format="csr") - sp.kron(H_sp.T, eye, format="csr"))
    for gamma, L_op in zip(rates, sparse_jumps, strict=True):
        if gamma == 0.0:
            continue
        LdagL = (L_op.conj().T @ L_op).tocsr()
        L_super = L_super + gamma * (
            sp.kron(L_op.conj(), L_op, format="csr")
            - 0.5 * sp.kron(eye, LdagL, format="csr")
            - 0.5 * sp.kron(LdagL.T, eye, format="csr")
        )
    return L_super.tocsr()
