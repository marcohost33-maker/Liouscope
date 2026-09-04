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

from .._consts import EPS_HERMITICITY
from ..numerics.linalg import overflow_safe_mean_real


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
    # Physics gates in parity with the dense builder (core/lindblad.py): the
    # sparse path previously accepted non-Hermitian H, negative/non-finite
    # rates and mis-shaped jump operators silently, producing a generator
    # that is not GKSL at all while the dense twin raised.
    if H_sp.nnz and not np.all(np.isfinite(H_sp.data)):
        raise ValueError("H contains non-finite entries")
    # Scale-relative Hermiticity gate, in parity with the dense builder
    # (issue #109). Computed on the sparse data arrays so the d^2 dense
    # materialisation the sparse path exists to avoid is not reintroduced.
    herm_defect = H_sp - H_sp.conj().T
    defect = float(np.max(np.abs(herm_defect.data))) if herm_defect.nnz else 0.0
    # Gauge-fixed scale, in parity with the dense builder (twelfth-round
    # review): a real identity offset is physically inert but inflates the
    # scale, loosening the gate. Subtracting the real trace part touches only
    # the diagonal, so sparsity is preserved.
    # ROUND-19 REVIEW (external, PR #127), the mirrored calculation the finding
    # names explicitly. Identical overflow, identical fail-open, and it has to
    # be repaired here too or the two builders disagree about whether the same
    # Hamiltonian is valid. See
    # :func:`liouscope.numerics.linalg.overflow_safe_mean_real`.
    gauge_shift = overflow_safe_mean_real(H_sp.diagonal())
    eye_d = sp.identity(d, dtype=complex, format="csr")
    H_gauge = (H_sp - eye_d * gauge_shift).tocsr()
    scale = float(np.max(np.abs(H_gauge.data))) if H_gauge.nnz else 0.0
    # ROUND-18 REVIEW (external, PR #127), the mirrored calculation the
    # finding names explicitly. Same machine-round-off allowance as the dense
    # builder in core/lindblad.py, and it has to be here too or the two
    # builders disagree about whether the same Hamiltonian is valid --
    # which is the failure mode "in parity with the dense builder" above
    # exists to prevent. See that function for the full derivation.
    roundoff_allowance = d * float(np.finfo(float).eps) * abs(gauge_shift)
    # Half-scale restatement in parity with the dense builder: the gauge-fixed
    # DIAGONAL can overflow even for a finite shift, which makes ``scale``
    # infinite and the relative test vacuous. See core/lindblad.py for the
    # measured case and for why halving keeps the predicate identical.
    if np.isfinite(scale):
        excessive = defect > EPS_HERMITICITY * scale + roundoff_allowance
    else:
        H_half = (H_sp * 0.5 - eye_d * (0.5 * gauge_shift)).tocsr()
        half_scale = float(np.max(np.abs(H_half.data))) if H_half.nnz else 0.0
        scale = 2.0 * half_scale
        excessive = (
            0.5 * defect > EPS_HERMITICITY * half_scale + 0.5 * roundoff_allowance
        )
    if excessive:
        raise ValueError(
            f"H must be Hermitian within a relative {EPS_HERMITICITY:g} "
            f"(max|H - H^dag| = {defect:.3e}, max|H| = {scale:.3e}, "
            f"machine-round-off allowance for the removed identity "
            f"component = {roundoff_allowance:.3e}, "
            f"relative defect = {defect / scale if scale else float('inf'):.3e})"
        )
    if jump_ops is None:
        jump_ops = []
    sparse_jumps = [sp.csr_matrix(L, dtype=complex) for L in jump_ops]
    for L_op in sparse_jumps:
        if L_op.shape != (d, d):
            raise ValueError(f"jump_op shape {L_op.shape} != ({d}, {d})")
        if L_op.nnz and not np.all(np.isfinite(L_op.data)):
            raise ValueError("jump_op contains non-finite entries")
    if rates is None:
        rates = [1.0] * len(sparse_jumps)
    rates = list(rates)
    if len(rates) != len(sparse_jumps):
        raise ValueError("len(rates) != len(jump_ops)")
    for g in rates:
        if not np.isfinite(g):
            raise ValueError(f"rate {g} must be finite")
        if g < 0:
            raise ValueError(f"rate {g} must be non-negative")

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
