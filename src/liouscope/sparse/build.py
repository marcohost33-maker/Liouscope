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


def _sparse_canonical_generator_scales(
    H_sp: sp.csr_matrix,
    jump_ops: Sequence[np.ndarray | sp.spmatrix] | None,
    rates: Sequence[float] | None,
    d: int,
) -> tuple[float, float] | None:
    """Sparse twin of :func:`liouscope.core.lindblad._canonical_generator_scales`.

    Same canonical Lindblad gauge on the sparse data arrays; ``None`` (keep
    the gauge-fixed scale of H alone) for an empty or malformed jump list and
    for a non-finite derived scale.
    """
    if jump_ops is None or len(jump_ops) == 0:
        return None
    try:
        ops = [sp.csr_matrix(L, dtype=complex) for L in jump_ops]
        gammas = [1.0] * len(ops) if rates is None else [float(g) for g in rates]
    except (TypeError, ValueError):
        return None
    if len(gammas) != len(ops):
        return None
    eye = sp.identity(d, dtype=complex, format="csr")
    H0 = sp.csr_matrix(H_sp, dtype=complex, copy=True)
    K = sp.csr_matrix((d, d), dtype=complex)
    with np.errstate(all="ignore"):
        for gamma, L_op in zip(gammas, ops, strict=True):
            if L_op.shape != (d, d):
                return None
            if L_op.nnz and not np.all(np.isfinite(L_op.data)):
                return None
            if not (np.isfinite(gamma) and gamma >= 0.0):
                return None
            if gamma == 0.0:
                continue
            m = complex(np.sum(L_op.diagonal() / d))
            L0 = (L_op - eye * m).tocsr()
            H0 = (H0 + (L_op.conj().T * m - L_op * np.conj(m)) * (gamma / 2j)).tocsr()
            K = (K + gamma * (L0.conj().T @ L0)).tocsr()
        shift0 = overflow_safe_mean_real(H0.diagonal())
        H0g = (H0 - eye * shift0).tocsr()
        coherent = float(np.max(np.abs(H0g.data))) if H0g.nnz else 0.0
        dissipation = 0.5 * float(np.max(np.abs(K.data))) if K.nnz else 0.0
    if not (np.isfinite(coherent) and np.isfinite(dissipation)):
        return None
    return coherent, dissipation


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
    # ROUND-19 REVIEW (external, PR #127), mirrored: overflow-safe shift.
    gauge_shift = overflow_safe_mean_real(H_sp.diagonal())
    eye_d = sp.identity(d, dtype=complex, format="csr")
    H_gauge = (H_sp - eye_d * gauge_shift).tocsr()
    scale = float(np.max(np.abs(H_gauge.data))) if H_gauge.nnz else 0.0
    # Non-finite scale or defect: refused, in parity with the dense builder
    # and with ``main`` (PR #127 round-2 review, P4).
    if not np.isfinite(scale) or not np.isfinite(defect):
        raise ValueError(
            "H is finite but its gauge-fixed Hermiticity scale is not "
            f"(max|H - H^dag| = {defect}, gauge-fixed max|H| = {scale}); "
            "the Hermiticity gate cannot be evaluated, so H is refused"
        )
    # Generator-relative tolerance in the canonical Lindblad gauge, in parity
    # with the dense builder; see core/lindblad.py for the derivation.
    canonical = _sparse_canonical_generator_scales(H_sp, jump_ops, rates, d)
    reference = scale
    coherent_scale = scale
    dissipation_scale = 0.0
    if canonical is not None:
        coherent_scale, dissipation_scale = canonical
        # OPEN QUESTION (E3, cross-family review requested): whether a large
        # PHYSICAL dissipation may excuse a Hermiticity defect of the coherent
        # part at all. The answer changes exactly this one expression -- e.g.
        # to ``coherent_scale`` alone -- and nothing else in the gate.
        reference = max(coherent_scale, dissipation_scale)
    # Written as ``not <=`` so that a NaN defect cannot be accepted either.
    if not defect <= EPS_HERMITICITY * reference:
        raise ValueError(
            f"H must be Hermitian within a relative {EPS_HERMITICITY:g} "
            f"of the generator scale (max|H - H^dag| = {defect:.3e}, "
            f"gauge-fixed max|H| = {scale:.3e}, canonical-gauge coherent "
            f"scale = {coherent_scale:.3e}, dissipation scale "
            f"max|sum gamma L0^dag L0|/2 = {dissipation_scale:.3e}, "
            f"relative defect = "
            f"{defect / reference if reference else float('inf'):.3e})"
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
