"""Structural restriction of trace-preserving generators to traceless operators.

For column-stacked operators let ``q = vec(I) / sqrt(d)``. Trace preservation is
exactly ``q^H L = 0``. Therefore ``ker(q^H)`` -- the traceless operator space --
is invariant under ``L``. Every eigenvector with non-zero eigenvalue lies in
that subspace, because ``0 = q^H L v = lambda q^H v``.

This gives a structural way to remove the unique stationary direction before an
eigensolve: choose an orthonormal basis ``B`` of ``ker(q^H)`` and solve the
restriction ``L0 = B^H L B``. No eigenvalue-magnitude threshold is involved in
constructing the subspace. If the stationary manifold is degenerate, its
additional traceless zero modes correctly remain in ``L0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .linalg import trace_preservation_defect
from .norms import scaled_euclidean_norm


@dataclass(frozen=True, slots=True)
class TracelessRestriction:
    """A trace-preserving generator represented on the traceless subspace."""

    operator: np.ndarray
    basis: np.ndarray
    trace_defect: float
    operator_scale: float
    invariance_defect: float
    reconstruction_defect: float


def trace_vector(d: int) -> np.ndarray:
    """Return the unit Hilbert-Schmidt trace vector ``vec(I)/sqrt(d)``."""
    if d < 1:
        raise ValueError(f"d must be positive, got {d}")
    q = np.zeros(d * d, dtype=complex)
    diag = np.arange(d, dtype=int)
    q[diag + diag * d] = 1.0 / np.sqrt(float(d))
    return q


def traceless_basis(d: int) -> np.ndarray:
    """Return a deterministic orthonormal basis for traceless ``d x d`` matrices.

    The first ``d(d-1)`` columns are off-diagonal matrix units. The final
    ``d-1`` columns are the standard orthonormal diagonal traceless generators

    ``diag(1,...,1,-k,0,...)/sqrt(k(k+1))``, ``k=1,...,d-1``.

    This explicit construction avoids asking a numerical rank routine to infer
    a subspace whose defining left-null vector is known analytically.
    """
    if d < 1:
        raise ValueError(f"d must be positive, got {d}")
    n = d * d
    if d == 1:
        return np.empty((1, 0), dtype=complex)

    basis = np.zeros((n, n - 1), dtype=complex)
    col = 0
    # Column-stacking index: matrix element (i, j) -> i + j*d.
    for j in range(d):
        for i in range(d):
            if i == j:
                continue
            basis[i + j * d, col] = 1.0
            col += 1

    for k in range(1, d):
        norm = np.sqrt(float(k * (k + 1)))
        for i in range(k):
            basis[i + i * d, col] = 1.0 / norm
        basis[k + k * d, col] = -float(k) / norm
        col += 1

    assert col == n - 1
    return basis


def restrict_to_traceless(
    L_super: np.ndarray,
    *,
    tp_rtol: float = 1.0e-10,
) -> TracelessRestriction:
    """Restrict a trace-preserving ``d^2 x d^2`` generator to traceless space.

    The routine fails closed when trace preservation is not established. It
    does *not* decide how many stationary modes exist: for a unique stationary
    state the one non-traceless zero mode is removed exactly; for a degenerate
    stationary manifold the remaining traceless zero directions stay in the
    reduced operator and must be handled explicitly by the caller.

    ``invariance_defect`` measures ``||q^H L B||_2`` and
    ``reconstruction_defect`` measures ``||L B - B (B^H L B)||_F``. Both should
    be at round-off for a legal trace-preserving input and make the subspace
    reduction auditable rather than assumed.
    """
    if not np.isfinite(tp_rtol) or tp_rtol < 0.0:
        raise ValueError(f"tp_rtol must be finite and non-negative, got {tp_rtol}")

    L_c = np.asarray(L_super, dtype=complex)
    if L_c.ndim != 2 or L_c.shape[0] != L_c.shape[1] or L_c.size == 0:
        raise ValueError(
            f"L_super must be a non-empty square 2-D array, got shape {L_c.shape}"
        )
    if not np.all(np.isfinite(L_c)):
        raise ValueError("L_super must contain only finite entries")

    n = int(L_c.shape[0])
    d = math.isqrt(n)
    if d * d != n:
        raise ValueError(
            f"L_super dimension must be a perfect square d^2, got {n}"
        )

    trace_defect, operator_scale = trace_preservation_defect(L_c)
    if not np.isfinite(trace_defect) or not np.isfinite(operator_scale):
        raise ValueError(
            "trace-preservation evidence is not representable as finite float64"
        )
    if trace_defect > tp_rtol * operator_scale:
        raise ValueError(
            "L_super is not trace preserving within the requested relative tolerance: "
            f"defect={trace_defect:.6e}, scale={operator_scale:.6e}, "
            f"tp_rtol={tp_rtol:.6e}"
        )

    basis = traceless_basis(d)
    if basis.shape[1] == 0:
        reduced = np.empty((0, 0), dtype=complex)
        return TracelessRestriction(
            operator=reduced,
            basis=basis,
            trace_defect=trace_defect,
            operator_scale=operator_scale,
            invariance_defect=0.0,
            reconstruction_defect=0.0,
        )

    image = L_c @ basis
    reduced = basis.conj().T @ image
    q = trace_vector(d)
    invariance_defect = scaled_euclidean_norm(q.conj() @ image)
    reconstruction_defect = scaled_euclidean_norm(image - basis @ reduced)

    return TracelessRestriction(
        operator=reduced,
        basis=basis,
        trace_defect=trace_defect,
        operator_scale=operator_scale,
        invariance_defect=invariance_defect,
        reconstruction_defect=reconstruction_defect,
    )
