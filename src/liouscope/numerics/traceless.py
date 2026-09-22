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

Trace preservation is checked in BOTH readings before the restriction is
formed, exactly as the certified eigensolver paths in :mod:`.linalg` do. The
normwise ratio ``||q^H L|| / ||L||`` answers a question about the operator as
a whole and can be driven to round-off by a large entry that takes part in no
violated trace equation; the componentwise reading normalises every column's
trace equation by only those coefficients that can cancel within it. A gate
that reads only the first accepts generators whose traceless subspace is not
invariant, which is the one property this module exists to rely on.

After the restriction is formed, its own audit numbers are enforced as a
postcondition rather than only reported (issue #150): the invariance and
reconstruction defects must agree, to rounding, with the two identities the
reduction rests on. See :func:`_require_invariant_reduction`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._consts import ZERO_MODE_EPS_FACTOR
from .linalg import (
    trace_preservation_componentwise_error,
    trace_preservation_defect,
)
from .norms import scaled_euclidean_norm


@dataclass(frozen=True, slots=True)
class TracelessRestriction:
    """A trace-preserving generator represented on the traceless subspace."""

    operator: np.ndarray
    basis: np.ndarray
    trace_defect: float
    operator_scale: float
    trace_componentwise_error: float
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


def _require_invariant_reduction(
    *,
    invariance_defect: float,
    reconstruction_defect: float,
    projection_bound: float,
    arithmetic: float,
) -> None:
    """Enforce, to rounding, the two identities the reduction rests on.

    With ``q`` the unit trace vector and ``B`` an orthonormal basis of
    ``ker(q^H)``:

    1. ``||q^H L B|| <= ||q^H L||``. ``q^H L B`` holds the coordinates of the
       orthogonal projection of ``q^H L`` onto ``q^perp``, and a projection
       cannot exceed the vector it projects. The admission gates bound
       ``||q^H L||`` and therefore already bound the invariance defect, which
       is why no SEPARATE relative threshold is placed on it: that would be a
       second, slightly different gate on a quantity that is not free.
    2. ``L B - B (B^H L B) = (I - B B^H) L B = q (q^H L B)``, because
       ``I - B B^H = q q^H``. The reconstruction defect therefore EQUALS the
       invariance defect in exact arithmetic; it is not an independent
       structural quantity, and anything it adds beyond the invariance defect
       is either rounding or a subspace that is not invariant.

    ``projection_bound`` is ``||q^H L||`` and ``arithmetic`` the rounding
    either computed number may carry. Each comparison is written as
    ``not (x <= bound)`` so that a NaN or infinite reading -- including one
    propagated from a non-finite entry of the reduced operator -- refuses
    instead of comparing false.
    """
    if not (invariance_defect <= projection_bound + arithmetic):
        raise ValueError(
            "the traceless restriction violates its projection bound: "
            f"invariance_defect={invariance_defect:.6e} exceeds "
            f"||q^H L||={projection_bound:.6e} by more than the arithmetic "
            f"bound {arithmetic:.6e}"
        )
    if not (reconstruction_defect <= invariance_defect + arithmetic):
        raise ValueError(
            "the traceless subspace is not invariant under L_super within "
            f"rounding: reconstruction_defect={reconstruction_defect:.6e}, "
            f"invariance_defect={invariance_defect:.6e}, arithmetic bound "
            f"{arithmetic:.6e}"
        )


def restrict_to_traceless(
    L_super: np.ndarray,
    *,
    tp_rtol: float = 1.0e-10,
    reduction_rtol: float = ZERO_MODE_EPS_FACTOR,
) -> TracelessRestriction:
    """Restrict a trace-preserving ``d^2 x d^2`` generator to traceless space.

    The routine fails closed when trace preservation is not established, in
    both the normwise and the componentwise reading of ``q^H L = 0``. Those are
    the same two gates, in the same order and with the same comparisons, that
    the certified eigensolvers in :mod:`.linalg` apply, and ``tp_rtol`` bounds
    both. It does *not* decide how many stationary modes exist: for a unique
    stationary state the one non-traceless zero mode is removed exactly; for a
    degenerate stationary manifold the remaining traceless zero directions stay
    in the reduced operator and must be handled explicitly by the caller.

    ``invariance_defect`` measures ``||q^H L B||_2`` and
    ``reconstruction_defect`` measures ``||L B - B (B^H L B)||_F``. They make
    the subspace reduction auditable rather than assumed, and since issue #150
    they are also ENFORCED: the routine fails closed unless both agree with
    their exact values to within the floating-point arithmetic of forming them,
    ``reduction_rtol * (eps * ||L||_F + d**3 * 2**-1074)`` -- the standard
    rounding model with its gradual-underflow term, so the verdict does not
    change with the rate unit down to the subnormal range. The default
    ``reduction_rtol`` is :data:`liouscope._consts.ZERO_MODE_EPS_FACTOR`, the
    multiplier the certified eigensolver residuals already use, not a newly
    calibrated number. ``trace_componentwise_error`` reports the worst single
    trace equation, the reading the normwise ratio cannot make.
    """
    if not np.isfinite(tp_rtol) or tp_rtol < 0.0:
        raise ValueError(f"tp_rtol must be finite and non-negative, got {tp_rtol}")
    if not np.isfinite(reduction_rtol) or reduction_rtol < 0.0:
        raise ValueError(
            f"reduction_rtol must be finite and non-negative, got {reduction_rtol}"
        )

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

    # REVIEW (PR #139), finding B1. The normwise test above divides ONE number
    # by the norm of the whole operator, so an entry that participates in no
    # violated trace equation can dilute a violation elsewhere until the ratio
    # is round-off. Measured on the 4x4 input ``L[1, 0] = 1e300``,
    # ``L[0, 2] = 1``: global ratio 1e-300, hence accepted at the default
    # ``tp_rtol``, while the trace equation for column 2 has relative error 1.
    # The restriction returned for it carried ``invariance_defect = 0.707`` --
    # the subspace this module restricts to was not invariant, which is the
    # single assumption everything downstream rests on.
    #
    # Issue #130 already built the correct reading and both certified
    # eigensolver paths already gate on it (``linalg.py`` lines 1116-1117 and
    # 1392-1393). The SAME comparison is used here rather than a second,
    # slightly different one: the componentwise error is already dimensionless,
    # so it is compared DIRECTLY against ``tp_rtol`` with no scale factor, and
    # a non-finite reading is refused instead of being compared.
    tp_componentwise = trace_preservation_componentwise_error(L_c)
    # ``L_c`` is already validated finite, square and of perfect-square
    # dimension above, which are the only inputs for which the helper returns
    # NaN -- so this branch is unreachable today and is kept only so the gate
    # still fails closed if any of those validations is ever relaxed.
    if not np.isfinite(tp_componentwise):  # pragma: no cover
        raise ValueError(
            "componentwise trace-preservation evidence is not representable as "
            "finite float64"
        )
    if tp_componentwise > tp_rtol:
        raise ValueError(
            "L_super is not trace preserving componentwise within the requested "
            f"relative tolerance: componentwise_error={tp_componentwise:.6e}, "
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
            trace_componentwise_error=tp_componentwise,
            invariance_defect=0.0,
            reconstruction_defect=0.0,
        )

    image = L_c @ basis
    reduced = basis.conj().T @ image
    q = trace_vector(d)
    invariance_defect = scaled_euclidean_norm(q.conj() @ image)
    reconstruction_defect = scaled_euclidean_norm(image - basis @ reduced)

    # Issue #150. The rounding the three products above may commit, in the
    # standard model with gradual underflow: a relative part ``eps * ||L||_F``
    # and an absolute part of at most half a subnormal per product -- each
    # entry is an inner product over at most ``d`` non-zero basis
    # coefficients, so the Frobenius norm of that absolute error over the
    # ``n * (n - 1)`` entries stays below ``n * d * 2**-1075``, which
    # ``n * d * 2**-1074`` covers twice over. Measured across random GKSL
    # generators with ``d <= 16`` over rate scales 1e-300..1e300, both defects
    # stay below ``0.9 * eps * ||L||_F`` of their exact values, so the default
    # multiplier leaves three orders of magnitude of margin. The absolute part
    # is not decoration: a legal ``d = 2`` generator at scale 1e-318 carries a
    # reconstruction defect of two subnormals while ``eps * ||L||_F``
    # underflows to exactly zero, and the relative part alone refuses it.
    eps = float(np.finfo(float).eps)
    arithmetic = reduction_rtol * (eps * operator_scale + n * d * math.ulp(0.0))
    if not math.isfinite(arithmetic):
        # Same rule as ``operator_zero_tolerance``: a bound that is not finite
        # admits everything, so it is refused rather than compared against.
        raise ValueError(
            "the reduction tolerance derived from L_super is not finite "
            f"(reduction_rtol = {reduction_rtol}, ||L||_F = {operator_scale}); "
            "no reduction could fail it"
        )
    _require_invariant_reduction(
        invariance_defect=invariance_defect,
        reconstruction_defect=reconstruction_defect,
        projection_bound=trace_defect / math.sqrt(d),
        arithmetic=arithmetic,
    )

    return TracelessRestriction(
        operator=reduced,
        basis=basis,
        trace_defect=trace_defect,
        operator_scale=operator_scale,
        trace_componentwise_error=tp_componentwise,
        invariance_defect=invariance_defect,
        reconstruction_defect=reconstruction_defect,
    )
