"""Single source of truth for the Liouvillian builder.

The GKSL generator

    L[rho] = -i [H, rho] + sum_k gamma_k ( L_k rho L_k^dag
                                          - 1/2 { L_k^dag L_k, rho } )

is vectorised in column-stacking convention (Roth's identity):

    M_L = -i ( I (x) H - H.T (x) I )
        + sum_k gamma_k [ L_k.conj() (x) L_k
                        - 1/2 ( I (x) L_k^dag L_k )
                        - 1/2 ( (L_k^dag L_k).T (x) I ) ]

Anchor A: ``order='F'`` everywhere. We add a runtime guard that the keyword
is explicit so callers cannot silently flip it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..numerics.kronecker import unvec, vec

# Near-zero-trace threshold for the unit-2-norm null-vector candidate in
# steady_state(). The candidate's trace is dimensionless (bounded by sqrt(d)),
# so this is deliberately NOT tied to the rate-unit null-space tolerances.
_TRACE_TOL = 1.0e-9


class DegenerateSteadyStateError(ValueError):
    """Raised when the Liouvillian null space has dimension > 1.

    A multi-dimensional null space means the steady state is **not unique**:
    the GKSL generator has more than one stationary state (e.g. a
    decoherence-free subspace, a conserved quantity, or two non-communicating
    sectors). In that case any single ``rho_ss`` returned by an SVD/eig solve
    is an *arbitrary* point in the steady-state manifold, picked by numerical
    happenstance rather than physics. Returning it silently would be a
    correctness trap, so :func:`steady_state` fails closed by default.
    """

    def __init__(self, null_dim: int) -> None:
        self.null_dim = null_dim
        super().__init__(
            f"Liouvillian null space has dimension {null_dim} > 1: the steady "
            "state is not unique (degenerate NESS / decoherence-free subspace "
            "or conserved quantity). An SVD picks an arbitrary representative, "
            "which is physically meaningless. Pass allow_degenerate=True to "
            "obtain one (trace-normalised) representative with a RuntimeWarning."
        )


def build_liouvillian(
    H: np.ndarray,
    jump_ops: Sequence[np.ndarray] | None = None,
    rates: Sequence[float] | None = None,
    *,
    order: Literal["F"] = "F",
) -> np.ndarray:
    """Build the GKSL superoperator in column-stacking convention.

    Parameters
    ----------
    H
        Hermitian Hamiltonian of shape ``(d, d)``.
    jump_ops
        Sequence of Lindblad jump operators each of shape ``(d, d)``.
        May be empty for purely unitary dynamics.
    rates
        Optional sequence of non-negative rates with the same length as
        ``jump_ops``. Defaults to ones.
    order
        Must be ``"F"`` (column-stacking). Provided as a guard against
        accidental row-stacking calls (anchor A).

    Returns
    -------
    np.ndarray
        Complex ``(d^2, d^2)`` array.
    """
    if order != "F":
        raise ValueError(
            "build_liouvillian only supports column-stacking (order='F'); "
            "this guard exists because mixing column- and row-stacking silently "
            "garbles the physics (anchor A)."
        )
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got {H.shape}")
    # Explicit finiteness gate: NaN entries happen to fail the Hermiticity
    # comparison below, but an all-real +/-inf diagonal is "Hermitian" to
    # np.allclose and would propagate silently into the superoperator.
    if not np.all(np.isfinite(H)):
        raise ValueError("H contains non-finite entries")
    d = H.shape[0]
    # rtol=0 explicit: np.allclose's default rtol=1e-5 would let an H that is
    # non-Hermitian at ~1e-5*|entry| pass a gate the message advertises as
    # 1e-9 absolute (physical Hamiltonians are Hermitian to machine precision,
    # so a true 1e-9 gate rejects only genuinely malformed input).
    if not np.allclose(H, H.conj().T, rtol=0.0, atol=1.0e-9):
        raise ValueError("H must be Hermitian within 1e-9 atol")

    if jump_ops is None:
        jump_ops = []
    jump_ops = [np.asarray(L, dtype=complex) for L in jump_ops]
    for L in jump_ops:
        if L.shape != (d, d):
            raise ValueError(f"jump_op shape {L.shape} != ({d}, {d})")
        if not np.all(np.isfinite(L)):
            raise ValueError("jump_op contains non-finite entries")
    if rates is None:
        rates = [1.0] * len(jump_ops)
    rates = list(rates)
    if len(rates) != len(jump_ops):
        raise ValueError(
            f"len(rates)={len(rates)} != len(jump_ops)={len(jump_ops)}"
        )
    for g in rates:
        # ``NaN < 0`` is False, so the sign test alone would wave NaN (and
        # +inf) rates through into an all-NaN/inf Liouvillian.
        if not np.isfinite(g):
            raise ValueError(f"rate {g} must be finite")
        if g < 0:
            raise ValueError(f"rate {g} must be non-negative")

    eye = np.eye(d, dtype=complex)

    # Coherent part: -i ( I (x) H - H.T (x) I )
    L_super = -1j * (np.kron(eye, H) - np.kron(H.T, eye))

    # Dissipative part
    for gamma, L_op in zip(rates, jump_ops, strict=True):
        if gamma == 0.0:
            continue
        LdagL = L_op.conj().T @ L_op
        L_super += gamma * (
            np.kron(L_op.conj(), L_op)
            - 0.5 * np.kron(eye, LdagL)
            - 0.5 * np.kron(LdagL.T, eye)
        )
    return L_super


def steady_state(
    L_super: np.ndarray,
    *,
    rtol: float = 1.0e-9,
    atol: float = 0.0,
    allow_degenerate: bool = False,
) -> np.ndarray:
    """Return the steady state ``rho_ss`` with ``L rho_ss = 0`` and unit trace.

    Uses null-space extraction on the superoperator. If the SVD finds no
    singular value below tolerance (no null vector: the generator has no
    steady state at this tolerance), the smallest-singular-value direction is
    returned as a best-effort proxy together with a :class:`RuntimeWarning`
    carrying the residual ``||L rho|| = s_min`` -- the result is then NOT a
    verified steady state and must not be treated as one.

    Tolerance semantics (scale-relative, issue #97 item 5)
    ------------------------------------------------------
    The null-space tolerance is *relative to the largest singular value*:
    ``tol = max(atol, max(rtol, n2 * eps) * s[0])``. A Liouvillian has rate
    dimension, so its singular values scale linearly under a pure change of
    units ``L -> c L`` while the null space (and hence the steady-state
    diagnosis) is unit-independent; a relative tolerance is invariant under
    that rescaling. The previous absolute floor (``atol = 1e-9`` in arbitrary
    rate units) misdiagnosed small-scale *unique* systems as degenerate --
    e.g. ``1e-10 * L`` for amplitude damping, whose every singular value fell
    below the floor -- and would conversely have masked genuine near-
    degeneracy in large-scale systems.

    Parameters
    ----------
    L_super
        ``d^2 x d^2`` Liouvillian superoperator.
    rtol
        Relative null-space tolerance, applied as ``rtol * s[0]`` (unit-free;
        never below the SVD noise floor ``n2 * eps * s[0]``).
    atol
        Optional absolute floor in the caller's rate units, default ``0.0``
        (no absolute floor). Pass a positive value only when the physical
        rate scale is known and an absolute cutoff is genuinely intended.
    allow_degenerate
        Guard against a degenerate steady state (multi-dimensional null
        space). When ``False`` (default, fail-closed), a null space of
        dimension > 1 raises :class:`DegenerateSteadyStateError` because no
        single ``rho_ss`` is physically meaningful then. When ``True``, one
        arbitrary trace-normalised representative is returned together with a
        :class:`RuntimeWarning`.

    Raises
    ------
    DegenerateSteadyStateError
        If the null space has dimension > 1 and ``allow_degenerate`` is False.
    """
    # Cast up front: integer/bool input would crash np.finfo below, and the
    # SVD/normalisation math assumes an inexact dtype anyway.
    L_super = np.asarray(L_super)
    if not np.issubdtype(L_super.dtype, np.inexact):
        L_super = L_super.astype(complex)
    if L_super.ndim != 2 or L_super.shape[0] != L_super.shape[1]:
        raise ValueError(
            f"L superoperator must be a square 2-D array, got shape {L_super.shape}"
        )
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    if d < 1 or d * d != n2:
        raise ValueError(f"L superoperator must have square-d dimension, got {n2}")

    # Right null space of L: solve via SVD. The tolerance is relative to
    # s[0] so the diagnosis is invariant under a change of rate units
    # L -> c L (see docstring); atol is an opt-in absolute floor only.
    u, s, vh = np.linalg.svd(L_super)
    tol = max(atol, max(rtol, n2 * np.finfo(L_super.dtype).eps) * s[0])
    null_indices = np.where(s <= tol)[0]
    # Degeneracy guard: a null space of dimension > 1 means rho_ss is NOT
    # unique. Picking null_indices[0] would return an arbitrary point in the
    # steady-state manifold (anchor: S1 audit 2026-06-04).
    if null_indices.size > 1:
        if not allow_degenerate:
            raise DegenerateSteadyStateError(int(null_indices.size))
        warnings.warn(
            f"Liouvillian null space has dimension {null_indices.size} > 1: "
            "the steady state is not unique. Returning one arbitrary "
            "trace-normalised representative (allow_degenerate=True).",
            RuntimeWarning,
            stacklevel=2,
        )
    if null_indices.size == 0:
        # No singular value below tolerance: the generator has no verified
        # steady state, and the smallest-singular-value direction is only a
        # proxy with residual ||L rho|| = s[-1] > tol. Returning it SILENTLY
        # would fabricate a steady state (the fail-open mirror image of the
        # degeneracy guard above), so the caller is warned with the residual.
        warnings.warn(
            f"No Liouvillian null vector within tolerance {tol:.3e}: the "
            f"returned matrix is the smallest-singular-value direction with "
            f"residual ||L rho|| = {s[-1]:.3e} and is NOT a verified steady "
            "state.",
            RuntimeWarning,
            stacklevel=2,
        )
        rho_vec = vh.conj().T[:, -1]
    else:
        rho_vec = vh.conj().T[:, null_indices[0]]
    rho = unvec(rho_vec, d=d)
    # Hermitise and project to unit trace. The candidate vector has unit
    # 2-norm (SVD/eig convention), so its trace is dimensionless and bounded
    # by sqrt(d) -- the near-zero-trace test is therefore scale-free and must
    # NOT reuse the rate-unit ``atol`` (which now defaults to 0.0).
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < _TRACE_TOL:
        # Try flipping the global phase via the leading eigenvector
        eigvals, eigvecs = np.linalg.eig(L_super)
        idx = int(np.argmin(np.abs(eigvals)))
        rho = unvec(eigvecs[:, idx], d=d)
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) < _TRACE_TOL:
            raise RuntimeError("Cannot normalise steady state: trace too small")
    rho = rho / tr
    # Force Hermitian projection one more time
    rho_out: np.ndarray = 0.5 * (rho + rho.conj().T)
    return rho_out


__all__ = [
    "DegenerateSteadyStateError",
    "build_liouvillian",
    "steady_state",
    "unvec",
    "vec",
]
