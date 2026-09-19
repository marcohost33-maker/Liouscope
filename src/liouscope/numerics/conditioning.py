"""Eigenvalue-conditioning evidence for the stationary mode (issue #117).

AUDIT ONLY. Nothing in this module changes a reported diagnostic, a filter
threshold or a verdict. It measures *why* the stationary eigenvalue sits where
the eigensolver put it, so a reader of a run report can tell a failed eigensolve
apart from a correctly located eigenvalue of an operator that is only
approximately trace preserving.

Why a second instrument is needed
---------------------------------
The structural certificate in :mod:`.linalg` measures the BACKWARD error: the
trace-preservation defect ``||q^H L||`` with ``q = vec(I)/sqrt(d)`` and the
eigensolver residuals. For a NON-NORMAL operator a backward error does not bound
the forward eigenvalue displacement by the same number. Measured on the 4x4
fixture from the 2026-09-11 external review of PR #127 (a leading block
``[[0, 1e-14], [1, 0]]`` rotated into a basis whose first vector is
``vec(I)/sqrt(2)``, remaining eigenvalues ``-1`` and ``-2``):

===========================================  ==========
trace-preservation defect ``||q^H L||``      1.415e-14
certificate band ``rtol * eps * ||L||_2``    4.441e-13
OBSERVED displacement ``min |lambda|``       1.000e-07
===========================================  ==========

The band under-predicts the displacement by 2.25e5x. The displacement is not a
solver failure: the exact eigenvalues of that matrix ARE ``+-1e-7``, and the
reason a 1e-14 perturbation moves a zero eigenvalue by 1e-7 is the conditioning
of that eigenvalue. First-order perturbation theory for a simple eigenvalue
``lambda`` of ``L`` with unit right/left eigenvectors ``x``, ``y``
(``y^H L = lambda y^H``) gives

    ``|lambda(L + E) - lambda| <= ||E||_2 / s(lambda) + O(||E||^2)``,
    ``s(lambda) = |y^H x|``  in ``(0, 1]``,

so ``s`` is the reciprocal eigenvalue condition number. The perturbation ``E``
in question is the minimum-norm correction that makes ``L`` trace preserving,
``E = -q (q^H L)`` with ``q`` a UNIT vector, so ``||E||_2 = ||q^H L||_2`` -- the
raw defect above divided by ``sqrt(d)``. On that fixture ``s = 2.000e-07`` and
``||E|| / s = 5.00e-08`` against the observed ``1.00e-07``, a factor of 2. That
is the quantity this module reports.

Why ``|y^H x|`` and not LAPACK ``RCONDE``
-----------------------------------------
Issue #117 proposed reading ``RCONDE`` out of LAPACK ``xGEEVX``. Two measured
reasons not to:

1. **It is not reachable from SciPy.** ``scipy.linalg.lapack`` exposes no
   ``?geevx`` wrapper (measured on SciPy 1.17.1: ``zgeevx`` raises
   ``AttributeError``; only ``?geev`` exists). Adding a Fortran dependency for a
   quantity that is two BLAS-1 calls away would be a large cost for no accuracy.
2. **It would answer the wrong question.** ``xGEEVX`` documents that its
   reciprocal condition numbers refer to the BALANCED matrix, and that diagonal
   scaling -- the part of balancing that is not a permutation -- changes them.
   ``?geev`` (what :func:`liouscope.numerics.linalg.eig_nonhermitian` uses)
   balances internally but back-transforms the eigenvectors, so ``|y^H x|``
   formed from what it returns is the conditioning of the operator AS THE CALLER
   WROTE IT. Measured on a 9x9 random complex matrix (d = 3, a superoperator
   side length) under the diagonal similarity ``D A D^-1`` with
   ``D = diag(1e-6 .. 1e9)`` -- which leaves the spectrum invariant to
   ``5.2e-15`` -- the per-mode ``s`` moves from ``0.189..0.761`` to
   ``1.55e-15..1.18e-14``, a factor of ~1.2e14. A certificate whose semantics
   are tied to the caller's operator must measure the caller's operator.

Degenerate and near-degenerate stationary manifolds
---------------------------------------------------
A per-mode ``s`` is meaningless for a repeated eigenvalue and MISLEADING for a
nearly repeated one: the individual eigenvectors are then ill-determined even
though the invariant subspace they span is not. Measured on
``[[0, 1, 0], [0, delta, 0], [0, 0, -1]]``, per-mode ``s`` of the two small
modes falls as ``delta`` (1e-2 -> 1e-14) while the two-dimensional subspace
stays perfectly conditioned:

=========  ==================  ===========================
``delta``  per-mode ``s``      ``sigma_min(Y^H X)``
=========  ==================  ===========================
1e-2       1.0e-02             1.000
1e-6       1.0e-06             1.000
1e-10      1.0e-10             1.000
1e-14      1.0e-14             1.000
=========  ==================  ===========================

A degenerate stationary manifold -- a conserved quantity or a symmetry sector --
is exactly this case and is PHYSICAL, so reporting per-mode ``s`` alone would
raise a false alarm on a healthy generator. The reported figure is therefore the
cluster quantity ``sigma_min(Y^H X)`` over orthonormal bases of the left and
right invariant subspaces of the whole zero set (Stewart-Sun), which reduces to
``|y^H x|`` for a simple eigenvalue. The per-mode minimum is kept alongside it
as evidence, never as the verdict.

What this instrument does NOT catch
-----------------------------------
The stiff deflation failure of issue #112 is invisible to conditioning. On the
canonical four-level network (``tests/test_spectral_certificate.py``) raw
``zgeev`` loses the zero mode -- it returns ``7.28e-06`` against a certificate
band of ``8.48e-08``, and ``dgeev-real`` repairs it to ``4.08e-17``. Conditioning
that WRONG spectrum gives ``s = 0.29`` for the stationary mode and ``0.040`` at
worst over the whole spurious spectrum, i.e. condition numbers between 1 and 25:
the solver reports a wrong answer as well conditioned, because the information
was destroyed by the Ahues-Tisseur deflation before any conditioning estimate
could see it. That range independently reproduces the 2026-08-25 measurement
already recorded in ``linalg.py``. Conditioning evidence is therefore ADDITIVE to
the structural certificate and no substitute for it -- which is also why it
changes no filter.

For the same reason the audit refuses to describe a spectrum it did not
reproduce: pass the accepted ``eigenvalues`` and a repaired spectrum yields
``available=False`` rather than the raw solve's conditioning (PR #166 review).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .._consts import CONDITIONING_AGREEMENT_FACTOR, ZERO_MODE_EPS_FACTOR
from .linalg import require_finite_square_2d, trace_preservation_defect
from .norms import scaled_euclidean_norm

#: Verdict strings. ``BENIGN`` and ``CONDITIONING_LIMITED`` are the two decided
#: outcomes; ``UNAVAILABLE`` is the absence of a measurement, never a claim.
CONDITIONING_BENIGN = "BENIGN"
CONDITIONING_LIMITED = "CONDITIONING_LIMITED"
CONDITIONING_UNAVAILABLE = "UNAVAILABLE"


@contextmanager
def _quiet_arithmetic() -> Iterator[None]:
    """Run the audit with float warnings demoted to non-finite VALUES.

    The suite runs under ``filterwarnings = ["error"]``, and so may a caller.
    An overflow inside a residual on a 1e200-scaled generator would then raise
    out of the audit and abort the analysis the audit exists to describe --
    which is the one failure mode this module may not have. Every quantity is
    checked for finiteness explicitly afterwards, so a suppressed warning
    becomes an ``inf``/``nan`` that the fail-closed branches already read, never
    a silently accepted number.
    """
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _unit(v: np.ndarray) -> np.ndarray | None:
    """Return ``v / ||v||`` without losing a subnormal or huge vector.

    Same discipline as :func:`liouscope.numerics.norms.scaled_euclidean_norm`:
    the division is carried out after a POWER-OF-TWO rescaling, so the mantissa
    survives exactly and a vector whose norm is subnormal is not divided by
    something that overflows inside NumPy's complex division. ``None`` denotes
    an exactly zero or non-finite vector, for which no direction exists.
    """
    v = np.asarray(v)
    nrm = scaled_euclidean_norm(v)
    if not np.isfinite(nrm) or nrm == 0.0:
        return None
    exp = int(np.frexp(nrm)[1])
    scaled = np.ldexp(np.real(v), -exp) + 1j * np.ldexp(np.imag(v), -exp)
    out = scaled / np.ldexp(nrm, -exp)
    if not np.all(np.isfinite(out)):
        return None
    return np.asarray(out, dtype=complex)


def eigenvalue_conditioning(
    right_vectors: np.ndarray, left_vectors: np.ndarray
) -> np.ndarray:
    """Per-mode reciprocal eigenvalue condition numbers ``s_j = |y_j^H x_j|``.

    Both matrices hold the eigenvectors COLUMN-WISE in the same order, as
    ``scipy.linalg.eig(..., left=True, right=True)`` returns them. The result is
    in ``[0, 1]``; ``0.0`` marks a mode whose left/right pair is orthogonal
    (defective) or whose vectors carry no usable direction, and is the
    fail-closed value -- it maximises every forward-error estimate built on it.
    """
    right = np.asarray(right_vectors)
    left = np.asarray(left_vectors)
    if right.shape != left.shape or right.ndim != 2:
        raise ValueError(
            "eigenvalue_conditioning expects two equally shaped 2-D vector "
            f"matrices, got {right.shape} and {left.shape}"
        )
    out = np.zeros(right.shape[1], dtype=float)
    for j in range(right.shape[1]):
        x = _unit(right[:, j])
        y = _unit(left[:, j])
        if x is None or y is None:
            continue
        s = float(abs(np.vdot(y, x)))
        # |y^H x| <= 1 by Cauchy-Schwarz on unit vectors; the clip only removes
        # a round-off overshoot, never a real value.
        out[j] = min(s, 1.0)
    return out


def _orthonormal_basis(vectors: np.ndarray) -> np.ndarray | None:
    """Orthonormal basis of ``span(columns)``, or ``None`` if they do not span.

    Each column is normalised first, so a cluster mixing a huge and a tiny
    eigenvector is not reduced to the huge one by the SVD. ``None`` means the
    numerical rank is below the column count -- the eigenvector set is
    defective, and there is no k-dimensional invariant subspace to condition.
    """
    cols = []
    for j in range(vectors.shape[1]):
        u = _unit(vectors[:, j])
        if u is None:
            return None
        cols.append(u)
    M = np.column_stack(cols)
    basis, sv, _ = np.linalg.svd(M, full_matrices=False)
    if sv.size == 0 or not np.isfinite(sv[0]) or sv[0] == 0.0:
        return None
    tol = max(M.shape) * float(np.finfo(float).eps) * float(sv[0])
    if int(np.count_nonzero(sv > tol)) < M.shape[1]:
        return None
    return np.asarray(basis[:, : M.shape[1]], dtype=complex)


def cluster_conditioning(
    right_vectors: np.ndarray, left_vectors: np.ndarray, indices: np.ndarray
) -> float:
    """Reciprocal condition number of the invariant subspace of a cluster.

    ``sigma_min(Y^H X)`` for orthonormal bases ``X``, ``Y`` of the right and
    left invariant subspaces spanned by ``indices``. For a single index this is
    exactly ``|y^H x|``, so a simple eigenvalue and a degenerate manifold are
    measured by one formula. Returns ``0.0`` -- the fail-closed value -- when
    either set of vectors is defective.
    """
    idx = np.asarray(indices, dtype=int).ravel()
    if idx.size == 0:
        return 0.0
    X = _orthonormal_basis(np.asarray(right_vectors)[:, idx])
    Y = _orthonormal_basis(np.asarray(left_vectors)[:, idx])
    if X is None or Y is None:
        return 0.0
    sv = np.linalg.svd(Y.conj().T @ X, compute_uv=False)
    if sv.size == 0 or not np.all(np.isfinite(sv)):
        return 0.0
    return float(min(float(sv.min()), 1.0))


@dataclass(frozen=True, slots=True)
class ZeroModeConditioning:
    """AUDIT-ONLY conditioning evidence for the stationary eigenvalue.

    Every field is a measurement or a derived estimate. None of them is read by
    a filter, a gap, a verdict or a tier -- see the module docstring for why the
    instrument is additive to the structural certificate rather than a
    replacement for it.
    """

    available: bool
    #: Why the measurement is absent. ``"ok"`` when it is present.
    reason: str
    #: The stationary eigenvalue the evidence refers to (smallest ``|lambda|``).
    eigenvalue: complex
    #: How many eigenvalues the zero set holds. ``> 1`` is a degenerate
    #: stationary manifold and is not by itself a defect.
    cluster_size: int
    #: ``sigma_min(Y^H X)`` over the zero set -- the reported conditioning.
    reciprocal_condition: float
    #: Smallest PER-MODE ``|y^H x|`` inside the zero set. Evidence only: it
    #: collapses on a benign near-degenerate manifold (module docstring).
    per_mode_reciprocal_condition: float
    #: Smallest per-mode ``|y^H x|`` over the WHOLE spectrum.
    spectrum_min_reciprocal_condition: float
    #: ``max ||L x - lambda x||`` over the zero set, unit ``x``.
    right_residual: float
    #: ``max ||L^H y - conj(lambda) y||`` over the zero set, unit ``y``.
    left_residual: float
    #: Distance from the zero set to the nearest eigenvalue outside it.
    separation: float
    #: ``||q^H L|| = ||vec(I)^H L|| / sqrt(d)`` with ``q`` the UNIT trace
    #: vector: the norm of the minimum-norm perturbation that would make
    #: the operator exactly trace preserving.
    trace_defect: float
    #: ``max(right_residual, left_residual) / s`` -- how far the SOLVER can have
    #: moved this eigenvalue. ``s`` here is the PER-MODE figure below, not the
    #: subspace one: this is a scalar eigenvalue's displacement.
    solver_forward_estimate: float
    #: ``trace_defect / s`` -- how far the exact zero of the nearest trace-
    #: preserving operator can be from where the eigenvalue was found. Same
    #: per-mode ``s`` as above.
    structural_forward_estimate: float
    #: ``min |lambda|`` actually observed.
    observed_displacement: float
    #: The cutoff the layer's filters apply, for comparison only.
    zero_tolerance: float
    #: ``max(structural, solver) forward estimate > zero_tolerance``: the
    #: conditioning-scaled displacement budget does not fit inside the cutoff
    #: the layer's filters apply, so the zero mode cannot be pinned that
    #: closely. Reading BOTH estimates is deliberate: an exactly trace-
    #: preserving generator has ``structural = 0`` and can still carry a
    #: defective stationary pair whose solver estimate is unbounded.
    conditioning_limited: bool
    #: Whether the observed displacement is consistent with the two estimates.
    #: ``None`` when an estimate is not finite, never ``False`` by default.
    displacement_explained: bool | None
    verdict: str

    @classmethod
    def unavailable(cls, reason: str) -> ZeroModeConditioning:
        """An absent measurement. Never a claim about the operator."""
        nan = float("nan")
        return cls(
            available=False,
            reason=reason,
            eigenvalue=complex(nan, nan),
            cluster_size=0,
            reciprocal_condition=nan,
            per_mode_reciprocal_condition=nan,
            spectrum_min_reciprocal_condition=nan,
            right_residual=nan,
            left_residual=nan,
            separation=nan,
            trace_defect=nan,
            solver_forward_estimate=nan,
            structural_forward_estimate=nan,
            observed_displacement=nan,
            zero_tolerance=nan,
            conditioning_limited=False,
            displacement_explained=None,
            verdict=CONDITIONING_UNAVAILABLE,
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view for the run report (RFC 8259: no NaN/inf)."""

        def _f(x: float) -> float | None:
            return float(x) if np.isfinite(x) else None

        return {
            "available": self.available,
            "reason": self.reason,
            "eigenvalue_real": _f(float(np.real(self.eigenvalue))),
            "eigenvalue_imag": _f(float(np.imag(self.eigenvalue))),
            "cluster_size": int(self.cluster_size),
            "reciprocal_condition": _f(self.reciprocal_condition),
            "per_mode_reciprocal_condition": _f(self.per_mode_reciprocal_condition),
            "spectrum_min_reciprocal_condition": _f(
                self.spectrum_min_reciprocal_condition
            ),
            "right_residual": _f(self.right_residual),
            "left_residual": _f(self.left_residual),
            "separation": _f(self.separation),
            "trace_defect": _f(self.trace_defect),
            "solver_forward_estimate": _f(self.solver_forward_estimate),
            "structural_forward_estimate": _f(self.structural_forward_estimate),
            "observed_displacement": _f(self.observed_displacement),
            "zero_tolerance": _f(self.zero_tolerance),
            "conditioning_limited": bool(self.conditioning_limited),
            "displacement_explained": self.displacement_explained,
            "verdict": self.verdict,
            "audit_only": True,
        }


def _divide_by_conditioning(numerator: float, s: float) -> float:
    """``numerator / s`` with the two degenerate cases named rather than raised.

    ``s == 0`` is a defective pair: the first-order estimate does not exist and
    the honest value is ``inf``, which every comparison below reads as "not
    resolvable". ``numerator == 0`` with ``s == 0`` is ``0/0``; the backward
    error being exactly zero is the stronger statement, so the estimate is
    ``0.0`` -- an exact eigenpair of an exactly trace-preserving operator needs
    no conditioning correction.
    """
    if not np.isfinite(numerator) or numerator < 0.0:
        return float("nan")
    if numerator == 0.0:
        return 0.0
    if not np.isfinite(s) or s <= 0.0:
        return float("inf")
    return float(numerator / s)


def _spectra_agree(audit: np.ndarray, accepted: np.ndarray) -> bool:
    """Whether the audit's own solve reproduced the spectrum being reported.

    PR #166 review, finding P2. The audit runs its own ``?geev`` to obtain the
    left/right pair it needs, but :func:`..linalg.certified_eigvals` may have
    REPAIRED the spectrum through a different LAPACK route. Measured on the
    canonical stiff #112 network: the certificate accepts ``dgeev-real`` with a
    stationary eigenvalue at ``4.08e-17``, while a raw ``zgeev`` on the same
    operator returns the spurious ``7.28e-06`` that issue #112 exists to reject.
    Conditioning evidence computed from the second describes a spectrum the
    report does not contain, which is worse than no evidence.

    The comparison is on the numbers, not on the solver name, so a future repair
    route that happens to reproduce the spectrum still yields usable evidence.
    Both are sorted lexicographically, which keeps a degenerate cluster
    together, and compared against the library's own "indistinguishable at
    double precision" band ``ZERO_MODE_EPS_FACTOR * eps * max|lambda|`` (issue
    #108) rather than a tolerance invented here. A false disagreement only
    withholds an audit; a false agreement publishes the wrong spectrum's
    conditioning, so the strict direction is the safe one.
    """
    if audit.size != accepted.size:
        return False
    if audit.size == 0:
        return True
    scale = max(float(np.abs(audit).max()), float(np.abs(accepted).max()))
    if not np.isfinite(scale):
        return False
    if scale == 0.0:
        return True
    band = ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * scale
    order_a = np.lexsort((np.imag(audit), np.real(audit)))
    order_b = np.lexsort((np.imag(accepted), np.real(accepted)))
    return bool(np.all(np.abs(audit[order_a] - accepted[order_b]) <= band))


def zero_mode_conditioning(
    L_super: np.ndarray,
    *,
    zero_tolerance: float | None = None,
    eigenvalues: np.ndarray | None = None,
) -> ZeroModeConditioning:
    """Measure the conditioning of the stationary mode of ``L_super``.

    AUDIT ONLY: the return value is evidence attached to a report, never a
    threshold. The function is total on the OPERATOR argument -- unusable input
    and a failed eigensolve come back as
    :meth:`ZeroModeConditioning.unavailable`, not as an exception, because an
    audit that can abort the analysis it audits is a liability rather than
    evidence. A malformed KEYWORD is a caller contract violation and does raise,
    as in :func:`eigenvalue_conditioning`.

    ``zero_tolerance`` is the cutoff the layer's own filters apply, supplied so
    the report can state whether the conditioning-scaled displacement fits
    inside it. It must be non-negative: a negative cutoff admits no eigenvalue
    to the zero set at all and then labels even an exact, well-conditioned zero
    mode ``CONDITIONING_LIMITED``, which is valid-looking nonsense. When it is
    omitted the comparison degrades to ``conditioning_limited=False`` and
    ``displacement_explained=None``; nothing downstream depends on either.

    ``eigenvalues`` is the spectrum the caller is actually going to report.
    Supply it whenever one exists: the audit then refuses rather than describe a
    spectrum the report does not contain (see :func:`_spectra_agree`).
    """
    if zero_tolerance is not None and not (
        np.isfinite(zero_tolerance) and zero_tolerance >= 0.0
    ):
        # Round-2 review. Rejecting a negative cutoff while letting NaN/inf
        # through was the same defect in a second guise: both were silently
        # demoted to "no cutoff supplied" and the run came back
        # ``BENIGN`` with a null tolerance, which is the valid-looking nonsense
        # the negative case is refused for. ``None`` remains the way to say
        # "no cutoff".
        raise ValueError(
            "zero_tolerance must be a finite non-negative number or None, got "
            f"{zero_tolerance!r}"
        )
    try:
        L = require_finite_square_2d(L_super, name="L_super")
    except (TypeError, ValueError):
        return ZeroModeConditioning.unavailable("input_not_finite_square")
    L = np.asarray(L, dtype=complex)
    # No ``n == 0`` branch: ``require_finite_square_2d`` already rejects a 0x0
    # input, so one here would be unreachable code implying a state that cannot
    # occur (measured: a 0x0 array comes back as "input_not_finite_square").
    n = L.shape[0]
    # PR #166 review, finding P2. A square side that is not ``d * d`` is not a
    # superoperator, so ``vec(I)^H L = 0`` is not a statement about it and
    # ``trace_preservation_defect`` returns NaN. Continuing published
    # ``available=True`` / ``verdict="BENIGN"`` on top of a structural estimate
    # that does not exist -- a benign-looking verdict resting on nothing.
    dim = int(round(np.sqrt(n)))
    if dim * dim != n:
        return ZeroModeConditioning.unavailable("dimension_not_a_superoperator")
    with _quiet_arithmetic():
        return _measure(L, zero_tolerance, eigenvalues, dim)


def _measure(
    L: np.ndarray,
    zero_tolerance: float | None,
    accepted: np.ndarray | None,
    dim: int,
) -> ZeroModeConditioning:
    """The measurement itself, always called inside :func:`_quiet_arithmetic`."""
    try:
        values, left, right = sla.eig(L, left=True, right=True)
    except (ValueError, np.linalg.LinAlgError, sla.LinAlgError):
        return ZeroModeConditioning.unavailable("eigensolve_failed")
    values = np.asarray(values)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return ZeroModeConditioning.unavailable("nonfinite_spectrum")
    if accepted is not None:
        accepted = np.asarray(accepted).ravel()
        if not np.all(np.isfinite(accepted)) or not _spectra_agree(values, accepted):
            return ZeroModeConditioning.unavailable("accepted_spectrum_mismatch")

    per_mode = eigenvalue_conditioning(right, left)
    magnitudes = np.abs(values)
    tol = (
        float(zero_tolerance)
        if zero_tolerance is not None and np.isfinite(zero_tolerance)
        else float("nan")
    )
    if np.isfinite(tol) and bool(np.any(magnitudes <= tol)):
        cluster = np.flatnonzero(magnitudes <= tol)
    else:
        # No supplied cutoff, or none of the spectrum falls inside it: the
        # CANDIDATE stationary mode is the smallest one. Reporting on it is the
        # point -- that is exactly the case in which a reader needs to know
        # whether its distance from zero is explained by conditioning.
        cluster = np.asarray([int(np.argmin(magnitudes))], dtype=int)

    s_cluster = cluster_conditioning(right, left, cluster)
    s_per_mode = float(per_mode[cluster].min())
    right_res = 0.0
    left_res = 0.0
    for j in cluster:
        x = _unit(right[:, j])
        y = _unit(left[:, j])
        if x is not None:
            right_res = max(right_res, scaled_euclidean_norm(L @ x - values[j] * x))
        if y is not None:
            left_res = max(
                left_res,
                scaled_euclidean_norm(L.conj().T @ y - np.conj(values[j]) * y),
            )
    outside = np.setdiff1d(np.arange(values.size), cluster, assume_unique=False)
    separation = (
        float(np.min(np.abs(values[outside][:, None] - values[cluster][None, :])))
        if outside.size
        else float("inf")
    )
    # PR #166 review, finding P2. ``trace_preservation_defect`` returns
    # ``||vec(I)^H L||`` with an UNNORMALISED ``vec(I)``, while the perturbation
    # this bound is about is the minimum-norm correction ``E = -q (q^H L)`` that
    # makes ``L`` trace preserving, with ``q = vec(I)/sqrt(d)`` a UNIT vector.
    # Since ``||E||_2 = ||q^H L||_2 = ||vec(I)^H L|| / sqrt(d)``, dividing the
    # raw defect by ``s`` inflated every structural estimate by ``sqrt(d)`` --
    # on the PR #127 fixture 7.07e-08 where the perturbation argument gives
    # 5.00e-08 -- and could flip the two report booleans near their thresholds.
    raw_defect, _scale = trace_preservation_defect(L)
    defect = raw_defect / np.sqrt(float(dim))

    stationary = int(cluster[int(np.argmin(magnitudes[cluster]))])
    observed = float(magnitudes[stationary])
    # Round-2 review. Both estimates are about the displacement of ONE scalar
    # eigenvalue, so they divide by a PER-MODE condition number, never by the
    # subspace figure ``s_cluster``. Those differ by arbitrarily much: for
    # ``[[0, 1], [0, delta]]`` the invariant subspace is perfectly conditioned
    # while each eigenvalue's own ``s`` is ``delta`` (measured 1e-8 at
    # ``delta = 1e-8``), so the subspace figure understated the sensitivity by
    # ``1/delta``. The minimum over the cluster is used rather than the selected
    # mode's own value: inside a near-degenerate cluster it is not determined
    # which member is "the" stationary one, and the smaller ``s`` gives the
    # larger estimate, which is the fail-closed direction. ``s_cluster`` remains
    # the REPORTED conditioning, because that is the honest figure for a
    # degenerate stationary manifold (see the module docstring).
    solver_estimate = _divide_by_conditioning(max(right_res, left_res), s_per_mode)
    structural_estimate = _divide_by_conditioning(float(defect), s_per_mode)

    # Round-2 review. A NaN estimate used to be dropped from the budget, so the
    # verdict was computed from whichever half survived: measured on the finite
    # 4x4 input ``diag(1.3e308, 0, -1, 1.3e308)``, whose trace-defect norm is not
    # representable, ``structural_forward_estimate`` serialised as null while a
    # zero solver residual left a budget of zero and the audit published
    # ``BENIGN``. A verdict resting on an estimate that does not exist is
    # exactly what this module is supposed to refuse, so it withholds instead.
    if np.isnan(structural_estimate) or np.isnan(solver_estimate):
        return ZeroModeConditioning.unavailable("forward_estimate_unavailable")
    budget_estimate = max(structural_estimate, solver_estimate)
    limited = bool(
        np.isfinite(tol)
        and (budget_estimate > tol or not np.isfinite(budget_estimate))
    )
    # PR #166 review, finding P2. ``tol`` used to enter this budget, which made
    # the answer tautological: cluster selection already guarantees
    # ``observed <= tol`` for every eigenvalue in the zero set, so the field
    # said "explained" on evidence it had not consulted. It now reads ONLY the
    # two forward estimates -- the quantities that actually claim to explain the
    # displacement -- and abstains when either is unavailable. Abstention covers
    # ``inf`` as well as ``nan``: an unbounded estimate would "explain" any
    # displacement whatsoever, and it serialises to null, so reporting ``True``
    # beside a missing number contradicts the field's own semantics.
    explained: bool | None
    if (
        not np.isfinite(observed)
        or not np.isfinite(structural_estimate)
        or not np.isfinite(solver_estimate)
    ):
        explained = None
    else:
        explained = bool(
            observed
            <= CONDITIONING_AGREEMENT_FACTOR
            * max(structural_estimate, solver_estimate)
        )
    return ZeroModeConditioning(
        available=True,
        reason="ok",
        eigenvalue=complex(values[stationary]),
        cluster_size=int(cluster.size),
        reciprocal_condition=s_cluster,
        per_mode_reciprocal_condition=s_per_mode,
        spectrum_min_reciprocal_condition=float(per_mode.min()),
        right_residual=float(right_res),
        left_residual=float(left_res),
        separation=separation,
        trace_defect=float(defect),
        solver_forward_estimate=solver_estimate,
        structural_forward_estimate=structural_estimate,
        observed_displacement=observed,
        zero_tolerance=tol,
        conditioning_limited=limited,
        displacement_explained=explained,
        verdict=CONDITIONING_LIMITED if limited else CONDITIONING_BENIGN,
    )
