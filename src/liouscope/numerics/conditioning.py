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

=============================================  ==========
raw defect ``||vec(I)^H L||``                  1.415e-14
perturbation norm ``||q^H L||`` (``q`` unit)   1.000e-14
certificate band ``rtol * eps * ||L||_2``      4.441e-13
OBSERVED displacement ``min |lambda|``         1.000e-07
=============================================  ==========

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
   round-off (below ``1e-14``; the exact difference is itself a round-off
   quantity and differs between BLAS builds, so it is stated as a bound rather
   than quoted) -- the per-mode ``s`` moves from ``0.189..0.761`` to
   ``1.55e-15..1.18e-14``, a factor of ~1.2e14. A certificate whose semantics
   are tied to the caller's operator must measure the caller's operator.

Degenerate and near-degenerate stationary manifolds
---------------------------------------------------
For a simple eigenvalue the reciprocal condition is ``|y^H x|``. For a
cluster, the invariant quantity is the condition of its spectral projector.
On a semisimple cluster this agrees with the familiar left/right-invariant-
subspace formula, but deriving that formula from returned eigenvectors breaks
down at a defective repeated eigenvalue because the advertised eigenspace does
not exist.

For a multi-mode zero cluster, the headline ``reciprocal_condition`` therefore
comes from a reordered Schur form and the spectral projector, not from
eigenvector spans. A simple zero mode keeps the scalar ``|y^H x|`` condition.
The legacy ``cluster_conditioning`` primitive is retained for direct
semisimple-subspace work and regression comparison, but
``zero_mode_conditioning`` does not use it as its multi-mode headline
measurement. Scalar first-order estimates are reported only when the selected
zero cluster is simple.
Schur-projector conditioning for repeated/defective clusters (issue #168)
-----------------------------------------------------------------------------
A repeated stationary eigenvalue can be DEFECTIVE, in which case an eigenvector
decomposition of the advertised dimension does not exist and any conditioning
figure manufactured from those vectors can depend on the basis in which the
same operator is written. The audit therefore no longer uses eigenvectors for
its headline zero-cluster condition.

For a multi-mode zero cluster, the selected modes are reordered to the leading
block of a complex Schur form with LAPACK ``xTRSEN``. A simple mode keeps the
direct ``|y^H x|`` scalar condition; an independent Schur solve would add
round-off without improving that quantity. Crucially, the ``S`` returned by
``xTRSEN`` is
*not* reported as ``1 / ||P||_2``: LAPACK defines it from ``||R||_F`` as a
lower bound. After reordering, this module solves
``T11 R - R T22 = T12`` and computes the actual 2-norm expression
``1 / sqrt(1 + ||R||_2**2) = 1 / ||P||_2`` for the spectral projector. This
cluster quantity is invariant under unitary re-expression and remains meaningful
for a defective Jordan block so long as the cluster is separated from the
complement.

That does NOT manufacture a scalar condition number for an individual root
inside a defective cluster. No proximity test can reliably decide whether a
numerically split multiple root is a genuine pair of simple eigenvalues or the
round-off image of a Jordan block. Therefore a multi-mode zero cluster returns
``verdict="CLUSTER_ONLY"``: the projector condition, residuals, separation and
trace defect remain audit evidence, while the scalar first-order forward-error
estimates are withheld (NaN / JSON null) and ``displacement_explained`` abstains.
A simple zero cluster keeps the scalar first-order path. Nothing downstream
reads this audit, so the change affects no D1-D24 value, filter, certificate or
classification tier.

No cutoff, no claim (PR #173 review)
------------------------------------
Which modes form the zero set is decided by the caller's ``zero_tolerance``.
Without one, a default call used to pick the group nearest to zero by
proximity -- and on a defective double root ``?geev`` splits the pair by
``~sqrt(eps) * ||L||``, so the same Jordan-block operator gave a two-mode
``CLUSTER_ONLY`` in one orthonormal basis and a one-mode ``BENIGN`` in another.
A default call therefore returns ``verdict="ABSTAIN"``: observations only
(smallest eigenvalue, its magnitude, trace defect), every zero-set quantity
withheld, never ``BENIGN``. The spectral layer always supplies its certified
cutoff and is unaffected.

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
from scipy.optimize import linear_sum_assignment

from .._consts import CONDITIONING_AGREEMENT_FACTOR, ZERO_MODE_EPS_FACTOR
from .linalg import require_finite_square_2d, trace_preservation_defect
from .norms import scaled_euclidean_norm

#: Verdict strings. ``BENIGN`` and ``CONDITIONING_LIMITED`` are scalar-mode
#: outcomes and are only ever issued against a caller-supplied
#: ``zero_tolerance``. ``CLUSTER_ONLY`` means a valid Schur-projector cluster
#: measurement exists but scalar first-order attribution is intentionally
#: withheld. ``ABSTAIN`` means no cutoff was supplied, so no zero set is defined
#: and the audit makes no conditioning claim at all (PR #173 review, see
#: :func:`zero_mode_conditioning`); ``UNAVAILABLE`` means no trustworthy
#: measurement exists.
CONDITIONING_BENIGN = "BENIGN"
CONDITIONING_LIMITED = "CONDITIONING_LIMITED"
CONDITIONING_CLUSTER_ONLY = "CLUSTER_ONLY"
CONDITIONING_ABSTAIN = "ABSTAIN"
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


def _schur_cluster_projector_condition(
    L: np.ndarray,
    *,
    reference_eigenvalues: np.ndarray,
    cluster_indices: np.ndarray,
) -> float | None:
    """Reciprocal 2-norm condition of the selected Schur spectral projector.

    The selected eigenvalue cluster is reordered to the leading Schur block
    with LAPACK ``xTRSEN``. LAPACK's reported ``S`` is deliberately not
    used: ``xTRSEN`` documents it as the Frobenius-norm lower bound
    ``(1 + ||R||_F**2)**(-1/2)`` on the true reciprocal projector 2-norm.

    After reordering we solve ``T11 R - R T22 = T12`` and return
    ``1 / sqrt(1 + ||R||_2**2)``. This equals ``1 / ||P||_2`` for
    ``P = [[I, R], [0, 0]]`` in Schur coordinates. It is invariant under
    unitary re-expression and remains defined for defective eigenvalues when
    the selected cluster is separated from its complement.

    ``reference_eigenvalues`` is the spectrum the report actually carries,
    aligned to the audit eigendecomposition. The Schur spectrum must first
    agree with it mode-for-mode inside the library's existing round-off band.
    ``cluster_indices`` then selects exactly the report's cluster through that
    pairing. A disagreement fails closed instead of conditioning a numerically
    different Schur cluster.
    """
    try:
        T, Q = sla.schur(L, output="complex", check_finite=False)
    except (ValueError, np.linalg.LinAlgError, sla.LinAlgError):
        return None
    w = np.asarray(np.diag(T), dtype=complex)
    if w.size == 0 or not np.all(np.isfinite(w)):
        return None

    reference = np.asarray(reference_eigenvalues, dtype=complex).ravel()
    idx = np.asarray(cluster_indices, dtype=np.intp).ravel()
    if reference.size != w.size or idx.size == 0:
        return None
    if not np.all(np.isfinite(reference)):
        return None
    if np.any(idx < 0) or np.any(idx >= reference.size):
        return None
    matched = _match_spectra(w, reference, _agreement_band(w, reference))
    if matched is None:
        return None
    perm, _worst = matched
    selected = np.isin(perm, idx)

    m = int(np.count_nonzero(selected))
    if m != int(idx.size) or m <= 0:
        return None
    n = int(L.shape[0])
    if m == n:
        return 1.0

    try:
        trsen = sla.get_lapack_funcs("trsen", (T,))
        T_ord, _Q_ord, _w_ord, m_out, _s, _sep, info = trsen(
            np.asarray(selected, dtype=np.int32),
            T,
            Q,
            job="N",
            wantq=0,
            lwork=1,
            overwrite_t=0,
            overwrite_q=0,
        )
    except (ValueError, TypeError, np.linalg.LinAlgError, sla.LinAlgError):
        return None
    if int(info) != 0 or int(m_out) != m:
        return None

    T11 = np.asarray(T_ord[:m, :m], dtype=complex)
    T12 = np.asarray(T_ord[:m, m:], dtype=complex)
    T22 = np.asarray(T_ord[m:, m:], dtype=complex)
    try:
        R = sla.solve_sylvester(T11, -T22, T12)
        if not np.all(np.isfinite(R)):
            return 0.0
        if R.size == 0:
            return 1.0
        sv = sla.svdvals(R)
    except (ValueError, np.linalg.LinAlgError, sla.LinAlgError):
        return None
    if sv.size == 0:
        return 1.0
    r_norm_2 = float(sv[0])
    if not np.isfinite(r_norm_2):
        return 0.0
    return float(1.0 / np.hypot(1.0, r_norm_2))


@dataclass(frozen=True, slots=True)
class ZeroModeConditioning:
    """AUDIT-ONLY conditioning evidence for the stationary eigenvalue.

    Every field is a measurement or a derived estimate. None of them is read by
    a filter, a gap, a verdict or a tier -- see the module docstring for why the
    instrument is additive to the structural certificate rather than a
    replacement for it.
    """

    available: bool
    #: Measurement status. ``"ok"`` means scalar + cluster evidence is present;
    #: ``"cluster_conditioning_only"`` means only the Schur-projector cluster
    #: measurement is claimable; ``"zero_tolerance_not_supplied"`` (verdict
    #: ``ABSTAIN``) means the spectrum and the trace defect were measured but no
    #: zero set exists to condition; other strings explain an unavailable audit.
    reason: str
    #: The stationary eigenvalue the evidence refers to (smallest ``|lambda|``).
    eigenvalue: complex
    #: How many eigenvalues the zero set holds. ``> 1`` is a degenerate
    #: stationary manifold and is not by itself a defect. ``0`` for ``ABSTAIN``
    #: and ``UNAVAILABLE``: no zero set was selected.
    cluster_size: int
    #: Reciprocal 2-norm condition of the zero-cluster spectral projector,
    #: ``1 / ||P||_2``, obtained from a reordered Schur form and a Sylvester
    #: solve. For a simple eigenvalue this equals ``|y^H x|``.
    reciprocal_condition: float
    #: Reciprocal scalar eigenvalue condition used by the two first-order
    #: forward estimates. It equals the Schur-projector condition for a simple
    #: zero cluster and is NaN for a multi-mode cluster: a defective repeated
    #: eigenvalue has no basis-independent scalar first-order condition number,
    #: and proximity cannot safely distinguish it from a semisimple cluster.
    per_mode_reciprocal_condition: float
    #: Smallest per-mode ``|y^H x|`` over the whole spectrum on the scalar
    #: path. NaN for ``CLUSTER_ONLY`` because per-root eigenvector conditioning
    #: is precisely the quantity that is not claimable there.
    spectrum_min_reciprocal_condition: float
    #: ``max ||L x - lambda x||`` over the scalar unresolved group, or over the
    #: whole selected zero cluster in ``CLUSTER_ONLY`` mode.
    right_residual: float
    #: Left-eigenvector residual on the same scope as ``right_residual``.
    left_residual: float
    #: Distance from the scope being conditioned to its spectral complement:
    #: the scalar unresolved group on the simple path, or the whole selected
    #: zero cluster in ``CLUSTER_ONLY`` mode.
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
    #: The cutoff the layer's filters apply, for comparison only. NaN (JSON
    #: null) for ``ABSTAIN``, where none was supplied.
    zero_tolerance: float
    #: Scalar-path verdict bit: ``structural + solver forward estimate >
    #: zero_tolerance``. The two estimates are summed because they represent
    #: successive perturbations. It is always ``False`` for ``CLUSTER_ONLY``:
    #: that status is an abstention from the scalar comparison, not a benign
    #: scalar verdict.
    conditioning_limited: bool
    #: Whether the observed scalar displacement is consistent with the two
    #: first-order estimates. ``None`` for ``CLUSTER_ONLY``, for non-finite
    #: estimates, and outside the local/separated perturbative regime.
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

    @classmethod
    def abstained(
        cls, *, eigenvalue: complex, observed: float, trace_defect: float
    ) -> ZeroModeConditioning:
        """No cutoff was supplied: observations only, no conditioning claim.

        The three arguments are properties of the computed spectrum and of the
        operator, not of a zero set. Every field that depends on WHICH modes
        form the zero set -- the cluster, both condition figures, residuals,
        separation and the two forward estimates -- is NaN, because without a
        cutoff that set is not defined in a basis-independent way (PR #173
        review: a numerically split defective root otherwise decided it).
        """
        nan = float("nan")
        return cls(
            available=True,
            reason="zero_tolerance_not_supplied",
            eigenvalue=complex(eigenvalue),
            cluster_size=0,
            reciprocal_condition=nan,
            per_mode_reciprocal_condition=nan,
            spectrum_min_reciprocal_condition=nan,
            right_residual=nan,
            left_residual=nan,
            separation=nan,
            trace_defect=float(trace_defect),
            solver_forward_estimate=nan,
            structural_forward_estimate=nan,
            observed_displacement=float(observed),
            zero_tolerance=nan,
            conditioning_limited=False,
            displacement_explained=None,
            verdict=CONDITIONING_ABSTAIN,
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


def _scalar_verdict(budget: float, tol: float) -> str:
    """The scalar-path verdict, with the ACCEPTANCE bound to a true comparison.

    ``BENIGN`` is issued only when ``budget <= tol`` evaluates TRUE for a finite
    cutoff. IEEE 754 makes every ordered comparison with NaN False, so the
    orientation decides where a NaN lands: here an unbounded or NaN budget is
    ``CONDITIONING_LIMITED`` and a missing (non-finite) cutoff is ``ABSTAIN``
    -- neither can become ``BENIGN``. The previous form, ``limited = isfinite(
    tol) and budget > tol`` with ``BENIGN`` as the fall-through, returned
    ``BENIGN`` for every operator whenever the cutoff was NaN (PR #173 review).
    :func:`_measure` never reaches this with a non-finite cutoff or a NaN
    budget; the branches exist so that a future caller cannot either.
    """
    if not np.isfinite(tol):
        return CONDITIONING_ABSTAIN
    if bool(budget <= tol):
        return CONDITIONING_BENIGN
    return CONDITIONING_LIMITED


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


def _match_spectra(
    audit: np.ndarray, accepted: np.ndarray, band: float
) -> tuple[np.ndarray, float] | None:
    """Pair two spectra mode for mode, returning ``(perm, worst_distance)``.

    ``perm[i]`` is the index in ``accepted`` matched to ``audit[i]``.

    Round-4 review. This used to sort both spectra lexicographically by
    ``(real, imag)``, which does not pair the same modes when a round-off-sized
    real perturbation reorders them. Measured: the audit spectrum ``[i, -i]``
    against an accepted ``[i - 1e-14, -i + 1e-14]`` -- the same two modes, well
    inside the agreement band -- sorted into opposite orders, so the comparison
    paired ``i`` with ``-i``, measured a distance of 2, and threw away the
    conditioning evidence for a spectrum it should have accepted.

    A minimum-cost assignment on the distance matrix is permutation invariant by
    construction, so the pairing no longer depends on an ordering convention.
    ``scipy.optimize.linear_sum_assignment`` is the same primitive
    :mod:`.linalg` already uses to pair eigenvalues (``_injective_pairing``).

    ROUND-8 REVIEW. That assignment minimises the TOTAL distance while the
    caller rejects on the LARGEST one, and the two optima are different
    problems: with a band of 1.71 the sum-minimising pairing can have distances
    ``(0.52, 1.95, 0.48)``, total 2.95, while another permutation has
    ``(0.52, 1.50, 1.26)``, total 3.28 and every pair inside the band. The
    first is chosen and the spectrum rejected although the two agree mode for
    mode. The min-sum pairing is still preferred when it fits -- it is the best
    pairing available -- and only when it does not is feasibility decided
    separately, by minimising the NUMBER of over-band pairs: a perfect matching
    within the band exists exactly when that minimum is zero.
    """
    if audit.size != accepted.size:
        return None
    if audit.size == 0:
        return np.empty(0, dtype=int), 0.0
    cost = np.abs(audit[:, None] - accepted[None, :])
    # ROUND-9 REVIEW. Rejecting the whole matrix when ANY entry is non-finite
    # throws away matchings that exist exactly: two finite eigenvalues of
    # opposite sign near the top of the range overflow when subtracted, and that
    # entry need not be one the matching uses. A non-finite distance is simply
    # an over-band edge, which the feasibility pass below already handles; only
    # the edges actually chosen have to be finite, and they are, because they
    # are inside the band. (Defensive: I could not construct an input that
    # reaches it, because ``?geev`` garbles a spectrum of that range long before
    # the subtraction overflows -- see the PR thread.)
    usable = np.isfinite(cost)
    inside = usable & (cost <= band)
    if usable.all():
        rows, cols = linear_sum_assignment(cost)
        worst = float(cost[rows, cols].max())
    else:
        rows = cols = np.empty(0, dtype=int)
        worst = float("inf")
    if worst > band:
        # 0/1 costs, so the optimum counts the over-band pairs a perfect
        # matching cannot avoid. Zero means one exists entirely inside the band;
        # every pair in it then agrees to within round-off, so which of them the
        # solver returns cannot matter to membership.
        rows, cols = linear_sum_assignment((~inside).astype(float))
        if not inside[rows, cols].all():
            return None
        worst = float(cost[rows, cols].max())
    perm = np.empty(audit.size, dtype=int)
    perm[rows] = cols
    return perm, worst


def _agreement_band(audit: np.ndarray, accepted: np.ndarray) -> float:
    """The library's own "indistinguishable at double precision" band (#108)."""
    scale = max(float(np.abs(audit).max()), float(np.abs(accepted).max()))
    if not np.isfinite(scale) or scale == 0.0:
        return 0.0
    return ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * scale


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
    mode ``CONDITIONING_LIMITED``, which is valid-looking nonsense.

    When it is omitted the audit ABSTAINS: ``verdict="ABSTAIN"``,
    ``reason="zero_tolerance_not_supplied"``, ``cluster_size=0``, and every
    zero-set quantity (both condition figures, residuals, separation, forward
    estimates) is NaN / JSON null, ``conditioning_limited=False`` and
    ``displacement_explained=None``. Only the smallest-magnitude eigenvalue,
    its magnitude and the trace defect are reported, as observations. A
    default call can therefore never return ``BENIGN`` (PR #173 review), for
    two reasons:

    * ``BENIGN`` is the statement that the forward-error budget FITS INSIDE the
      cutoff. With no cutoff that comparison does not exist, and the previous
      default -- an internal NaN cutoff that fell through to ``BENIGN`` --
      published it for any operator, including one that is not trace
      preserving at all (``diag(0, -1, -1, -2)``, trace defect ``1.414``).
    * Without a cutoff the zero set had to be chosen by proximity, and
      proximity is not basis invariant on a defective root: the exact
      issue-#168 Jordan fixture gave ``CLUSTER_ONLY`` (two modes) in its plain
      basis and ``BENIGN`` (one mode, ``s = 3.6e-8``) in a unitary
      re-expression, because ``?geev`` splits the double root by
      ``~sqrt(eps) * ||L||`` -- far outside any round-off band. No proximity
      test separates that split from two genuine simple roots, so the default
      makes no claim that depends on it.

    Supply the cutoff the report actually filters with -- the spectral layer
    passes its certified zero-set tolerance -- to obtain a verdict. The round-8
    contract under which ``displacement_explained`` answered from a default
    call is withdrawn with it; nothing downstream depended on it.

    ``eigenvalues`` is the spectrum the caller is actually going to report.
    Supply it whenever one exists: the audit then refuses rather than describe a
    spectrum the report does not contain (see :func:`_spectra_agree`).
    """
    # ROUND-8 REVIEW. The check runs on the value AS NARROWED to the working
    # precision, not as supplied: ``np.longdouble("1e400")`` is finite in its
    # own dtype, passed this gate, then became ``inf`` in float64 and was
    # silently demoted to "no cutoff supplied" -- ``cluster_size=1`` and
    # ``BENIGN`` where a 1e400 cutoff admits the whole spectrum. That is the
    # same valid-looking nonsense the negative and non-finite cases are refused
    # for, reached through the dtype instead.
    if zero_tolerance is not None:
        try:
            zero_tolerance = float(zero_tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "zero_tolerance must be a finite non-negative number or None, got "
                f"{zero_tolerance!r}"
            ) from exc
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
        validated = require_finite_square_2d(L_super, name="L_super")
    except (TypeError, ValueError):
        return ZeroModeConditioning.unavailable("input_not_finite_square")
    # Round-3 review. The narrowing to complex128 used to happen HERE, outside
    # the guard. A dtype with a wider range than float64 -- ``np.longdouble``
    # holding 1e400, which is finite and passes validation -- makes the cast
    # emit ``RuntimeWarning: overflow encountered in cast``, and under the
    # suite's ``filterwarnings = ["error"]`` this function RAISED instead of
    # returning UNAVAILABLE, breaking the one contract it has. The cast is now
    # inside the guard and its result is checked, so a value the working
    # precision cannot hold is reported rather than thrown.
    with _quiet_arithmetic():
        L = np.asarray(validated, dtype=complex)
        if not bool(np.all(np.isfinite(L))):
            return ZeroModeConditioning.unavailable("not_representable_in_complex128")
        # No ``n == 0`` branch: ``require_finite_square_2d`` already rejects a
        # 0x0 input, so one here would be unreachable code implying a state
        # that cannot occur (measured: a 0x0 array comes back as
        # "input_not_finite_square").
        n = L.shape[0]
        # PR #166 review. A square side that is not ``d * d`` is not a
        # superoperator, so ``vec(I)^H L = 0`` is not a statement about it and
        # ``trace_preservation_defect`` returns NaN. Continuing published
        # ``available=True`` / ``verdict="BENIGN"`` on top of a structural
        # estimate that does not exist -- a verdict resting on nothing.
        dim = int(round(np.sqrt(n)))
        if dim * dim != n:
            return ZeroModeConditioning.unavailable("dimension_not_a_superoperator")
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
    # Round-4 review, two findings that share one mechanism. The audit's own
    # solve decides which modes it CAN condition, but the report filters the
    # ACCEPTED spectrum, so membership of the zero set has to be read off that
    # one. Measured: an audit mode at ``1e-6`` against an accepted ``1e-6 -
    # 1e-13`` at a cutoff of ``1e-6 - 5e-14`` agrees comfortably, yet the
    # accepted zero set holds two modes where the audit's holds one, and the
    # evidence reported ``cluster_size=1`` for a two-mode zero set. The
    # assignment below pairs the two spectra mode for mode and is then used for
    # BOTH the agreement test and the membership decision.
    magnitudes = np.abs(values)
    reported = values
    if accepted is not None:
        # ROUND-8 REVIEW. The accepted spectrum arrives in the CALLER's dtype.
        # A finite ``np.clongdouble`` spectrum built a ``float128`` cost matrix
        # and ``linear_sum_assignment`` raised ``TypeError: Cannot cast array
        # data from dtype('float128')`` -- an exception out of an audit whose
        # contract is that it never raises on the operator side. It is narrowed
        # to the working precision here, inside the protected block, and
        # rechecked: a value that only becomes non-finite in float64 cannot be
        # matched against a float64 spectrum either.
        accepted = np.asarray(accepted).ravel().astype(complex, copy=False)
        if not np.all(np.isfinite(accepted)):
            return ZeroModeConditioning.unavailable("accepted_spectrum_mismatch")
        matched = _match_spectra(values, accepted, _agreement_band(values, accepted))
        if matched is None:
            return ZeroModeConditioning.unavailable("accepted_spectrum_mismatch")
        # Aligned to audit index, so ``reported[j]`` is the eigenvalue the
        # report carries for the pair whose vectors sit at audit column ``j``.
        reported = accepted[matched[0]]
        magnitudes = np.abs(reported)

    per_mode = eigenvalue_conditioning(right, left)
    # ROUND-6 REVIEW. Modes are grouped by whether the arithmetic SEPARATES
    # them, using the library's own "indistinguishable at double precision"
    # band (#108) -- the same ``ZERO_MODE_EPS_FACTOR * eps * max|lambda|`` that
    # :func:`_agreement_band` already applies between two spectra, applied here
    # within one.
    #
    # The backward error (the residuals below) was the first candidate for this
    # and is WRONG, for a reason this module is required to care about: a
    # residual is not scale invariant in the way a conditioning figure must be.
    # Measured on the packaged two-level generator at ``c = 1e200``, where the
    # eigendecomposition itself degrades, the relative residual rises from
    # ``1e-16`` to ``1.0`` and a residual-scaled grouping merged the entire
    # spectrum -- moving the reported ``reciprocal_condition`` from ``0.8833``
    # to ``1.0`` for the same operator in different rate units, which is exactly
    # the invariance ``s(cL) = s(L)`` that #108 and #130 require of every
    # threshold here. The round-off band is scale covariant by construction: gap
    # and band scale together with ``c``.
    band = _agreement_band(reported, reported)

    def _unresolved_from(anchor: int, candidates: np.ndarray) -> np.ndarray:
        """Those candidates the arithmetic cannot separate from ``anchor``."""
        return np.asarray(
            [
                int(j)
                for j in candidates
                if abs(reported[int(j)] - reported[anchor]) <= band
            ],
            dtype=np.intp,
        )

    # Structural evidence is a prerequisite for any audit statement, including
    # an abstention. Check it before any independent Schur solve so an already
    # unrepresentable trace-preservation correction has one deterministic,
    # primary failure reason across LAPACK builds.
    raw_defect, _scale = trace_preservation_defect(L)
    defect = raw_defect / np.sqrt(float(dim))
    if not np.isfinite(defect):
        return ZeroModeConditioning.unavailable("forward_estimate_unavailable")

    # PR #173 REVIEW (Codex P2 on the former ``tol = ... else nan`` line, and an
    # independent audit finding on the same line). No cutoff used to become an
    # internal NaN cutoff; every ordered comparison with NaN is False, so the
    # scalar path could only ever answer ``BENIGN`` -- measured ``BENIGN`` for
    # ``diag(0, -1, -1, -2)``, which is not trace preserving (defect 1.414). And
    # the zero set was then chosen by proximity alone, which on a defective
    # double root is decided by LAPACK's ``sqrt(eps)`` split and therefore by
    # the basis: the issue-#168 pair gave ``CLUSTER_ONLY`` plain and ``BENIGN``
    # rotated. Without a cutoff the audit therefore abstains before any zero
    # set is chosen. See the docstring of :func:`zero_mode_conditioning`.
    if zero_tolerance is None:
        nearest = int(np.argmin(magnitudes))
        return ZeroModeConditioning.abstained(
            eigenvalue=complex(reported[nearest]),
            observed=float(magnitudes[nearest]),
            trace_defect=float(defect),
        )
    tol = float(zero_tolerance)
    cluster_from_cutoff = bool(np.any(magnitudes <= tol))
    if cluster_from_cutoff:
        cluster = np.flatnonzero(magnitudes <= tol)
    else:
        # Round-4 review. This used to take the single smallest mode, which
        # contradicted the advertised zero-SET conditioning on an exactly
        # degenerate stationary manifold: two-level pure dephasing has spectrum
        # ``[0, 0, -2, -2]`` and the default call reported ``cluster_size=1``,
        # conditioning one arbitrary vector of a two-dimensional stationary
        # subspace. Every mode the solve cannot separate from the smallest is
        # kept instead.
        #
        # Reached only with a FINITE cutoff: the no-cutoff case abstained above.
        #
        # ROUND-9 REVIEW asked for this branch to withhold when a FINITE cutoff
        # admits no eigenvalue, on the grounds that the evidence then describes
        # modes the report did not classify as zero. It must not: that case is
        # this PR's central one. On the #127 fixture the certificate band is
        # ``4.44e-13`` and the stationary eigenvalue sits at ``1e-7``, so the
        # cutoff is empty by construction, and explaining WHY the mode is out
        # there is the whole reason the audit exists. Withholding turned
        # ``test_pr127_fixture_is_reported_conditioning_limited`` red. The
        # fallback is therefore deliberate for an empty cutoff as well as for no
        # cutoff, and a reader should read ``cluster_size`` against
        # ``zero_tolerance``: when no eigenvalue is within the cutoff, the group
        # below is the nearest one, not a filtered zero set.
        #
        # ROUND-6 REVIEW. That rule was exact equality of the eigenvalues, and
        # the round-6 finding against the ``tied`` group applies here with more
        # force, because on this path the headline ``reciprocal_condition`` is
        # wrong too and this is the exported function's DEFAULT. Measured on the
        # round-5 two-sector fixture under a unitary change of basis -- the same
        # physical generator, so every conditioning figure must be invariant --
        # the default call reported ``0.1026`` against the operator's
        # ``0.7071``, because the rotation splits the exactly degenerate
        # manifold by ``2.9e-16`` -- against a #108 round-off band of ``1.6e-13`` --
        # and exact equality no longer sees it.
        #
        # This is not the "invented threshold" #108 and #113 prohibit. That
        # prohibition is against a second zero-mode MAGNITUDE cutoff competing
        # with the certificate's; no magnitude cutoff is introduced here. The
        # test is between computed eigenvalues, it is the band #108 already
        # defines for that question, and it reduces to the old rule whenever the
        # spectrum is exactly tied.
        cluster = _unresolved_from(
            int(np.argmin(magnitudes)), np.arange(values.size, dtype=np.intp)
        )
        if cluster.size == 0:  # pragma: no cover - anchor is always its own member
            cluster = np.flatnonzero(magnitudes == magnitudes.min())

    stationary = int(cluster[int(np.argmin(magnitudes[cluster]))])

    if int(cluster.size) == 1:
        # A simple eigenvalue already has the exact scalar first-order
        # condition |y^H x| from the accepted eigendecomposition. Routing this
        # case through an independent Schur solve adds numerical error with no
        # new information. Issue #168 concerns repeated/defective clusters, so
        # Schur-projector conditioning is reserved for size > 1.
        s_cluster = float(per_mode[stationary])
    else:
        schur_condition = _schur_cluster_projector_condition(
            L,
            reference_eigenvalues=reported,
            cluster_indices=cluster,
        )
        if schur_condition is None:
            return ZeroModeConditioning.unavailable("schur_cluster_mismatch")
        s_cluster = float(schur_condition)
    # Round-4 review. Round 2 replaced the subspace figure here with the MINIMUM
    # per-mode value over the cluster, on a fail-closed argument. That was
    # wrong: it made the selected eigenvalue's estimate depend on unrelated
    # modes that happen to fall inside the caller's cutoff. Measured on a 4x4
    # whose trace-basis blocks are ``1e-10``, ``[[1e-5, 1], [0, 1.00001e-5]]``
    # and ``-1``, widening the cutoff from ``1e-9`` to ``2e-5`` leaves the
    # selected stationary eigenvalue at ``1e-10`` but moves
    # ``structural_forward_estimate`` from ``1e-10`` to ``1.0`` -- ten orders of
    # magnitude, contributed entirely by a near-defective pair the stationary
    # mode has nothing to do with.
    #
    # ROUND-5 REVIEW amends that to the group of modes EXACTLY TIED with the
    # selected one, conditioned as a subspace. For a repeated eigenvalue the
    # left and right eigenvectors LAPACK returns are an arbitrary basis of their
    # eigenspaces and can be rotated independently, so a per-mode ``|y^H x|`` is
    # basis dependent: measured on two decoupled damped sectors, rotating only
    # the right basis inside the degenerate stationary eigenspace moves the
    # per-mode values from 0.7206 to 0.5408 while ``sigma_min(Y^H X)`` stays
    # exactly 0.7071. A physical degenerate manifold would have been reported
    # conditioning-limited on LAPACK's choice of basis alone.
    #
    # One expression covers both: ``cluster_conditioning`` over a single index
    # IS ``|y^H x|``, so a simple mode still divides by its own condition number
    # and only a genuinely repeated one is conditioned as the subspace it is.
    #
    # ROUND-6 REVIEW. Round 5 grouped by EXACT eigenvalue equality, which
    # recognises a repeated mode only when LAPACK happens to return bit-
    # identical values. It does not for a generator written in a rotated basis.
    # Measured on the two-sector fixture of the round-5 test under a UNITARY
    # change of basis -- the same physical generator, so every conditioning
    # figure must be invariant -- the zero set's four eigenvalues come back
    # spread over ``2.9e-16`` instead of exactly tied, the exact-tie group
    # collapses to one mode, and the divisor becomes ``0.1026`` where the
    # operator's actual value is ``0.7071``: a 6.9x inflation of both forward
    # estimates, from the choice of basis alone.
    #
    # Modes are therefore grouped by whether the decomposition RESOLVES them,
    # using the backward error it already measured: two computed eigenvalues
    # closer than the sum of their own residuals are not separated by the data.
    # This invents no threshold -- the residuals are measurements of this very
    # solve, and the test reduces to exact equality when they vanish (the
    # unrotated fixture still groups all four). It deliberately does NOT merge a
    # split LARGER than the backward error: such a split is a property of the
    # operator, not of the arithmetic, and the per-mode value is then a genuine
    # -- if ill-conditioned -- figure, with ``reciprocal_condition`` beside it
    # for a reader to compare against.
    tied = _unresolved_from(stationary, cluster)
    # A multi-mode zero set has a well-defined spectral projector, but no
    # basis-independent scalar eigenvalue condition in the defective case.
    # Issue #168 showed that proximity cannot safely distinguish semisimple
    # from defective multiplicity. Keep the cluster evidence and abstain from
    # scalar first-order claims whenever the report zero set has >1 mode.
    cluster_only = int(cluster.size) > 1
    s_scalar = float("nan") if cluster_only else float(s_cluster)
    residuals: dict[int, tuple[float, float]] = {}
    for j in cluster:
        x = _unit(right[:, j])
        y = _unit(left[:, j])
        # An unusable eigenvector leaves the mode's backward error unmeasured.
        # ``inf`` is the fail-closed reading, and it reaches the estimates only
        # through the maxima below.
        residuals[int(j)] = (
            scaled_euclidean_norm(L @ x - values[j] * x)
            if x is not None
            else float("inf"),
            scaled_euclidean_norm(L.conj().T @ y - np.conj(values[j]) * y)
            if y is not None
            else float("inf"),
        )
    # ROUND-6 REVIEW. These maxima used to run over the whole cutoff cluster
    # while the divisor below is the conditioning of the TIED group only, so an
    # unrelated in-cutoff mode's residual entered an estimate that claims to
    # bound the displacement of the selected eigenvalue. That is the same
    # cutoff coupling round 4 removed from the structural estimate, in the other
    # numerator: measured on ``diag(1e-12, M, -1)`` with ``M`` a 2x2 block of
    # eigenvalues ``+-1e-10``, the selected mode's eigenvector is exact
    # (residual 0.0) yet widening the cutoff from ``1e-11`` to ``1e-9`` moves
    # ``solver_forward_estimate`` off zero, contributed entirely by modes the
    # selected eigenvalue has nothing to do with. The magnitudes there are
    # round-off, so no verdict flips on that fixture -- the defect is that the
    # number is not a property of what it names.
    residual_group = cluster if cluster_only else tied
    right_res = max((residuals[int(j)][0] for j in residual_group), default=0.0)
    left_res = max((residuals[int(j)][1] for j in residual_group), default=0.0)
    # ROUND-7 REVIEW, the completion of the round-6 change. This measured the
    # distance from the CUTOFF CLUSTER to its complement, while everything above
    # now conditions ``tied``. A mode the caller's cutoff admits but the
    # arithmetic RESOLVES from the selected one is then neither conditioned nor
    # counted as a neighbour -- so it drops out of the locality test even though
    # it is exactly the eigenvalue that test exists to notice. Measured on a
    # valid 4x4 in the unit-trace basis with eigenvalues
    # ``[1e-6, 2e-6, -1, -2]`` at ``zero_tolerance=3e-6``: the reported
    # separation was ``1.000001`` while the selected mode's nearest neighbour
    # sits ``1e-6`` away, exactly its own structural budget, and
    # ``displacement_explained`` said ``True`` where the module's own criterion
    # demands abstention. The separation is therefore measured from ``tied`` to
    # its complement in the WHOLE spectrum, the same group the estimates and the
    # residuals describe.
    separation_group = cluster if cluster_only else tied
    outside = np.setdiff1d(
        np.arange(values.size), separation_group, assume_unique=False
    )
    separation = (
        float(
            np.min(
                np.abs(
                    reported[outside][:, None]
                    - reported[separation_group][None, :]
                )
            )
        )
        if outside.size
        else float("inf")
    )
    observed = float(magnitudes[stationary])
    if cluster_only:
        return ZeroModeConditioning(
            available=True,
            reason="cluster_conditioning_only",
            eigenvalue=complex(reported[stationary]),
            cluster_size=int(cluster.size),
            reciprocal_condition=float(s_cluster),
            per_mode_reciprocal_condition=float("nan"),
            spectrum_min_reciprocal_condition=float("nan"),
            right_residual=float(right_res),
            left_residual=float(left_res),
            separation=separation,
            trace_defect=float(defect),
            solver_forward_estimate=float("nan"),
            structural_forward_estimate=float("nan"),
            observed_displacement=observed,
            zero_tolerance=tol,
            conditioning_limited=False,
            displacement_explained=None,
            verdict=CONDITIONING_CLUSTER_ONLY,
        )
    # Round-2 review, amended in round 4. Both estimates are about the
    # displacement of ONE scalar eigenvalue, so they divide by a PER-MODE
    # condition number, never by the subspace figure ``s_cluster``: for
    # ``[[0, 1], [0, delta]]`` the invariant subspace is perfectly conditioned
    # while each eigenvalue's own ``s`` is ``delta`` (measured 1e-8 at
    # ``delta = 1e-8``), so the subspace figure understated the sensitivity by
    # ``1/delta``. The divisor is the SELECTED mode's own value -- see
    # ``s_scalar`` above for why the cluster minimum, tried in round 2, was
    # wrong. ``s_cluster`` remains the REPORTED conditioning, because that is
    # the honest figure for a degenerate stationary manifold.
    solver_estimate = _divide_by_conditioning(max(right_res, left_res), s_scalar)
    structural_estimate = _divide_by_conditioning(float(defect), s_scalar)

    # Round-2 review. A NaN estimate used to be dropped from the budget, so the
    # verdict was computed from whichever half survived: measured on the finite
    # 4x4 input ``diag(1.3e308, 0, -1, 1.3e308)``, whose trace-defect norm is not
    # representable, ``structural_forward_estimate`` serialised as null while a
    # zero solver residual left a budget of zero and the audit published
    # ``BENIGN``. A verdict resting on an estimate that does not exist is
    # exactly what this module is supposed to refuse, so it withholds instead.
    if np.isnan(structural_estimate) or np.isnan(solver_estimate):
        return ZeroModeConditioning.unavailable("forward_estimate_unavailable")
    # Round-3 review. The two contributions are SUCCESSIVE perturbations -- from
    # the exact zero of the nearest trace-preserving operator, to that
    # operator's eigenvalue, to the one the solver returned -- so their
    # displacement bounds add by the triangle inequality rather than being
    # alternatives. ``max`` reported BENIGN whenever each was individually below
    # the cutoff even though the total was not: two estimates at ``0.6 * tol``
    # permit a displacement of ``1.2 * tol``. The sum is the conservative
    # budget, and it is the same quantity ``displacement_explained`` measures
    # against, so the two fields cannot disagree about what the evidence allows.
    budget_estimate = structural_estimate + solver_estimate
    # PR #173 REVIEW. ``BENIGN`` is an ACCEPTANCE, so it is bound to a
    # comparison that must come out TRUE: ``budget <= tol``. An unbounded
    # (``inf``) budget -- a defective pair, ``s == 0`` -- fails it and reads as
    # limited, and so would any NaN that ever reached this line (both NaN
    # estimates withhold above, and ``tol`` is finite by validation). The old
    # form ``isfinite(tol) and budget > tol`` made BENIGN the fall-through of a
    # False comparison, which is how the no-cutoff NaN became ``BENIGN``.
    verdict = _scalar_verdict(budget_estimate, tol)
    limited = verdict == CONDITIONING_LIMITED
    # PR #166 review, finding P2. ``tol`` used to enter this budget, which made
    # the answer tautological: cluster selection already guarantees
    # ``observed <= tol`` for every eigenvalue in the zero set, so the field
    # said "explained" on evidence it had not consulted. It now reads ONLY the
    # two forward estimates -- the quantities that actually claim to explain the
    # displacement -- and abstains when either is unavailable. Abstention covers
    # ``inf`` as well as ``nan``: an unbounded estimate would "explain" any
    # displacement whatsoever, and it serialises to null, so reporting ``True``
    # beside a missing number contradicts the field's own semantics.
    # ROUND-5 REVIEW. ``CONDITIONING_AGREEMENT_FACTOR`` is headroom on a LOCAL
    # first-order estimate, and no fixed factor can rescue it where first-order
    # theory does not apply at all. Measured on a 16x16 companion matrix in the
    # trace-vector basis -- subdiagonal ones and a single trace-row entry
    # ``1e-8``, so the eigenvalues exactly satisfy ``lambda**16 = 1e-8`` -- the
    # displacement is ``0.3162`` against a combined estimate of ``0.02196``, a
    # ratio of 14.4 that would be reported as NOT explained even though the
    # perturbation is the entire cause of it. The displacement there scales as
    # ``eps**(1/m)`` along a Jordan chain of length ``m``, so the ratio grows
    # without bound with ``m`` and calibrating the factor on an order-two
    # fixture cannot justify it.
    #
    # The regime test is measured rather than tuned: a perturbation small enough
    # for a local expansion cannot move an eigenvalue as far as its nearest
    # neighbour. When it has, the modes have interacted and no first-order
    # attribution is meaningful, so the field abstains instead of denying.
    # ``observed_displacement`` and ``separation`` are both in the payload, so a
    # reader can see why. On the fixture above ``0.3162 >= 0.1234`` and the
    # answer becomes ``None``; on the PR #127 fixture ``1.0e-07 < 2.0e-07`` and
    # the ordinary comparison still applies.
    #
    # ROUND-6 REVIEW. That test read only the OUTCOME. Locality is a property of
    # the PERTURBATION, and a perturbation larger than the separation invalidates
    # the expansion however small the displacement it happens to produce.
    # Measured on ``diag(1e-6, -1, -2, -3)`` at ``zero_tolerance=1e-5``: the
    # observed displacement is ``1e-06`` against a separation of ``1.000001``,
    # so the outcome test passes, while the trace correction that has to be
    # attributed is ``2.1213`` -- twice the distance to the next eigenvalue. The
    # field reported ``True``, an attribution first-order theory cannot support.
    # Both readings are kept: they fail in different directions, the outcome
    # test catching modes that demonstrably interacted (the companion matrix
    # above, whose estimate is small) and the estimate test catching a
    # perturbation too large to expand around (this fixture, whose displacement
    # is small).
    explained: bool | None
    if (
        not np.isfinite(observed)
        or not np.isfinite(structural_estimate)
        or not np.isfinite(solver_estimate)
        or (
            np.isfinite(separation)
            and (observed >= separation or budget_estimate >= separation)
        )
    ):
        explained = None
    else:
        explained = bool(
            observed <= CONDITIONING_AGREEMENT_FACTOR * budget_estimate
        )
    return ZeroModeConditioning(
        available=True,
        reason="ok",
        eigenvalue=complex(reported[stationary]),
        cluster_size=int(cluster.size),
        reciprocal_condition=s_cluster,
        per_mode_reciprocal_condition=s_scalar,
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
        verdict=verdict,
    )
