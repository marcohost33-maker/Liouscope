"""General linear-algebra helpers used across diagnostics.

Includes:

* :func:`eig_nonhermitian` --- wraps ``scipy.linalg.eig`` (LAPACK ``zgeev``),
  the non-Hermitian eigensolver required by Liouvillians. Anchor D.
* :func:`is_hermitian`, :func:`is_density_matrix` --- input validation.
* :func:`support_check` --- enforces ``supp(rho_0) subset supp(rho_ss)``
  with ``eps = 1e-12`` regularisation (anchor J).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .._consts import (
    EPS_HERMITICITY,
    EPS_SUPP,
    EPS_TRACE,
    VECTOR_RESIDUAL_REL_MAX,
    ZERO_MODE_AMBIGUITY_FACTOR,
    ZERO_MODE_APOSTERIORI_MARGIN,
    ZERO_MODE_EPS_FACTOR,
)


@dataclass(frozen=True, slots=True)
class EigenDecomposition:
    """Right eigendecomposition of a (possibly non-Hermitian) matrix."""

    eigenvalues: np.ndarray
    right_vectors: np.ndarray
    left_vectors: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.eigenvalues.size


def eig_nonhermitian(
    A: np.ndarray,
    *,
    compute_left: bool = False,
) -> EigenDecomposition:
    """Return the eigendecomposition of a generic non-Hermitian matrix.

    Always uses LAPACK ``zgeev`` (via ``scipy.linalg.eig``). Anchor D: never
    fall back to ``eigh`` which assumes Hermiticity that Liouvillians do not
    have.

    The promotion to ``complex128`` below is what makes the "always zgeev"
    contract TRUE: unlike ``numpy.linalg`` (which computes in double
    regardless of input dtype and only casts the result back),
    ``scipy.linalg.eig`` dispatches by dtype and would run single-precision
    ``cgeev`` on a ``complex64`` input — measured at ~30x the eigenvalue error
    of the double solve on the same stored matrix. Every zero-mode/backward-
    error tolerance downstream (issue #108) is calibrated against the double
    solve, so the solver precision must not silently follow the caller's
    storage dtype. Representation error already present in single-precision
    INPUT data is the caller's data quality and is not (cannot be) undone
    here.
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"eig_nonhermitian expects a square matrix, got {A.shape}")
    if compute_left:
        eigenvalues, vl, vr = sla.eig(A, left=True, right=True)
        return EigenDecomposition(
            eigenvalues=np.asarray(eigenvalues),
            right_vectors=np.asarray(vr),
            left_vectors=np.asarray(vl),
        )
    eigenvalues, vr = sla.eig(A, left=False, right=True)
    return EigenDecomposition(
        eigenvalues=np.asarray(eigenvalues),
        right_vectors=np.asarray(vr),
    )


@dataclass(frozen=True, slots=True)
class ZeroModeCertificate:
    """Outcome of the structural zero-mode check on a computed spectrum.

    ``applicable`` is False when the operator is not trace preserving, in which
    case no zero eigenvalue is guaranteed and the check says nothing.
    """

    applicable: bool
    certified: bool
    solver: str
    residual: float
    #: The RAW operator-derived band ``rtol * eps * ||L||_2`` that certification
    #: was carried out against. REPORT AND DIAGNOSTICS ONLY -- it is NOT a
    #: filter cutoff. After an a posteriori refinement (issue #113 second axis)
    #: it is strictly LARGER than the tolerance actually applied, so filtering
    #: by it discards the very slow mode the certificate rescued. Every filter
    #: takes its cutoff from :meth:`zero_set_tolerance`.
    bound: float
    trace_defect: float
    zero_mode_count: int = 1
    ambiguous_count: int = 0
    #: Tolerance the gap filters must apply. Equal to ``bound`` unless the a
    #: posteriori certificate (issue #113 second axis) pulled a genuine slow
    #: mode out of the band -- then it separates the refined zero set from that
    #: mode, so D1/D3/D4 stop discarding physics the certificate has kept.
    zero_tolerance: float = float("nan")

    @property
    def applied_tolerance(self) -> float:
        """``zero_tolerance`` with the pre-refinement fallback to ``bound``."""
        return self.bound if not np.isfinite(self.zero_tolerance) else self.zero_tolerance

    def zero_set_tolerance(
        self,
        eigenvalues: np.ndarray,
        *,
        atol: float | None = None,
        name: str = "eigenvalues",
    ) -> float:
        """The ONE cutoff a zero-mode filter may apply to this spectrum.

        Round-17 review (PR #121). Five call sites had independently written
        the same two-branch expression -- the certificate's band when it is
        applicable, the radius-based proxy otherwise -- and four of them were
        never migrated when the a posteriori refinement (issue #113 second
        axis) made ``bound`` the wrong half of that branch. Duplicating a
        DECISION at five sites is what made a four-fold miss possible, so the
        decision is made here, once:

        * applicable certificate -> :attr:`applied_tolerance`: the raw band
          ``bound``, unless the refinement pulled a genuine slow mode out of
          it, in which case the refined split (never larger) is used, so the
          rescued mode survives the filter it was rescued for;
        * inapplicable certificate -> the radius-based proxy of
          :func:`liouscope.numerics.scale.spectral_zero_tolerance`. Without
          established trace preservation no zero mode is guaranteed and the
          operator-norm band can exceed the entire spectrum of a strongly
          non-normal input (measured: 2.2e3 against eigenvalues of order 1),
          which would discard every mode and report a gapless D1;
        * an explicit ``atol`` (the legacy pre-#108 absolute opt-in) still
          wins over both.

        The spectrum is validated by ``spectral_zero_tolerance`` in every
        branch, so an override cannot reintroduce silent acceptance of
        corrupted solver output.
        """
        # Deferred import: ``numerics.scale`` imports ``require_finite_square_2d``
        # from this module, so a module-level import would be circular.
        from .scale import spectral_zero_tolerance

        if atol is None and self.applicable:
            atol = self.applied_tolerance
        return spectral_zero_tolerance(eigenvalues, atol=atol, name=name)

    @property
    def resolved(self) -> bool:
        """True iff every mode is decidably zero or decidably non-zero.

        A count of zero modes greater than one is NOT by itself a problem: a
        conserved quantity or symmetry sector gives a genuinely degenerate
        stationary manifold whose extra zero modes sit at *machine zero*, and
        the gap read off the complement is correct.

        The failure mode (issue #113) is different and is what
        ``ambiguous_count`` measures: eigenvalues that fall inside the #108
        zero-mode tolerance while being far ABOVE the bare backward error
        ``eps * ||L||``. Such a mode is neither machine-zero nor resolved --
        it lives inside the safety factor the tolerance carries, so the
        arithmetic cannot say whether it is a genuine slow mode or noise.
        When any exist, D1 would silently discard real physics and report the
        next surviving (fast) mode instead.
        """
        return self.applicable and self.certified and self.ambiguous_count == 0

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view for the run report (RFC 8259: no NaN/inf)."""

        def _f(x: float) -> float | None:
            return float(x) if np.isfinite(x) else None

        return {
            "applicable": self.applicable,
            "certified": self.certified,
            "resolved": self.resolved,
            "solver": self.solver,
            "residual": _f(self.residual),
            "bound": _f(self.bound),
            "trace_defect": _f(self.trace_defect),
            "zero_mode_count": int(self.zero_mode_count),
            "ambiguous_count": int(self.ambiguous_count),
            "zero_tolerance": _f(self.applied_tolerance),
        }


def operator_zero_tolerance(
    L_super: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    name: str = "L_super",
) -> float:
    """Operator-derived zero-mode tolerance ``rtol * eps * ||L||_2``.

    Round-13 review (issue #112 follow-up): certification and downstream
    filtering must use ONE scale. The certificate accepts a stationary
    residual up to the true eigensolver backward error ``eps * ||L||_2``,
    but :func:`liouscope.numerics.scale.spectral_zero_tolerance` -- which
    only sees the spectrum -- can only use the spectral radius
    ``max|lambda|`` as a proxy. For a strongly non-normal generator
    ``||L||_2`` exceeds the radius by orders of magnitude (measured: 3.9e3
    on a 4x4 trace-preserving example), so a certified-resolved zero mode
    with residual between the two thresholds survived the radius filter and
    D1 reported ``~1e-12`` -- occasionally NEGATIVE -- instead of the true
    gap ``1``.

    Every consumer that holds the OPERATOR (not just its spectrum) must
    therefore filter on this scale -- but a consumer holding a
    :class:`ZeroModeCertificate` must take its cutoff from
    :meth:`ZeroModeCertificate.zero_set_tolerance` and NOT from the
    certificate's ``bound``. Round-17 review (PR #121): the two WERE the
    same expression when this paragraph was written, and stopped being so
    with the a posteriori refinement (issue #113 second axis), which lowers
    the applied tolerance below ``bound`` precisely when a genuine slow mode
    was rescued from the band. Filtering by ``bound`` there discards that
    mode again, one layer after the certificate saved it -- four call sites
    did exactly that, following this paragraph. The radius-based
    :func:`~liouscope.numerics.scale.spectral_zero_tolerance` remains the
    correct fallback for spectrum-only call sites. Genuine slow modes
    falling inside this coarser band are not silently swallowed: they are
    flagged by the certificate's ambiguity split (``resolved=False``,
    issue #113), which floors the verdict instead.

    Parameters
    ----------
    L_super
        The operator whose eigensolve backward error sets the scale.
    rtol
        Multiplier on ``eps * ||L||_2``, shared with the certificate and
        with ``spectral_zero_tolerance``.
    name
        Argument name for the fail-closed validation message.
    """
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    L_c = np.asarray(require_finite_square_2d(L_super, name=name), dtype=complex)
    norm2 = float(np.linalg.norm(L_c, 2)) if L_c.size else 0.0
    return float(rtol * float(np.finfo(float).eps) * norm2)


def trace_preservation_defect(L_super: np.ndarray) -> tuple[float, float]:
    """Return ``(||vec(I)^H L||_2, ||L||_F)`` for a vectorised generator.

    Every GKSL generator is trace preserving, which in column-stacking
    convention is the exact statement ``vec(I)^H L = 0``. The quotient of the
    two returned numbers is therefore a dimensionless, unit-invariant measure
    of how far ``L`` is from being a legal generator.
    """
    L_super = np.asarray(L_super)
    n = L_super.shape[0]
    d = int(round(np.sqrt(n)))
    if d * d != n:
        return float("nan"), float("nan")
    vec_i = np.eye(d, dtype=complex).reshape(-1, order="F")
    defect = float(np.linalg.norm(vec_i.conj() @ L_super))
    return defect, float(np.linalg.norm(L_super, ord="fro"))


def certified_nonzero_modes(
    L_c: np.ndarray,
    eigenvalues: np.ndarray,
    in_band: np.ndarray,
    *,
    right_vectors: np.ndarray | None = None,
    left_vectors: np.ndarray | None = None,
    margin: float = ZERO_MODE_APOSTERIORI_MARGIN,
) -> np.ndarray:
    """Modes inside the zero band that are provably NOT the stationary mode.

    The magnitude band ``|lambda| <= rtol * eps * ||L||`` is a worst-case
    backward-error argument: it assumes the eigensolver was as inaccurate as it
    is allowed to be. On a well-conditioned generator it usually was not, and
    the assumption is what discards genuine slow modes -- the
    ``ZERO_MODE_AMBIGUITY_FACTOR`` comment records that no threshold on
    ``|lambda|`` alone can separate the two populations, which are measured
    only a factor of 2 apart.

    This is the second axis. For each in-band mode we compute the *a
    posteriori* bound from the residuals actually attained,

        |lambda - lambda_hat| <~ max(||L x - lambda_hat x||,
                                     ||L^H y - conj(lambda_hat) y||) / |y^H x|

    with ``||x|| = ||y|| = 1``. Were ``lambda`` exactly zero, that bound would
    force ``|lambda_hat| <= bound``; so ``|lambda_hat| > margin * bound``
    certifies a genuine mode instead of merely scoring one. ``margin`` covers
    the first-order caveat only -- see ``ZERO_MODE_APOSTERIORI_MARGIN`` for the
    corpus measurement behind its value.

    Deliberately one-directional: the result can only move a mode OUT of the
    zero set, never into it, so it cannot manufacture the false "unresolved"
    verdict that the ambiguity split exists to avoid.

    Two invariants are enforced fail-closed:

    * fewer than two in-band modes -> nothing is certified. A trace-preserving
      generator has at least one stationary mode, so a lone in-band mode is the
      expected one and there is no decision to make.
    * the smallest-magnitude in-band mode is never certified. Trace
      preservation guarantees a zero eigenvalue exists; emptying the band would
      contradict the very precondition under which this certificate applies.
    """
    idx = np.flatnonzero(in_band)
    out = np.zeros(eigenvalues.shape, dtype=bool)
    if idx.size < 2:
        return out
    # The stationary mode itself is never a candidate (see docstring).
    idx = idx[idx != idx[int(np.argmin(np.abs(eigenvalues[idx])))]]

    vr, vl, ref = right_vectors, left_vectors, eigenvalues
    if vr is None or vl is None:
        # Own decomposition: the certificate is a property of the OPERATOR, so
        # it is legitimate to certify a candidate ladder's spectrum from the
        # operator's own eigenpairs -- but only where the two agree.
        try:
            ref, vl, vr = sla.eig(L_c, left=True, right=True)
        except (ValueError, sla.LinAlgError):
            return out

    tiny = np.finfo(float).tiny
    for i in idx:
        lam = eigenvalues[i]
        j = int(np.argmin(np.abs(ref - lam)))
        mag = float(abs(ref[j]))
        # Fail closed when the borrowed decomposition disagrees about this mode:
        # a bound computed for a different eigenvalue certifies nothing.
        if abs(ref[j] - lam) > 0.1 * max(abs(lam), mag) or mag == 0.0:
            continue
        x = vr[:, j] / max(float(np.linalg.norm(vr[:, j])), tiny)
        y = vl[:, j] / max(float(np.linalg.norm(vl[:, j])), tiny)
        sep = float(abs(np.vdot(y, x)))
        if sep <= 0.0:
            continue  # defective pair: the first-order bound does not apply
        res = max(
            float(np.linalg.norm(L_c @ x - ref[j] * x)),
            float(np.linalg.norm(L_c.conj().T @ y - np.conj(ref[j]) * y)),
        )
        if mag > margin * (res / sep):
            out[i] = True
    return out


def refine_zero_band(
    L_c: np.ndarray,
    eigenvalues: np.ndarray,
    magnitudes: np.ndarray,
    bound: float,
    *,
    right_vectors: np.ndarray | None = None,
    left_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Zero-mode membership plus the tolerance the gap filters must then apply.

    Returns ``(in_band, zero_tolerance)``. Without a certified non-stationary
    mode both are exactly the pre-refinement values, so the healthy path is
    unchanged bit for bit.

    The tolerance has to travel with the membership: D1/D3/D4 filter by
    magnitude, so a mode this certificate rescues would be discarded again one
    layer later if they kept reading the raw ``bound``. It is placed at the
    geometric mean of the two populations -- the scale-free midpoint, which is
    the right notion of "between" for quantities spanning decades.

    Fail-closed on inversion: if some mode kept as zero is LARGER than the
    smallest certified one, no single threshold can express the split, and the
    refinement is abandoned wholesale rather than applied half-way.
    """
    nonzero = certified_nonzero_modes(
        L_c,
        eigenvalues,
        magnitudes <= bound,
        right_vectors=right_vectors,
        left_vectors=left_vectors,
    )
    in_band = (magnitudes <= bound) & ~nonzero
    if not nonzero.any():
        return in_band, bound
    hi = float(np.min(magnitudes[nonzero]))
    lo = float(np.max(magnitudes[in_band])) if in_band.any() else 0.0
    if lo >= hi:
        return magnitudes <= bound, bound
    return in_band, float(np.sqrt(lo * hi)) if lo > 0.0 else hi * 0.5


def certified_eigvals(
    L_super: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    tp_rtol: float = 1.0e-10,
) -> tuple[np.ndarray, ZeroModeCertificate]:
    """Eigenvalues of a Liouvillian, checked against an exact structural fact.

    Issue #112. For a trace-preserving generator ``vec(I)^H L = 0`` holds
    *exactly*, so ``0`` is an exact eigenvalue and a correct eigensolve must
    return some ``|lambda| <= rtol * eps * ||L||``. This is a **theorem, not a
    threshold**: when the computed spectrum contains no such eigenvalue the
    solve has demonstrably failed, and no choice of zero-mode tolerance
    (relative or absolute, issue #108) can repair it -- the missing information
    is not in the eigenvalues.

    Why this is needed
    ------------------
    LAPACK's complex non-symmetric driver (``zgeev``) deflates a subdiagonal
    when the Ahues-Tisseur test ``|h[i,i-1] h[i-1,i]| <= eps |h[i,i]|
    |h[i-1,i-1] - h[i,i]|`` passes. On a STIFF generator the large diagonal
    entries inflate that right-hand side: with a fast rate ``2.7e5`` alongside
    slow rates ``~1e-5``, the bound reaches ``~1.6e-5`` and the entire slow
    block deflates to its own DIAGONAL. Measured on a four-level classical jump
    network, the exact zero mode vanished from the returned spectrum and a
    spurious mode appeared in its place, so ``D1`` reported ``7.28e-6`` for a
    generator whose true gap is ``1.07e-5``.

    That failure is invisible to conditioning: the per-eigenvalue condition
    numbers of the wrong spectrum are ``1``-``25``, i.e. the solver reports the
    wrong answer as well conditioned. Balancing does not fix it either
    (measured). What does fix it is solving the same matrix by a route whose
    deflation is not driven by the stiff diagonal.

    Repair ladder
    -------------
    Candidates are tried in order and the FIRST one satisfying the certificate
    WITHOUT ambiguous in-band modes is returned; the primary solve is
    therefore byte-identical whenever it is already correct (measured:
    400/400 random four-level networks, stiff and well-conditioned alike,
    keep the ``zgeev`` result). An ambiguous candidate does not end the
    ladder (round-15 review): a later route can resolve the very modes it
    cannot -- measured, ``zgeev`` certified a stiff network with one
    ambiguous mode at ``6.3e-7`` while ``dgeev-real`` returned a clean
    ``2.1e-15`` stationary mode. The best ambiguous candidate (fewest
    ambiguous modes) is returned, fail-closed, only when every route stays
    ambiguous.

    1. ``zgeev`` -- the incumbent, via :func:`eig_nonhermitian`;
    2. ``dgeev`` on the real part, when ``L`` is real-valued (a different
       LAPACK driver with a different deflation path);
    3. the complex Schur form ``zgees``, which deflates on the full Hessenberg
       structure rather than eigenvalue-by-eigenvalue;
    4. an explicitly balanced ``zgeev``.

    If no candidate is certified, the eigenvalues of the best (smallest
    residual) candidate are returned together with ``certified=False`` -- the
    caller decides how loudly to fail. The function never silently substitutes
    an uncertified spectrum for a certified one.

    Parameters
    ----------
    L_super
        ``d^2 x d^2`` vectorised generator.
    rtol
        Multiplier on the backward error ``eps * ||L||_2``, shared with
        :func:`liouscope.numerics.scale.spectral_zero_tolerance`.
    tp_rtol
        Relative tolerance deciding whether ``L`` is trace preserving at all.
        A non-trace-preserving operator has no guaranteed zero mode, so the
        certificate is reported as ``applicable=False`` and the incumbent
        spectrum is returned untouched.

    Returns
    -------
    tuple
        ``(eigenvalues, certificate)``.
    """
    L_super = require_finite_square_2d(L_super, name="L_super")
    L_c = np.asarray(L_super, dtype=complex)
    tp_defect, fro = trace_preservation_defect(L_c)
    norm2 = float(np.linalg.norm(L_c, 2)) if L_c.size else 0.0
    bound = rtol * float(np.finfo(float).eps) * norm2

    # Round-17 review (PR #121). The primary solve must not be able to end the
    # ladder. ``zgeev`` raising ``LinAlgError`` IS a nonconvergence -- exactly
    # the failure the real-driver / Schur / balanced routes exist to repair --
    # and letting the exception propagate defeated the ladder precisely in the
    # case it was built for. The error is carried instead and re-raised only
    # if NO route produced a spectrum at all, so the caller still sees the
    # original diagnosis when nothing worked.
    primary: np.ndarray | None
    primary_error: sla.LinAlgError | None
    try:
        primary = np.asarray(eig_nonhermitian(L_c).eigenvalues)
        primary_error = None
    except sla.LinAlgError as exc:
        primary, primary_error = None, exc
    if not np.isfinite(tp_defect) or tp_defect > tp_rtol * max(fro, np.finfo(float).tiny):
        # No certificate applies without trace preservation, so there is
        # nothing to repair TOWARDS and the ladder is not run: the primary
        # failure is the honest answer.
        if primary is None:
            assert primary_error is not None
            raise primary_error
        return primary, ZeroModeCertificate(
            applicable=False,
            certified=False,
            solver="zgeev",
            residual=float(np.min(np.abs(primary))) if primary.size else float("nan"),
            bound=bound,
            trace_defect=tp_defect,
        )

    def _candidates() -> Iterator[tuple[str, np.ndarray]]:
        # LAZY by design (twelfth-round review): the ladder stops at the first
        # certified spectrum, and the healthy path is the primary solve -- an
        # eager list would charge every caller three additional cubic
        # decompositions that the loop then never looks at.
        if primary is not None:
            yield ("zgeev", primary)
        # ``dgeev`` genuinely solves a DIFFERENT matrix unless L is EXACTLY
        # real: ``np.allclose`` carries an absolute default atol, so a stiff
        # complex generator in small rate units read as "real" and the repair
        # silently deleted its Hamiltonian part (measured: D3 moved from 1e-7
        # to ~9.7e-6 under a pure L -> 1e-10 L rescale). Exact realness is the
        # only scale-invariant criterion under which the route is valid.
        if not np.any(L_c.imag):
            # GUARDED (round-16 review): an exactly real generator whose
            # ``dgeev`` solve raises must not end the ladder. Without this
            # suppression the exception propagates out of the generator and
            # the two remaining repair routes below -- ``zgees-schur`` and
            # ``balanced-zgeev`` -- never run, even when either could recover
            # a certifiable spectrum. The route is a REPAIR attempt like the
            # others; a repair that fails is a route that did not work, not a
            # reason to abandon the remaining ones.
            #
            # The sibling ladder in ``certified_eig`` already guarded exactly
            # this step. The omission here was the inconsistency, not the
            # guard there.
            with contextlib.suppress(ValueError, sla.LinAlgError):
                yield ("dgeev-real", np.linalg.eigvals(L_c.real).astype(complex))
        with contextlib.suppress(ValueError, sla.LinAlgError):
            yield ("zgees-schur", np.diag(sla.schur(L_c, output="complex")[0]))
        with contextlib.suppress(ValueError, sla.LinAlgError):
            balanced = sla.matrix_balance(L_c, permute=True)[0]
            yield ("balanced-zgeev", np.linalg.eigvals(balanced))

    # Split the zero-tolerance band (issue #113). Below the split a mode is
    # machine-zero; above it -- but still inside the #108 tolerance, which
    # carries a factor ``rtol`` on top -- it is neither machine-zero nor
    # resolved.
    #
    # The factor is set from a MEASURED distribution, not from taste. Across
    # 83 healthy generators (single/two-qubit families plus random GKSL,
    # unique and degenerate steady states alike) the largest in-band
    # |lambda| reaches 2.38 * eps*||L||, median 0.39. Unresolved slow modes
    # were measured at 4.87 and 4.87e2. The two populations are therefore
    # NOT cleanly separated -- the nearest pair is 2.38 against 4.87, about
    # 2x -- so the split is placed at 30x, roughly an order of magnitude
    # clear of the healthy maximum rather than midway between the two.
    #
    # That choice is deliberately conservative: a false NaN on a healthy
    # generator destroys a correct analysis, while a missed marginal case
    # leaves behaviour exactly as it was before this check existed. The cost
    # is bounded reach -- above a spectral spread of ~1e14 the slow modes
    # sink below the split and the defect is undetectable by any magnitude
    # test, because double precision has genuinely lost the information.
    # That residual is documented in issue #113 rather than papered over
    # here.
    split = ZERO_MODE_AMBIGUITY_FACTOR * float(np.finfo(float).eps) * norm2
    best: tuple[str, np.ndarray, float] | None = None
    ambiguous_best: tuple[str, np.ndarray, float, int, int, float] | None = None
    for name, ev in _candidates():
        magnitudes = np.abs(ev)
        residual = float(np.min(magnitudes)) if ev.size else float("inf")
        if residual <= bound:
            # Issue #113 second axis: a mode the a posteriori bound certifies
            # as non-stationary leaves the zero set before it is counted, so a
            # genuine slow mode no longer masquerades as a second zero mode and
            # hands D1 the NEXT eigenvalue as the gap.
            in_band, zero_tolerance = refine_zero_band(
                L_c, ev, magnitudes, bound
            )
            zero_count = int(np.count_nonzero(in_band))
            ambiguous = int(np.count_nonzero(in_band & (magnitudes > split)))
            if ambiguous == 0:
                return ev, ZeroModeCertificate(
                    applicable=True,
                    certified=True,
                    solver=name,
                    residual=residual,
                    bound=bound,
                    trace_defect=tp_defect,
                    zero_mode_count=zero_count,
                    zero_tolerance=zero_tolerance,
                )
            # Round-15 review: an AMBIGUOUS candidate must not end the
            # ladder -- a later route can resolve the very modes this one
            # cannot (measured: zgeev certified a stiff network with one
            # ambiguous mode at 6.3e-7 while dgeev-real returns a clean
            # 2.1e-15 stationary mode). It is retained only as the
            # fail-closed fallback when EVERY route stays ambiguous, keyed
            # by the fewest ambiguous modes (ties: ladder order).
            if ambiguous_best is None or ambiguous < ambiguous_best[4]:
                ambiguous_best = (
                    name, ev, residual, zero_count, ambiguous, zero_tolerance
                )
            continue
        if best is None or residual < best[2]:
            best = (name, ev, residual)

    if ambiguous_best is not None:
        name, ev, residual, zero_count, ambiguous, zero_tolerance = ambiguous_best
        return ev, ZeroModeCertificate(
            applicable=True,
            certified=True,
            solver=name,
            residual=residual,
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=zero_count,
            ambiguous_count=ambiguous,
            zero_tolerance=zero_tolerance,
        )
    if best is None:
        # Every route raised (only reachable when the primary solve did too).
        assert primary_error is not None
        raise primary_error
    return best[1], ZeroModeCertificate(
        applicable=True,
        certified=False,
        solver=best[0],
        residual=best[2],
        bound=bound,
        trace_defect=tp_defect,
        zero_mode_count=0,
    )


def certified_eig(
    L_super: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    tp_rtol: float = 1.0e-10,
) -> tuple[EigenDecomposition, ZeroModeCertificate]:
    """Like :func:`certified_eigvals`, but also returns left/right eigenvectors.

    Issue #112 follow-up. The eigenvalue-only certificate protects D1/D3/D4, but
    the layers that consume eigen*vectors* -- D19's slowest-mode overlap above
    all -- were still reading the uncertified decomposition. That matters more
    than it sounds: on a stiff generator ``zgeev`` can return a *spurious* slow
    mode, and D19 then measures the initial state's overlap with an eigenvector
    that corresponds to no physical mode. A wrong overlap on that rung fires a
    false A11/F4 Mpemba candidate -- the highest-priority branch of the
    classifier, and exactly the failure mode #108 was fixed to prevent.

    Repair ladder, narrower than the eigenvalue-only one
    ----------------------------------------------------
    Only routes that actually produce eigenvectors in the ORIGINAL basis are
    usable here:

    1. ``zgeev`` -- the incumbent;
    2. ``dgeev`` on the real part, when ``L`` is real-valued.

    The Schur and balanced routes are deliberately absent: ``zgees`` yields no
    eigenvectors directly, and the balanced solve returns them in the balanced
    basis, where back-transforming re-introduces exactly the scaling that made
    the decomposition unreliable. A narrower ladder is the honest choice --
    it means some spectra that ``certified_eigvals`` can repair are NOT
    repairable here, and the certificate then reports ``certified=False``
    rather than pretending otherwise.

    Beyond the eigenvalue certificate, a candidate is accepted only when
    every consumed mode's left AND right eigen*vector* residuals pass the
    per-mode relative gate ``r_j <= max(VECTOR_RESIDUAL_REL_MAX * |lambda_j|,
    bound)`` (round-15 review) -- the vectors are what D19 and the Petermann
    factors consume, and a small ``|lambda|`` does not vouch for them. When
    the vectors fail on every route, the returned certificate carries
    ``certified=False`` with ``residual`` set to the offending (vector)
    residual, so the downstream warning shows a number that actually
    exceeds the printed bound. Ambiguous candidates likewise no longer end
    the ladder (same rule as :func:`certified_eigvals`).

    Returns
    -------
    tuple
        ``(EigenDecomposition with left_vectors set, certificate)``.
    """
    L_super = require_finite_square_2d(L_super, name="L_super")
    L_c = np.asarray(L_super, dtype=complex)
    tp_defect, fro = trace_preservation_defect(L_c)
    norm2 = float(np.linalg.norm(L_c, 2)) if L_c.size else 0.0
    bound = rtol * float(np.finfo(float).eps) * norm2

    # Round-17 review (PR #121). The primary solve must not be able to end the
    # ladder. ``zgeev`` raising ``LinAlgError`` IS a nonconvergence -- exactly
    # the failure the real-driver / Schur / balanced routes exist to repair --
    # and letting the exception propagate defeated the ladder precisely in the
    # case it was built for. The error is carried instead and re-raised only
    # if NO route produced a spectrum at all, so the caller still sees the
    # original diagnosis when nothing worked.
    primary: EigenDecomposition | None
    primary_error: sla.LinAlgError | None
    try:
        primary = eig_nonhermitian(L_c, compute_left=True)
        primary_error = None
    except sla.LinAlgError as exc:
        primary, primary_error = None, exc
    if not np.isfinite(tp_defect) or tp_defect > tp_rtol * max(fro, np.finfo(float).tiny):
        if primary is None:
            assert primary_error is not None
            raise primary_error
        return primary, ZeroModeCertificate(
            applicable=False,
            certified=False,
            solver="zgeev",
            residual=(
                float(np.min(np.abs(primary.eigenvalues)))
                if primary.eigenvalues.size
                else float("nan")
            ),
            bound=bound,
            trace_defect=tp_defect,
        )

    def _candidates() -> Iterator[tuple[str, EigenDecomposition]]:
        # Lazy + exact realness, for the same reasons as the eigenvalue ladder.
        if primary is not None:
            yield ("zgeev", primary)
        if not np.any(L_c.imag):
            with contextlib.suppress(ValueError, sla.LinAlgError):
                w, vl, vr = sla.eig(L_c.real, left=True, right=True)
                yield (
                    "dgeev-real",
                    EigenDecomposition(
                        eigenvalues=np.asarray(w).astype(complex),
                        right_vectors=np.asarray(vr).astype(complex),
                        left_vectors=np.asarray(vl).astype(complex),
                    ),
                )

    def _vector_residual(decomp: EigenDecomposition) -> float:
        """Largest OFFENDING per-mode eigenvector residual; 0.0 when all pass.

        Round-15 review: the eigenvalue certificate alone does not guarantee
        the eigen*vectors* -- the quantities D19 and the Petermann factors
        actually consume. Measured on a valid stiff classical network:
        certified and resolved with ``bound = 3.2e-7``, yet the LEFT vector
        of the selected slow mode had residual ``3.1e-4`` -- three orders of
        magnitude beyond the certificate, i.e. not an eigenvector of ``L``
        in any usable sense (the right vectors were fine).

        The acceptance test is PER MODE and RELATIVE:
        ``r_j <= max(VECTOR_RESIDUAL_REL_MAX * |lambda_j|, bound)`` for every
        non-zero-band mode, with ``r_j`` the larger of the unit-normalised
        left/right residuals. ``r_j / |lambda_j|`` is the first-order
        relative error scale of anything computed from the pair, and unlike
        a single operator-scale cutoff it separates the measured
        populations: a legitimate ``dgeev-real`` repair carries a few
        percent on its slow modes, while a corrupt ``zgeev`` decomposition
        carries 22% to 2900% (see
        :data:`liouscope._consts.VECTOR_RESIDUAL_REL_MAX` for the measured
        calibration). Modes inside the zero band are not gated here: their
        vectors span the stationary manifold and are not consumed through
        this decomposition.
        """
        ev = decomp.eigenvalues
        if not ev.size:
            return 0.0
        vr = decomp.right_vectors
        vl = decomp.left_vectors
        assert vl is not None  # every ladder route computes left vectors
        tiny = np.finfo(float).tiny
        res_r = np.linalg.norm(L_c @ vr - vr * ev[None, :], axis=0) / np.maximum(
            np.linalg.norm(vr, axis=0), tiny
        )
        res_l = np.linalg.norm(
            L_c.conj().T @ vl - vl * np.conj(ev)[None, :], axis=0
        ) / np.maximum(np.linalg.norm(vl, axis=0), tiny)
        res = np.maximum(res_r, res_l)
        magnitudes = np.abs(ev)
        consumed = magnitudes > bound
        if not consumed.any():
            return 0.0
        allowed = np.maximum(VECTOR_RESIDUAL_REL_MAX * magnitudes[consumed], bound)
        offending = res[consumed] > allowed
        return float(np.max(res[consumed][offending])) if offending.any() else 0.0

    split = ZERO_MODE_AMBIGUITY_FACTOR * float(np.finfo(float).eps) * norm2
    best: tuple[str, EigenDecomposition, float] | None = None
    ambiguous_best: (
        tuple[str, EigenDecomposition, float, int, int, float] | None
    ) = None
    vector_failed: tuple[str, EigenDecomposition, float, float] | None = None
    for name, decomp in _candidates():
        magnitudes = np.abs(decomp.eigenvalues)
        residual = float(np.min(magnitudes)) if magnitudes.size else float("inf")
        if residual <= bound:
            # Issue #113 second axis: a mode the a posteriori bound certifies
            # as non-stationary leaves the zero set before it is counted, so a
            # genuine slow mode no longer masquerades as a second zero mode and
            # hands D1 the NEXT eigenvalue as the gap.
            in_band, zero_tolerance = refine_zero_band(
                L_c,
                decomp.eigenvalues,
                magnitudes,
                bound,
                right_vectors=decomp.right_vectors,
                left_vectors=decomp.left_vectors,
            )
            zero_count = int(np.count_nonzero(in_band))
            ambiguous = int(np.count_nonzero(in_band & (magnitudes > split)))
            if ambiguous == 0:
                vec_residual = _vector_residual(decomp)
                if vec_residual == 0.0:
                    return decomp, ZeroModeCertificate(
                        applicable=True,
                        certified=True,
                        solver=name,
                        residual=residual,
                        bound=bound,
                        trace_defect=tp_defect,
                        zero_mode_count=zero_count,
                        zero_tolerance=zero_tolerance,
                    )
                # Eigenvalues fine, eigenvectors demonstrably not: keep for
                # the fail-closed record and try the next route.
                if vector_failed is None or vec_residual < vector_failed[3]:
                    vector_failed = (name, decomp, residual, vec_residual)
                continue
            # Round-15 review: an ambiguous candidate must not end the
            # ladder (same rule as certified_eigvals).
            if ambiguous_best is None or ambiguous < ambiguous_best[4]:
                ambiguous_best = (
                    name, decomp, residual, zero_count, ambiguous, zero_tolerance
                )
            continue
        if best is None or residual < best[2]:
            best = (name, decomp, residual)

    if ambiguous_best is not None:
        name, decomp, residual, zero_count, ambiguous, zero_tolerance = (
            ambiguous_best
        )
        return decomp, ZeroModeCertificate(
            applicable=True,
            certified=True,
            solver=name,
            residual=residual,
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=zero_count,
            ambiguous_count=ambiguous,
            zero_tolerance=zero_tolerance,
        )
    if vector_failed is not None:
        # No route produced consumable eigenvectors. certified=False keeps
        # ``resolved`` False, so every consumer withholds -- the eigenvalue
        # residual alone must not be allowed to read as success when the
        # vectors it vouches for are not eigenvectors.
        name, decomp, residual, vec_residual = vector_failed
        return decomp, ZeroModeCertificate(
            applicable=True,
            certified=False,
            solver=name,
            residual=max(residual, vec_residual),
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=0,
        )
    if best is None:
        # Every route raised (only reachable when the primary solve did too).
        assert primary_error is not None
        raise primary_error
    return best[1], ZeroModeCertificate(
        applicable=True,
        certified=False,
        solver=best[0],
        residual=best[2],
        bound=bound,
        trace_defect=tp_defect,
        zero_mode_count=0,
    )


def require_finite_square_2d(A: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Validate ``A`` as a finite, square, 2-D array at a public boundary.

    Returns ``np.asarray(A)`` unchanged when valid, else raises a
    :class:`ValueError` that names the offending argument *and* the concrete
    defect. This is a fail-closed guard for public entry points: without it a
    ``NaN``/``inf``-laden Liouvillian flows into ``scipy.linalg.expm`` / ``svd``
    and surfaces as an opaque, location-blind error deep in LAPACK
    (``"array must not contain infs or NaNs"`` / ``"SVD did not converge"``),
    leaving the caller no clue which input was malformed.

    Parameters
    ----------
    A
        Candidate array.
    name
        Human-readable argument name used in error messages.
    """
    arr = np.asarray(A)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.count_nonzero(np.isnan(arr)))
        n_inf = int(np.count_nonzero(np.isinf(arr)))
        raise ValueError(
            f"{name} contains non-finite entries "
            f"({n_nan} NaN, {n_inf} inf); diagnostics require a finite operator"
        )
    return arr


def hermiticity_defect(A: np.ndarray) -> tuple[float, float]:
    """Return ``(max|A - A^H|, max|A|)`` -- the absolute defect and its scale.

    Both are elementwise sup-norms, so their quotient is the *relative*
    Hermiticity defect: a dimensionless number that is unchanged by a pure
    change of units ``A -> cA``. Returning the pair (rather than the quotient)
    keeps the zero-operator case in the caller's hands, where the right
    semantics are known.
    """
    A = np.asarray(A)
    defect = float(np.max(np.abs(A - A.conj().T))) if A.size else 0.0
    scale = float(np.max(np.abs(A))) if A.size else 0.0
    return defect, scale


def is_hermitian(
    A: np.ndarray,
    atol: float | None = None,
    *,
    rtol: float = EPS_HERMITICITY,
) -> bool:
    """Return True iff ``A`` is Hermitian.

    The gate is SCALE-RELATIVE by default (issue #109): ``A`` is accepted iff
    ``max|A - A^H| <= rtol * max|A|``. An operator that is not trace-normalised
    -- a Hamiltonian, a jump operator -- carries a physical dimension, so an
    ABSOLUTE tolerance makes the validation verdict depend on the caller's
    choice of units, and it fails in both directions:

    * **fail-open** (the serious one) -- a genuinely non-Hermitian ``H`` with a
      relative defect of ``1e-6`` passes a fixed ``1e-9`` gate as soon as
      ``||H|| <~ 1e-3``. The resulting generator is not GKSL at all, and every
      downstream diagnostic then describes dynamics that are not a quantum
      channel;
    * **false rejection** -- an exactly Hermitian ``H`` obtained from a
      similarity transform carries round-off ``~eps*||H||``, which exceeds a
      fixed ``1e-9`` once ``||H|| >~ 1e8``, so valid input is refused.

    ``rtol`` is deliberately ``EPS_HERMITICITY = 1e-9``, i.e. the historical
    absolute value re-read as a relative one. At unit scale (``max|A| ~ 1``,
    the regime of every existing fixture) the gate is therefore unchanged; only
    operators far from unit scale move, which is exactly the defect. Note that
    ``1e-9`` is many orders above round-off (``~1e-16``) *by design*: this is a
    validation gate against malformed input, not a round-off discriminator.

    ``rtol=0`` semantics inside :func:`numpy.allclose` are not used here.
    ``np.allclose``'s own ``rtol`` compares against ``|b|`` elementwise, which
    for ``b = A^H`` makes the tolerance largest exactly where ``A`` is largest
    and vanish where ``A`` is zero -- not a statement about the operator's
    scale. The scale is taken once, from ``max|A|``.

    Parameters
    ----------
    A
        Candidate matrix. Non-square (or non-2-D) input returns ``False``.
    atol
        Legacy ABSOLUTE tolerance. When given, ``A`` is accepted iff
        ``max|A - A^H| <= atol``, reproducing the pre-#109 behaviour verbatim.
        This is the opt-in for callers whose operator is normalised to a known
        scale -- density matrices, where the absolute reading is already
        relative to ``tr(rho) = 1`` (see :func:`is_density_matrix`).
    rtol
        Relative tolerance on ``max|A|``; ignored when ``atol`` is given.

    Notes
    -----
    Zero-operator semantics: ``max|A| == 0`` gives ``tol = 0``, so the all-zero
    matrix (exactly Hermitian) passes and nothing else does. This mirrors the
    documented zero-operator behaviour of :func:`liouscope.numerics.scale.rate_scale`.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    defect, scale = hermiticity_defect(A)
    tol = float(atol) if atol is not None else rtol * scale
    return bool(defect <= tol)


def is_density_matrix(
    rho: np.ndarray,
    *,
    atol_trace: float = EPS_TRACE,
    atol_psd: float = 1.0e-10,
) -> bool:
    """Return True iff ``rho`` is a valid density matrix.

    Checks Hermiticity, unit trace and positive semi-definiteness.

    The Hermiticity gate stays ABSOLUTE here (issue #109). A density matrix is
    trace-normalised, so ``atol`` is already implicitly relative to a fixed
    scale ``tr(rho) = 1``; the rate-dimension argument that makes an absolute
    tolerance wrong for a Hamiltonian does not apply. Reading it as relative to
    ``max|rho|`` would instead *tighten* the gate by up to a factor ``d`` for
    nearly maximally mixed states -- a behaviour change with no defect behind it.
    """
    rho = np.asarray(rho)
    if not is_hermitian(rho, atol=1.0e-9):
        return False
    if abs(np.trace(rho) - 1.0) > atol_trace:
        return False
    # Hermitian => use eigvalsh for stability.
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return bool(evals.min() >= -atol_psd)


def support_check(
    rho_initial: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps: float = EPS_SUPP,
) -> tuple[bool, np.ndarray]:
    """Verify ``supp(rho_initial) subset supp(rho_steady)``.

    Returns a tuple ``(ok, regularised_steady)`` where ``regularised_steady``
    is ``rho_steady + eps * I / d`` shifted to avoid singularities when
    computing the relative entropy ``D(rho || pi)``.

    Anchor J: this guards against NaNs when ``rho_steady`` is rank-deficient.
    """
    rho_initial = np.asarray(rho_initial)
    rho_steady = np.asarray(rho_steady)
    d = rho_steady.shape[0]

    evals_steady, evecs_steady = np.linalg.eigh((rho_steady + rho_steady.conj().T) / 2)
    null_mask = evals_steady < eps
    if not null_mask.any():
        return True, rho_steady

    null_space = evecs_steady[:, null_mask]
    # Projector onto null space
    P_null = null_space @ null_space.conj().T
    leakage = float(np.real(np.trace(P_null @ rho_initial @ P_null)))
    ok = leakage < eps
    rho_steady_reg = rho_steady + (eps / d) * np.eye(d, dtype=rho_steady.dtype)
    return ok, rho_steady_reg


def matrix_2_norm(A: np.ndarray) -> float:
    """Spectral norm via SVD; works for non-square arrays."""
    return float(sla.svdvals(np.asarray(A))[0])
