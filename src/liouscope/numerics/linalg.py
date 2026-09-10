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
import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment

from .._consts import (
    EPS_HERMITICITY,
    EPS_SUPP,
    EPS_TRACE,
    VECTOR_RESIDUAL_REL_MAX,
    ZERO_MODE_AMBIGUITY_FACTOR,
    ZERO_MODE_APOSTERIORI_MARGIN,
    ZERO_MODE_EPS_FACTOR,
)
from .norms import (
    scaled_cancellation_ratio,
    scaled_column_sums,
    scaled_euclidean_norm,
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
    # ``require_finite_square_2d`` has already refused an empty operator, so
    # ``L_c.size`` is non-zero from here on.
    eps = float(np.finfo(float).eps)
    norm2 = float(np.linalg.norm(L_c, 2))
    if np.isfinite(norm2):
        tol = float(rtol * eps * norm2)
    else:
        # ROUND-24 REVIEW (PR #121). The round-22 finding, one helper across.
        # ``||L||_2`` can overflow for an operator every entry of which is
        # finite -- a 2x2 filled with ``1e308`` has a spectral norm of
        # ``2e308`` -- while the quantity this function returns,
        # ``rtol * eps * ||L||_2 ~ 4.4e295``, is comfortably representable.
        # Returning ``inf`` there is the same silent failure the spectrum-side
        # helper closed in round 22: no mode satisfies ``|lambda| > inf``, so a
        # consumer holding the OPERATOR discards its entire spectrum and reads
        # a huge non-zero mode as stationary. The input was valid; only the
        # intermediate was not.
        #
        # Scaling by the largest COMPONENT keeps every intermediate in range,
        # exactly as in :func:`liouscope.numerics.scale.spectral_zero_tolerance`
        # -- ``max(|Re|, |Im|)`` rather than ``max|.|`` because the modulus of
        # a finite complex entry can itself overflow. ``frexp`` selects a POWER
        # OF TWO, so the shift is exact in every mantissa bit and the round
        # trip introduces no rounding into a number whose only purpose is to be
        # compared against. ``m > 0`` holds here: a zero matrix has a finite
        # (zero) 2-norm and never reaches this branch.
        m = float(np.max(np.maximum(np.abs(np.real(L_c)), np.abs(np.imag(L_c)))))
        exp = int(np.frexp(m)[1])
        scaled = np.ldexp(np.real(L_c), -exp) + 1j * np.ldexp(np.imag(L_c), -exp)
        with np.errstate(over="ignore"):
            tol = float(
                np.ldexp(rtol * eps * float(np.linalg.norm(scaled, 2)), exp)
            )
    if not np.isfinite(tol):
        # Second line of defence, independent of the arithmetic above and the
        # same rule ``spectral_zero_tolerance`` applies to its own derived
        # value. A tolerance that is not finite cannot separate anything, so
        # the operator is refused rather than filtered against an unusable
        # threshold. Reached only when the tolerance ITSELF exceeds the double
        # range (``rtol`` near the top of it), not merely the norm.
        raise ValueError(
            f"the zero-mode tolerance derived from {name} is not finite "
            f"(rtol = {rtol}, ||L||_2 = {norm2}); no mode could exceed it, "
            "so the operator is refused rather than filtered against an "
            "unusable threshold"
        )
    return tol


def trace_preservation_defect(L_super: np.ndarray) -> tuple[float, float]:
    """Return robust ``(||vec(I)^H L||_2, ||L||_F)`` for a generator.

    Every GKSL generator is trace preserving, which in column-stacking
    convention is the exact statement ``vec(I)^H L = 0``. The quotient of the
    two returned numbers is therefore a dimensionless, unit-invariant measure
    of how far ``L`` is from being a legal generator.

    Issue #130: both quantities are evidence, so their arithmetic must be
    correct independently of the policy that consumes them. Ordinary
    ``np.linalg.norm`` squares before summing and can therefore return the
    false sentinels ``0.0`` (underflow) or ``inf`` (overflow) even when the
    mathematical norm of finite input is representable. Both norms now use
    :func:`scaled_euclidean_norm`, an xLASSQ-style power-of-two scaled
    sum-of-squares primitive. It returns a finite norm whenever the true
    float64 result is representable and returns infinity only when the input or
    the true norm is genuinely non-representable.

    This deliberately separates measurement from certificate policy. A very
    large but finite operator must not be rejected merely because a naive norm
    overflowed; structural guards such as ``band_discriminates`` decide whether
    the resulting zero-mode evidence actually discriminates.
    """
    L_super = np.asarray(L_super)
    # A 1-D argument used to reach the matrix product and come back with a
    # number, which said nothing about any superoperator. It is reported as
    # unevaluable through the same channel the non-square case already uses,
    # rather than as an index error from the row selection below.
    if L_super.ndim != 2:
        return float("nan"), float("nan")
    n = L_super.shape[0]
    d = int(round(np.sqrt(n)))
    if d * d != n:
        return float("nan"), float("nan")
    # ISSUE #139, P2 REVIEW. ``vec(I)^H @ L`` selects the diagonal-output rows
    # and adds them per column -- but an ordinary matrix product accumulates in
    # ordinary units, so a column that cancels to exactly zero can overflow on
    # the way there. Measured for d=16 with eight entries ``+2.5e307`` and
    # eight ``-2.5e307`` in one column: the product returned ``inf+nanj`` and
    # the caller refused a generator whose trace equation is exactly zero and
    # whose Frobenius norm, 1e308, is representable. The row selection and the
    # summation are the same mathematics as before; only the accumulation is
    # done in the scaled domain this module already uses everywhere else.
    trace_rows = np.arange(0, n, d + 1, dtype=int)
    defect_vector = scaled_column_sums(L_super[trace_rows, :])
    return scaled_euclidean_norm(defect_vector), scaled_euclidean_norm(L_super)


def trace_preservation_componentwise_error(L_super: np.ndarray) -> float:
    """Return the worst componentwise backward error of ``vec(I)^H L = 0``.

    The global ratio ``||vec(I)^H L|| / ||L||`` is necessary but insufficient
    as an applicability test: entries of ``L`` that do not participate in a
    particular trace equation can make the global operator norm enormous and
    thereby hide an order-one violation in another column. Trace preservation
    is instead a family of scalar equations, one for every superoperator
    column ``j``:

    ``sum_i L[(i, i), j] = 0``.

    For each equation we therefore compute
    ``abs(sum terms) / sum(abs(terms))`` and return the largest ratio. This is
    the componentwise relative backward-error reading: every equation is
    normalised only by the represented coefficients that can actually cancel
    in that equation. :func:`scaled_cancellation_ratio` evaluates the ratio in
    a power-of-two scaled domain, so the verdict is stable for subnormal and
    near-overflow rate units. An exact all-zero equation contributes zero.
    """
    arr = np.asarray(L_super)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        return float("nan")
    if not np.all(np.isfinite(arr)):
        return float("nan")
    n = arr.shape[0]
    d = int(round(np.sqrt(n)))
    if d * d != n:
        return float("nan")
    trace_rows = np.arange(0, n, d + 1, dtype=int)
    return float(
        max(
            (scaled_cancellation_ratio(arr[trace_rows, j]) for j in range(n)),
            default=0.0,
        )
    )


def underflow_safe_norm(v: np.ndarray) -> float:
    """2-norm of ``v`` that does not lose a nonzero vector to underflow.

    ROUND-24 REVIEW (PR #121), finding B4. ``np.linalg.norm`` squares before
    summing, so every component below ``sqrt(5e-324) ~ 1.5e-162`` contributes
    exactly zero and a demonstrably nonzero vector comes back with norm
    ``0.0``. That is not an accuracy loss but a change of KIND: a residual of
    zero certifies an exact eigenpair, which is the strongest statement the
    a-posteriori bound can make.

    Same discipline as :func:`trace_preservation_defect` above: the rescue is
    reachable ONLY from an exact ``0.0`` on a vector that is not itself zero,
    so no overflowing quantity is made finite and a genuinely zero vector
    keeps the correct answer rather than a rescued one. Scaling is by a POWER
    OF TWO via ``frexp``/``ldexp``, so the mantissa bits survive the round trip
    exactly and complex division by a subnormal -- which overflows inside
    NumPy's algorithm -- is never formed.
    """
    v = np.asarray(v)
    nrm = float(np.linalg.norm(v))
    if nrm != 0.0 or not v.size or not bool(np.any(v != 0.0)):
        return nrm
    m = float(np.max(np.maximum(np.abs(np.real(v)), np.abs(np.imag(v)))))
    if not np.isfinite(m) or m <= 0.0:  # pragma: no cover - unreachable via any(v != 0)
        return nrm
    exp = int(np.frexp(m)[1])
    scaled = np.ldexp(np.real(v), -exp) + 1j * np.ldexp(np.imag(v), -exp)
    return float(np.ldexp(float(np.linalg.norm(scaled)), exp))


def underflow_safe_column_norms(M: np.ndarray) -> np.ndarray:
    """Column-wise 2-norms of ``M``, none of which underflows to a false zero.

    ROUND-25 REVIEW (PR #121). :func:`underflow_safe_norm` repaired the
    residual pair inside ``certify_nonstationary``; the SEPARATE eigenvector
    gate in :func:`certified_eig` still formed its column norms with a raw
    ``np.linalg.norm(..., axis=0)``, so the same 1e-162 cliff produced a
    residual of ``0.0`` -- an exact-eigenpair claim -- for a demonstrably
    wrong eigenvector, and D9/D19 consumed a pair the gate exists to reject.

    Repairing one call site rather than the CLASS is what let the defect
    survive a round; this helper is the one place the rescue now lives for
    every column-wise residual.

    The healthy path is unchanged BIT FOR BIT: the vectorised NumPy result is
    returned unless a column came back exactly ``0.0`` while containing a
    nonzero entry, and only those columns are recomputed. A genuinely zero
    column keeps its correct ``0.0`` rather than a rescued value.
    """
    M = np.asarray(M)
    nrm = np.asarray(np.linalg.norm(M, axis=0), dtype=float)
    if not M.size:
        return nrm
    lost = (nrm == 0.0) & np.any(M != 0.0, axis=0)
    if not bool(np.any(lost)):
        return nrm
    out = np.array(nrm, dtype=float)
    for k in np.flatnonzero(lost):
        out[int(k)] = underflow_safe_norm(M[:, int(k)])
    return out


def band_discriminates(
    magnitudes: np.ndarray, zero_count: int, bound: float
) -> bool:
    """Did the zero-mode band decide anything, or accept everything?

    Round-20 review (PR #121). ``residual <= bound`` certifies that a zero
    mode was found. It says nothing about whether the band SEPARATED that mode
    from the rest, and on the reviewer's counterexample it did not: a finite
    4x4 with cancelling ``+-1e308`` entries has ``||L||_2 = 1.4e308``, hence a
    band of ``3.1e295``, and all four eigenvalues ``{1, 2, 3, 4}`` fall inside
    it. The certificate came back ``certified=True, resolved=True`` with
    ``zero_mode_count = 4`` -- a claim that the ENTIRE space is stationary,
    for a generator whose spectrum is nothing of the kind.

    An acceptance region that contains every measured value has discriminated
    nothing; passing it is not evidence. Note this is NOT a tightened
    threshold -- tightening ``rtol`` would move the same band, not repair its
    logic -- but a second, structural axis: the band must have had the
    OPPORTUNITY to reject.

    The one legitimate way for a band to contain the whole spectrum is
    ``bound == 0``: the exactly-zero generator, whose eigenvalues are exactly
    zero and for which "everything is stationary" is the correct physics
    measured exactly. A band of width zero accepts only exact zeros, so it
    discriminates by construction and is excluded here.
    """
    if bound <= 0.0:
        return True
    return zero_count < int(magnitudes.size)


#: Relative distance at which a borrowed decomposition still counts as talking
#: about the SAME mode. It is the pre-existing agreement guard of
#: :func:`certified_nonzero_modes`, lifted to a name because the cluster and
#: subspace paths must apply exactly the same criterion -- a second, silently
#: different tolerance is how the two halves of a fail-closed rule drift apart.
PAIR_AGREEMENT_RTOL: float = 0.1

#: How much farther the nearest NON-member reference eigenvalue must sit than
#: the farthest member before a cluster counts as separable. Below this the
#: ball that defines the invariant subspace cannot be placed unambiguously and
#: the verdict is ``unresolved`` rather than an optimistic guess.
SUBSPACE_SEPARATION_FACTOR: float = 4.0


def _frobenius_underflow_safe(M: np.ndarray) -> float:
    """Frobenius norm that does not collapse to a false ``0.0``.

    Round-24 finding B4 applies verbatim to the block residuals below: a
    uniformly tiny residual matrix squares to zero under ``np.linalg.norm`` and
    would certify an arbitrarily bad subspace. Built from the column norms so
    the rescue lives in exactly one place (:func:`underflow_safe_column_norms`).
    """
    return underflow_safe_norm(underflow_safe_column_norms(np.asarray(M)))


def _injective_pairing(
    eigenvalues: np.ndarray, ref: np.ndarray, rows: np.ndarray
) -> dict[int, int]:
    """Globally injective candidate -> reference-index map, or ``{}``.

    ROUND-28 REVIEW (PR #121), the adjudicated repair. Which reference eigenPAIR
    may serve as evidence for candidate ``i`` used to be decided by an
    independent nearest-value lookup per candidate. A lookup is not a map: two
    candidates that are merely CLOSE both take the same reference index, and the
    second is then certified from a residual measured for the first. For a
    certification layer that is not a convention but a violation of evidence
    identity, so the assignment is made injective BY CONSTRUCTION.

    ``scipy.optimize.linear_sum_assignment`` minimises the TOTAL distance rather
    than resolving collisions in candidate order. That distinction is
    load-bearing and was measured: on the D11 network the greedy order hands
    ``ev[3] = -3.3216107e-06`` the exact partner of ``ev[8] = -3.345505e-06``
    and pushes ``ev[8]`` onto a stranger, while the global optimum leaves both
    exact matches intact and strands ``ev[3]`` on ``ref[3] = -4.239e-06`` --
    where the agreement guard refuses it, which is the correct fail-closed
    verdict for a candidate the reference spectrum has no room for.

    ``rows`` deliberately spans the whole zero band including the stationary
    mode: it must reserve its own near-zero reference entry, or a slow candidate
    could be handed the zero eigenvalue's eigenpair.

    Returns ``{}`` -- certify nothing -- if any distance is non-finite, because
    the assignment is then undefined and guessing would be fail-open.
    """
    if rows.size == 0:
        return {}
    ref_arr = np.asarray(ref)
    cost = np.abs(ref_arr[np.newaxis, :] - np.asarray(eigenvalues)[rows][:, np.newaxis])
    cost = np.asarray(cost, dtype=float)
    if not bool(np.all(np.isfinite(cost))) or cost.shape[1] < cost.shape[0]:
        return {}
    r_idx, c_idx = linear_sum_assignment(cost)
    return {int(rows[a]): int(j) for a, j in zip(r_idx, c_idx, strict=True)}


def _relative_clusters(values: np.ndarray, indices: np.ndarray) -> list[list[int]]:
    """Group ``indices`` whose ``values`` are within ``PAIR_AGREEMENT_RTOL``.

    Transitive closure over the relative-distance relation, computed on the
    magnitude-sorted order so a chain of near neighbours forms one group. A
    group of size one is a separated, non-degenerate candidate and takes the
    individual path; a group of size two or more is the near-degenerate cluster
    the invariant-subspace argument is about.
    """
    if indices.size == 0:
        return []
    vals = np.asarray(values)[indices]
    order = np.argsort(np.abs(vals), kind="stable")
    groups: list[list[int]] = []
    current: list[int] = [int(indices[order[0]])]
    for prev, cur in itertools.pairwise(order):
        a, b = complex(vals[prev]), complex(vals[cur])
        scale = max(abs(a), abs(b))
        if scale > 0.0 and abs(a - b) <= PAIR_AGREEMENT_RTOL * scale:
            current.append(int(indices[cur]))
        else:
            groups.append(current)
            current = [int(indices[cur])]
    groups.append(current)
    return groups


def _certify_invariant_subspace(
    L_c: np.ndarray, ref: np.ndarray, cluster_values: np.ndarray, margin: float
) -> bool:
    """Is the whole near-degenerate cluster provably away from the zero mode?

    ROUND-28 REVIEW (PR #121). LAPACK states the case this exists for: when
    eigenvalues cluster, the INVARIANT SUBSPACE can be well conditioned while
    the individual eigenvectors are not determined at all. The per-mode bound
    then divides by a ``|y^H x|`` that has collapsed and abstains on a cluster
    whose position in the spectrum is in fact known to full accuracy. Forcing
    individual eigenvectors there is the wrong object; the subspace is the right
    one, and it is certified from the OPERATOR's own Schur basis, never from a
    neighbour's eigenpair.

    The bound is the exact block analogue of the scalar one already used above,

        dist(spec, spec(T11)) <~ max(||L Q - Q T11||, ||L^H P - P S11||)
                                 / sigma_min(P^H Q)

    with ``Q``/``P`` orthonormal bases of the right/left invariant subspaces.
    ``sigma_min(P^H Q)`` is the cosine of the LARGEST PRINCIPAL ANGLE between
    them -- for ``k = 1`` it is ``|y^H x|`` and the whole expression reduces to
    the scalar bound, so the two paths cannot disagree about a lone mode. The
    same first-order caveat therefore applies and is covered by the same
    ``margin``; the residual is measured in the Frobenius norm, which dominates
    the spectral norm, so the bound errs towards refusing.

    Certification is ALL-OR-NOTHING over the cluster. The block bound places the
    ``k`` eigenvalues of ``T11`` in bijection with ``k`` eigenvalues of ``L``, so
    the statement "none of these is the stationary mode" is a statement about the
    group; splitting it would hand one member evidence gathered for the group.

    Every separability failure returns ``False`` -- ``unresolved``, nothing
    certified -- and never a fallback to a neighbouring eigenpair:

    * fewer than two members: a one-dimensional subspace is the scalar case,
      which the caller has already tried and which failed on its own evidence;
    * no reference cluster of the SAME multiplicity within the agreement guard;
    * the nearest non-member closer than ``SUBSPACE_SEPARATION_FACTOR`` times
      the cluster radius, so no ball selects the cluster unambiguously;
    * a Schur reordering that does not return exactly ``k`` selected modes;
    * a singular or non-finite ``P^H Q``, or a cluster containing an exact zero.
    """
    vals = np.asarray(cluster_values, dtype=complex)
    k = int(vals.size)
    if k < 2 or not bool(np.all(np.isfinite(vals))):
        return False
    centre = complex(np.mean(vals))
    ref_arr = np.asarray(ref, dtype=complex)
    if ref_arr.size <= k or not bool(np.all(np.isfinite(ref_arr))):
        return False

    dist = np.sort(np.abs(ref_arr - centre))
    inner, outer = float(dist[k - 1]), float(dist[k])
    # The reference spectrum must carry a cluster of the SAME multiplicity that
    # agrees with this one; otherwise the candidate ladder is claiming more
    # modes at this position than the operator has, which is precisely the D11
    # situation and must stay unresolved.
    if not (inner <= PAIR_AGREEMENT_RTOL * abs(centre)):
        return False
    if inner > 0.0:
        if not (outer > SUBSPACE_SEPARATION_FACTOR * inner):
            return False
    elif not (outer > 0.0):
        return False
    radius = 0.5 * (inner + outer)
    if not np.isfinite(radius) or radius <= 0.0:
        return False

    A = np.asarray(L_c, dtype=complex)
    try:
        T_r, Z_r, sdim_r = sla.schur(
            A, output="complex", sort=lambda x: bool(abs(x - centre) <= radius)
        )
        T_l, Z_l, sdim_l = sla.schur(
            A.conj().T,
            output="complex",
            sort=lambda x: bool(abs(x - np.conj(centre)) <= radius),
        )
    except (ValueError, sla.LinAlgError):
        return False
    if int(sdim_r) != k or int(sdim_l) != k:
        return False

    Q, P = Z_r[:, :k], Z_l[:, :k]
    T11, S11 = T_r[:k, :k], T_l[:k, :k]
    mu = np.asarray(np.diag(T11), dtype=complex)
    if not bool(np.all(np.isfinite(mu))) or bool(np.any(mu == 0.0)):
        return False

    # The cluster the Schur basis selected must be the cluster the candidates
    # describe -- checked injectively, for the same reason the outer pairing is.
    agree = np.abs(mu[np.newaxis, :] - vals[:, np.newaxis])
    scale = np.maximum(np.abs(vals)[:, np.newaxis], np.abs(mu)[np.newaxis, :])
    a_idx, b_idx = linear_sum_assignment(np.asarray(agree, dtype=float))
    if bool(np.any(agree[a_idx, b_idx] > PAIR_AGREEMENT_RTOL * scale[a_idx, b_idx])):
        return False

    overlap = P.conj().T @ Q
    if not bool(np.all(np.isfinite(overlap))):
        return False
    sep = float(np.min(np.linalg.svd(overlap, compute_uv=False)))
    if not np.isfinite(sep) or sep <= 0.0:
        return False
    res = max(
        _frobenius_underflow_safe(A @ Q - Q @ T11),
        _frobenius_underflow_safe(A.conj().T @ P - P @ S11),
    )
    bound = res / sep
    if not np.isfinite(bound):
        return False
    smallest = min(float(np.min(np.abs(vals))), float(np.min(np.abs(mu))))
    return smallest > margin * bound


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
    borrowed = vr is None or vl is None
    if vr is None or vl is None:
        # Own decomposition: the certificate is a property of the OPERATOR, so
        # it is legitimate to certify a candidate ladder's spectrum from the
        # operator's own eigenpairs -- but only where the two agree.
        try:
            ref, vl, vr = sla.eig(L_c, left=True, right=True)
        except (ValueError, sla.LinAlgError):
            return out

    # ROUND-26 REVIEW (PR #121). Which eigenPAIR belongs to candidate ``i`` was
    # decided by ``argmin(|ref - lam|)``, a lookup BY VALUE. On a degenerate
    # spectrum that is not a bijection: ``argmin`` returns the FIRST index
    # attaining the minimum, so two equal in-band eigenvalues both borrow the
    # first one's vectors and the second mode is certified on evidence that
    # was never measured for it. Measured on a trace-preserving 4x4 with
    # spectrum ``{0, -1e-14, -1e-14, -1}`` whose SECOND slow right vector was
    # replaced by the fast mode's vector (residual 1.0 rather than 6e-18):
    # both modes came back certified, ``certified=True, resolved=True``, and
    # ``certified_eig`` handed the invalid pair to D9/D19. The later
    # ``_vector_residual`` gate does not catch it -- it tests only modes above
    # the RAW ``bound``, and a rescued mode is by construction below it.
    #
    # Two cases, and they need different repairs:
    #
    # * vectors SUPPLIED -> ``ref is eigenvalues``, so the pair index for
    #   candidate ``i`` IS ``i``. No search is needed or wanted. For a
    #   non-degenerate spectrum this is bit for bit the old ``argmin``
    #   (``|ref[i] - lam| == 0`` is the unique minimum); it differs only where
    #   the old lookup was ambiguous, which is exactly the defect.
    # * vectors BORROWED -> ``ref`` really is a different decomposition and a
    #   search is unavoidable. It is then a GLOBAL INJECTIVE ASSIGNMENT
    #   (``_injective_pairing``), not a nearest-neighbour lookup per candidate.
    #
    # ROUND-28 REVIEW (PR #121): the adjudicated resolution of the collision an
    # earlier round recorded as "left open deliberately". The equal-value
    # ranking that stood here keyed on EXACT float equality, so two merely
    # CLOSE candidates both took ``argmin`` and landed on the same reference
    # eigenpair -- the same defect one step away from exact degeneracy, and
    # reachable on a fixture this suite carries. Measured on the D11 network of
    # ``tests/test_pr121_review_round17.py``: candidates
    # ``{-3.3216107e-06, -3.345505e-06, -3.345505e-06}`` against a reference
    # spectrum carrying ``-3.345505e-06`` EXACTLY TWICE (``ref[8]``,
    # ``ref[11]``). Both copies are claimed to the last bit by the two exact
    # candidates; ``ev[3]`` has no counterpart left, and its certification
    # rested entirely on ``ref[8]``'s residual -- a residual already spoken for,
    # and measured for a different eigenvalue.
    #
    # That is not a choosable convention. A residual is evidence for ONE
    # eigenpair, so a many-to-one assignment cannot certify a second candidate;
    # it must abstain. The three-way rule now reads:
    #
    #   separated, non-degenerate  -> index identity (same decomposition) or a
    #                                 globally injective assignment (borrowed),
    #                                 never greedy nearest-neighbour;
    #   near-degenerate cluster    -> certify the INVARIANT SUBSPACE from the
    #                                 operator's own Schur basis, because
    #                                 LAPACK's own caveat applies: a clustered
    #                                 subspace can be well conditioned while its
    #                                 individual eigenvectors are not determined;
    #   not separable              -> ``unresolved``: nothing certified, no
    #                                 borrowed residual.
    #
    # The global optimum is what makes the injective rule safe here. An earlier
    # round measured a BIJECTION as harmful, but that measurement was of a
    # greedy one -- resolving collisions in candidate order hands ``ev[3]`` the
    # exact partner of ``ev[8]``. Minimising the TOTAL distance instead leaves
    # both exact matches intact and strands ``ev[3]`` on ``ref[3] =
    # -4.239e-06`` at a relative distance of 0.216, where the agreement guard
    # below refuses it. ``ev[8]`` and ``ev[11]`` keep their own evidence and
    # stay certified; only the mode with no evidence of its own is lost, which
    # is the point.
    #
    # The cost for the D11 fixture is stated rather than hidden: the certificate
    # there goes ``resolved=True -> False`` and that network becomes a NEGATIVE
    # control. Conceding that a network does not resolve is the honest verdict
    # when the alternative is a borrowed residual; the positive path is carried
    # by a separable fixture instead.
    #
    # Bit-for-bit on the healthy path: where every candidate has a unique
    # nearest reference entry and no two compete for it, the minimum-cost
    # assignment IS that set of nearest pairs, so a non-degenerate,
    # well-separated spectrum takes exactly the old code path.
    pair_of: dict[int, int] = {}
    clusters: list[list[int]] = []
    cluster_of: dict[int, int] = {}
    ref_arr = np.asarray(ref)
    if borrowed:
        pair_of = _injective_pairing(eigenvalues, ref, np.flatnonzero(in_band))
        clusters = _relative_clusters(eigenvalues, idx)
        for pos, group in enumerate(clusters):
            for member in group:
                cluster_of[member] = pos
    else:
        pair_of = {int(raw): int(raw) for raw in np.flatnonzero(in_band)}

    tiny = np.finfo(float).tiny
    for i in idx:
        lam = eigenvalues[i]
        if borrowed:
            if len(clusters[cluster_of[int(i)]]) > 1:
                # Near-degenerate: no individual eigenpair is the right evidence
                # object here. Handled below, as a subspace, or not at all.
                continue
            # A SEPARATED candidate may be certified only when the reference
            # spectrum offers exactly ONE eigenpair that agrees with it. Two
            # admissible partners mean the assignment is a coin toss between
            # two different residuals -- minimising total distance would still
            # return one of them, and on a tie it returns an arbitrary one.
            # Measured on the D11 network before this gate: the minimum-cost
            # assignment there is EXACTLY degenerate (two assignments both cost
            # 9.1739e-07, because one candidate matches a reference entry to
            # the last bit), and the arbitrary winner re-created the borrowed
            # residual the round removes. Ambiguous evidence is unresolved.
            adm = np.flatnonzero(
                np.abs(ref_arr - lam)
                <= PAIR_AGREEMENT_RTOL * np.maximum(np.abs(lam), np.abs(ref_arr))
            )
            if adm.size != 1 or int(i) not in pair_of or int(adm[0]) != pair_of[int(i)]:
                continue
        if int(i) not in pair_of:
            continue  # no eigenpair left to match: nothing measured, nothing certified
        j = pair_of[int(i)]
        mag = float(abs(ref[j]))
        # Fail closed when the borrowed decomposition disagrees about this mode:
        # a bound computed for a different eigenvalue certifies nothing.
        if abs(ref[j] - lam) > PAIR_AGREEMENT_RTOL * max(abs(lam), mag) or mag == 0.0:
            continue
        # ROUND-24 REVIEW (PR #121), finding B4. Every norm in this block is
        # underflow-safe. The residual pair is the one the reviewer named: for
        # a generator whose nonzero eigenpair residuals are uniformly scaled
        # below roughly 1e-162, ``np.linalg.norm`` squares the components to
        # zero and returns ``0.0`` for a nonzero vector. ``mag > margin *
        # (0 / sep)`` is then true for EVERY nonzero in-band candidate, so a
        # numerically perturbed member of a degenerate stationary manifold is
        # promoted into the physical spectrum and D1 reports a spurious gap --
        # solely because the rate units changed. Measured on a deliberately
        # inexact eigenpair (candidate magnitude 1e-3 of the operator scale, a
        # residual 100x ABOVE it, so the correct verdict is "not certified"):
        # NOT certified at c = 1, 1e-80 and 1e-160; certified at c = 1e-170
        # and 1e-200. The refined midpoint below was already made
        # underflow-safe in round 22; the residual that ESTABLISHES the
        # refinement was not.
        #
        # The two normalisations are included for the same reason and in the
        # same direction: a right/left vector whose own norm underflows would
        # be divided by ``tiny`` and leave the unit sphere entirely, so ``x``
        # and ``y`` would no longer be the unit vectors the bound assumes.
        x = vr[:, j] / max(underflow_safe_norm(vr[:, j]), tiny)
        y = vl[:, j] / max(underflow_safe_norm(vl[:, j]), tiny)
        sep = float(abs(np.vdot(y, x)))
        if sep <= 0.0:
            continue  # defective pair: the first-order bound does not apply
        res = max(
            underflow_safe_norm(L_c @ x - ref[j] * x),
            underflow_safe_norm(L_c.conj().T @ y - np.conj(ref[j]) * y),
        )
        if mag > margin * (res / sep):
            out[i] = True

    # ROUND-28 REVIEW (PR #121), the second half of the adjudicated rule. The
    # loop above reasons about one eigenPAIR at a time, and that is the wrong
    # object exactly where LAPACK says it is: inside a tight eigenvalue cluster
    # the individual eigenvectors need not be determined at all -- ``|y^H x|``
    # collapses and the per-mode bound blows up -- while the INVARIANT SUBSPACE
    # they span is well conditioned and locates the cluster to full accuracy.
    # Such a cluster is certified as a subspace, from the operator's own Schur
    # basis, or not at all.
    #
    # A cluster takes this path ALWAYS, never as a rescue after the individual
    # path declined. An earlier draft of this repair made it a fallback -- try
    # each mode individually first, fall back to the subspace only if none
    # succeeded -- and that draft was MEASURED not to fix the defect: on the D11
    # network's ``dgeev-real`` route the two assignments
    # ``{ev[3]->ref[8], ev[11]->ref[3]}`` and ``{ev[3]->ref[3], ev[11]->ref[11]}``
    # cost exactly the same 9.1739e-07, so the individual path still certified
    # ``ev[3]`` from ``ref[8]``'s residual and the fallback never ran. Inside a
    # cluster the individual assignment is not merely imprecise, it is
    # UNDETERMINED, which is why the subspace is the object and the choice is
    # not offered.
    #
    # Restricted to the BORROWED route. When the caller supplies vectors, those
    # vectors are the evidence and are handed on to D9/D19 downstream, so the
    # certificate must vouch for THEM; a subspace certificate would clear
    # eigenvectors that were never examined. There the pair index is the
    # candidate's own index and no search happens at all.
    if borrowed:
        for group in clusters:
            if len(group) < 2:
                continue
            if _certify_invariant_subspace(
                L_c, ref, np.asarray(eigenvalues)[group], margin
            ):
                for g in group:
                    out[g] = True
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
    if lo <= 0.0:
        return in_band, hi * 0.5
    # ROUND-22 REVIEW (PR #121). ``sqrt(lo * hi)`` forms the PRODUCT first,
    # and that product underflows to exactly 0 whenever ``lo * hi`` drops
    # below ~5e-324 -- while ``lo``, ``hi`` and their geometric mean are all
    # perfectly representable. The tolerance then becomes 0.0, the strict
    # ``|lambda| > tol`` filters keep the numerical stationary residual as a
    # physical mode, and D1 reports it as the gap. Since ``lo * hi`` scales as
    # c^2 under a uniform rate rescale ``L -> cL`` while the refinement itself
    # is scale free, the failure is purely a units artefact: the same physics
    # in smaller rate units silently changes the answer.
    #
    # ``sqrt(lo) * sqrt(hi)`` is the same number mathematically and forms each
    # factor separately, so it can neither underflow (sqrt of the smallest
    # subnormal is ~2.2e-162) nor overflow (sqrt of the largest float is
    # ~1.3e154, and the product of two such factors is at most the largest
    # float again).
    return in_band, float(np.sqrt(lo) * np.sqrt(hi))


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
    # ROUND-22 REVIEW (PR #121). A tolerance that is not a number cannot
    # decide anything. With ``tp_rtol = NaN`` the trace-preservation test
    # ``defect > tp_rtol * fro`` is False for EVERY operator (all NaN
    # comparisons are), and with ``tp_rtol = inf`` its right-hand side is
    # infinite -- so a demonstrably non-trace-preserving generator walks
    # straight past the applicability gate and can come back
    # ``applicable=True, certified=True``. Measured before this guard:
    # ``certified_eigvals(diag([0,-1,-2,-3]), tp_rtol=nan)`` reported
    # applicable and certified with a trace defect of 3.0.
    #
    # Round 21 closed exactly this shape one variable further along (a
    # non-finite REFERENCE scale ``fro``) and left the PARAMETER beside it
    # unchecked. The generalisation is the fix: every tolerance this layer
    # compares against is validated the way ``operator_zero_tolerance`` and
    # ``spectral_zero_tolerance`` already validate theirs.
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if not np.isfinite(tp_rtol) or tp_rtol < 0.0:
        raise ValueError(f"tp_rtol must be finite and non-negative, got {tp_rtol}")
    L_super = require_finite_square_2d(L_super, name="L_super")
    L_c = np.asarray(L_super, dtype=complex)
    tp_defect, fro = trace_preservation_defect(L_c)
    tp_componentwise = trace_preservation_componentwise_error(L_c)
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
    if (
        not np.isfinite(tp_defect)
        # ROUND-21: a non-finite REFERENCE scale is as unusable as a
        # non-finite defect. ``defect <= tp_rtol * fro`` against an
        # infinite ``fro`` admits every operator, so the band that
        # follows is drawn from a scale that was never measured; which
        # modes it then swallows is decided by the LAPACK build (2 of 4
        # on the CI runners, 4 of 4 locally). Trace preservation
        # RELATIVE to the operator's own scale is not a statement that
        # can be made about infinity, so it is refused here.
        or not np.isfinite(fro)
        # ISSUE #130. The normwise ratio above can be diluted by enormous
        # entries that do not participate in a violated trace equation. The
        # componentwise gate normalises each column only by the diagonal-output
        # terms that can cancel in that equation, then takes the worst column.
        # Both views must pass before the exact zero-mode theorem is applicable.
        or not np.isfinite(tp_componentwise)
        or tp_componentwise > tp_rtol
        # ROUND-23 REVIEW (PR #121), same finding one line further down.
        # ``max(fro, tiny)`` floors the REFERENCE SCALE at a constant that
        # is larger than the operator whenever the operator is subnormal,
        # so the relative test stops being relative: measured for
        # ``diag([0, -1e-320, -2e-320, -3e-320])``, defect 3e-320 against a
        # floor-derived bound of 2.2e-318, hence 'trace preserving' for an
        # operator whose relative defect is 0.80. Repairing the norms above
        # without this leaves the rate-unit dependence intact below ~1e-317.
        # The floor protected nothing: ``defect <= sqrt(d) * fro`` holds, so
        # ``fro == 0`` forces ``defect == 0`` and the comparison ``0 > 0`` is
        # False either way -- the exactly-zero generator stays applicable.
        or tp_defect > tp_rtol * fro
    ):
        # No certificate applies without trace preservation, so there is
        # nothing to repair TOWARDS and the ladder is not run: the primary
        # failure is the honest answer.
        if primary is None:
            assert primary_error is not None
            raise primary_error
        # ROUND-22 REVIEW (PR #121). ``zero_mode_count`` kept its dataclass
        # default of 1 here, which asserts a stationary mode that nothing
        # counted and nothing guarantees -- the whole point of
        # ``applicable=False`` is that no zero eigenvalue is implied. Both
        # certificate APIs reported ``zero_mode_count: 1`` for
        # ``diag([1,2,3,4])``, a spectrum containing no zero at all, and the
        # dict is persisted as audit metadata. 0 is the honest count: nothing
        # was certified as stationary.
        return primary, ZeroModeCertificate(
            applicable=False,
            certified=False,
            solver="zgeev",
            residual=float(np.min(np.abs(primary))) if primary.size else float("nan"),
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=0,
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
            if not band_discriminates(magnitudes, zero_count, bound):
                # ROUND-20: the band swallowed the whole spectrum, so this
                # route proved nothing. Carried on as an UNcertified
                # candidate, exactly like a route whose residual missed the
                # band -- a later route on a different deflation path may
                # still produce a spectrum the band can separate.
                if best is None or residual < best[2]:
                    best = (name, ev, residual)
                continue
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
    # ROUND-22 REVIEW (PR #121). A tolerance that is not a number cannot
    # decide anything. With ``tp_rtol = NaN`` the trace-preservation test
    # ``defect > tp_rtol * fro`` is False for EVERY operator (all NaN
    # comparisons are), and with ``tp_rtol = inf`` its right-hand side is
    # infinite -- so a demonstrably non-trace-preserving generator walks
    # straight past the applicability gate and can come back
    # ``applicable=True, certified=True``. Measured before this guard:
    # ``certified_eigvals(diag([0,-1,-2,-3]), tp_rtol=nan)`` reported
    # applicable and certified with a trace defect of 3.0.
    #
    # Round 21 closed exactly this shape one variable further along (a
    # non-finite REFERENCE scale ``fro``) and left the PARAMETER beside it
    # unchecked. The generalisation is the fix: every tolerance this layer
    # compares against is validated the way ``operator_zero_tolerance`` and
    # ``spectral_zero_tolerance`` already validate theirs.
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if not np.isfinite(tp_rtol) or tp_rtol < 0.0:
        raise ValueError(f"tp_rtol must be finite and non-negative, got {tp_rtol}")
    L_super = require_finite_square_2d(L_super, name="L_super")
    L_c = np.asarray(L_super, dtype=complex)
    tp_defect, fro = trace_preservation_defect(L_c)
    tp_componentwise = trace_preservation_componentwise_error(L_c)
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
    if (
        not np.isfinite(tp_defect)
        # ROUND-21: a non-finite REFERENCE scale is as unusable as a
        # non-finite defect. ``defect <= tp_rtol * fro`` against an
        # infinite ``fro`` admits every operator, so the band that
        # follows is drawn from a scale that was never measured; which
        # modes it then swallows is decided by the LAPACK build (2 of 4
        # on the CI runners, 4 of 4 locally). Trace preservation
        # RELATIVE to the operator's own scale is not a statement that
        # can be made about infinity, so it is refused here.
        or not np.isfinite(fro)
        # ISSUE #130. Keep the eigenvector ladder on the exact same
        # applicability contract as the eigenvalue-only ladder above.
        or not np.isfinite(tp_componentwise)
        or tp_componentwise > tp_rtol
        # ROUND-23 REVIEW (PR #121), same finding one line further down.
        # ``max(fro, tiny)`` floors the REFERENCE SCALE at a constant that
        # is larger than the operator whenever the operator is subnormal,
        # so the relative test stops being relative: measured for
        # ``diag([0, -1e-320, -2e-320, -3e-320])``, defect 3e-320 against a
        # floor-derived bound of 2.2e-318, hence 'trace preserving' for an
        # operator whose relative defect is 0.80. Repairing the norms above
        # without this leaves the rate-unit dependence intact below ~1e-317.
        # The floor protected nothing: ``defect <= sqrt(d) * fro`` holds, so
        # ``fro == 0`` forces ``defect == 0`` and the comparison ``0 > 0`` is
        # False either way -- the exactly-zero generator stays applicable.
        or tp_defect > tp_rtol * fro
    ):
        if primary is None:
            assert primary_error is not None
            raise primary_error
        # ROUND-22 REVIEW (PR #121). ``zero_mode_count`` kept its dataclass
        # default of 1 here, which asserts a stationary mode that nothing
        # counted and nothing guarantees -- the whole point of
        # ``applicable=False`` is that no zero eigenvalue is implied. Both
        # certificate APIs reported ``zero_mode_count: 1`` for
        # ``diag([1,2,3,4])``, a spectrum containing no zero at all, and the
        # dict is persisted as audit metadata. 0 is the honest count: nothing
        # was certified as stationary.
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
            zero_mode_count=0,
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
        # ROUND-25 REVIEW (PR #121). Both residuals and both normalisations go
        # through :func:`underflow_safe_column_norms`. Round 24 made the
        # residual pair inside ``certify_nonstationary`` underflow-safe and
        # left THIS gate on the raw norms, so the identical failure survived:
        # for a generator in rate units below ~1e-162 the squares underflow,
        # ``res_r`` reads ``0.0`` for a wrong eigenvector, nothing is
        # "offending", and ``certified=True`` hands D9/D19 the pair this gate
        # was built to withhold. The denominators are included for the same
        # reason as in round 24: a vector whose own norm underflows would be
        # divided by ``tiny`` and leave the unit sphere, so the per-mode
        # relative residual would no longer be the quantity being compared.
        res_r = underflow_safe_column_norms(
            L_c @ vr - vr * ev[None, :]
        ) / np.maximum(underflow_safe_column_norms(vr), tiny)
        res_l = underflow_safe_column_norms(
            L_c.conj().T @ vl - vl * np.conj(ev)[None, :]
        ) / np.maximum(underflow_safe_column_norms(vl), tiny)
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
            if not band_discriminates(magnitudes, zero_count, bound):
                # ROUND-20, same guard as the sibling ladder. The two loops
                # having drifted apart is what produced the round-16 finding;
                # this one is added to both in the same commit.
                if best is None or residual < best[2]:
                    best = (name, decomp, residual)
                continue
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

    Raises
    ------
    ValueError
        If ``rtol`` or ``atol`` is not finite and non-negative, or if the
        DERIVED tolerance ``rtol * max|A|`` is not finite. Round-24 review
        (PR #121): with ``rtol = inf`` every finite square matrix passed --
        ``is_hermitian([[0, 1], [0, 0]], rtol=float("inf"))`` returned ``True``
        -- because any finite defect is ``<= inf``. ``rtol`` is a newly exposed
        validation threshold, and an invalid threshold must not be able to turn
        a validation gate into a fail-open pass-through. The same rule the
        zero-mode tolerance helpers apply (:func:`liouscope.numerics.scale.
        spectral_zero_tolerance`) applies here, including their second line of
        defence on the derived value: a non-finite ``max|A|`` would reproduce
        the hole through the scale instead of through ``rtol``.

        Refusing rather than returning ``False`` is deliberate. ``False`` is a
        statement about the MATRIX ("not Hermitian"); an unusable tolerance is
        a statement about the CALL, and the two must not be reported as the
        same thing -- the non-square early return above is the former, this is
        the latter.

    Notes
    -----
    Zero-operator semantics: ``max|A| == 0`` gives ``tol = 0``, so the all-zero
    matrix (exactly Hermitian) passes and nothing else does. This mirrors the
    documented zero-operator behaviour of :func:`liouscope.numerics.scale.rate_scale`.
    """
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if atol is not None and (not np.isfinite(atol) or atol < 0.0):
        raise ValueError(f"atol must be finite and non-negative, got {atol}")
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    defect, scale = hermiticity_defect(A)
    tol = float(atol) if atol is not None else rtol * scale
    if not np.isfinite(tol):
        raise ValueError(
            "the Hermiticity tolerance is not finite "
            f"(rtol={rtol}, max|A|={scale}); a tolerance that accepts every "
            "defect cannot validate anything"
        )
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
