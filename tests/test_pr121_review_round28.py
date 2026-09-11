"""Round-28 review of PR #121: whose residual is this, exactly?

``certified_nonzero_modes`` decides that an in-band eigenvalue is genuinely
non-stationary by measuring a residual. A residual belongs to ONE eigenpair, so
the question "which reference eigenpair is the evidence for this candidate" has
to have exactly one answer, or the certificate is quoting a measurement made
about a different mode.

Three regimes, and the tests below pin each one plus its refusal:

* separated and non-degenerate -- one admissible reference eigenpair, assigned
  globally injectively; two admissible ones is an undetermined answer, not a
  close call, and must abstain;
* near-degenerate cluster -- the individual eigenvectors need not be determined
  at all, so the INVARIANT SUBSPACE is the evidence and it comes from the
  operator's own Schur basis;
* not separable -- ``unresolved``.

Every refusal here is paired with a positive control on the same fixture, so a
certificate that had simply stopped certifying could not pass this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope._consts import ZERO_MODE_APOSTERIORI_MARGIN, ZERO_MODE_EPS_FACTOR
from liouscope.numerics.linalg import (
    _certify_invariant_subspace,
    certified_nonzero_modes,
)

_SEED = 20260908


def _operator_with_spectrum(values: list[complex]) -> np.ndarray:
    """A deliberately non-normal operator with EXACTLY ``values`` as spectrum.

    The similarity transform is fixed by a seed rather than left as the
    identity: with orthogonal eigenvectors ``|y^H x|`` is 1 and the a posteriori
    bound stops discriminating, which would make every assertion below pass for
    the wrong reason.
    """
    n = len(values)
    rng = np.random.default_rng(_SEED)
    W = np.eye(n, dtype=complex) + 0.25 * rng.standard_normal((n, n))
    return np.linalg.inv(W) @ np.diag(np.asarray(values, dtype=complex)) @ W


def _band(L: np.ndarray) -> float:
    return ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * float(np.linalg.norm(L, 2))


# --------------------------------------------------------------------------
# Separated candidates: exactly one admissible eigenpair, or nothing
# --------------------------------------------------------------------------

_SLOW = 2.0e-13


def test_a_separated_candidate_with_two_admissible_references_abstains() -> None:
    """Two answers to "whose residual is this" is no answer at all.

    The candidate ladder offers one slow mode at ``S``; the operator's own
    spectrum carries TWO modes within the agreement guard of it, at ``0.95 S``
    and ``1.05 S``. Either could supply the residual and they are different
    residuals, so the evidence is undetermined and the mode stays in the band.

    Certifying it would be the borrowed-residual defect in its purest form: the
    number that ends up in the certificate would depend on a tie-break.
    """
    L = _operator_with_spectrum([0.0, 0.95 * _SLOW, 1.05 * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)
    assert in_band.tolist() == [True, True, False], "precondition: the band holds two"

    out = certified_nonzero_modes(L, candidates, in_band)
    assert bool(out[1]) is False


def test_a_separated_candidate_with_one_admissible_reference_is_certified() -> None:
    """POSITIVE CONTROL for the refusal above, on the same shape of fixture.

    The only change is that the second slow reference mode moves out of the
    agreement guard, so the answer becomes unique. If this failed too, the test
    above would be passing because the certificate had stopped working.
    """
    L = _operator_with_spectrum([0.0, _SLOW, 5.0 * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)

    out = certified_nonzero_modes(L, candidates, in_band)
    assert bool(out[1]) is True
    assert bool(out[0]) is False, "the stationary mode is never certified"


# --------------------------------------------------------------------------
# Degenerate clusters: the invariant subspace, from the operator itself
# --------------------------------------------------------------------------


def test_a_degenerate_cluster_is_certified_as_an_invariant_subspace() -> None:
    """The regime LAPACK warns about, and the reason the subspace path exists.

    Both slow modes sit at exactly ``_SLOW``, so "which eigenvector belongs to
    which" has no answer -- and it does not need one. The two-dimensional
    invariant subspace is well conditioned and locates both modes away from the
    stationary one, which is the entire claim the certificate makes.

    Note what is NOT asserted: nothing about the individual eigenvectors. That
    is the point of the round.
    """
    L = _operator_with_spectrum([0.0, _SLOW, _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)
    assert in_band.tolist() == [True, True, True, False]

    out = certified_nonzero_modes(L, candidates, in_band)
    assert [bool(out[1]), bool(out[2])] == [True, True]
    assert bool(out[0]) is False and bool(out[3]) is False


def test_a_cluster_crowded_by_a_third_mode_stays_unresolved() -> None:
    """Separability is a property of the SPECTRUM, not a wish of the caller.

    The candidate cluster ``{S, 1.05 S}`` would be certifiable, but the
    operator carries a third mode at ``1.10 S`` -- closer to the cluster than
    the cluster is wide. No ball selects the pair rather than the triple, so the
    invariant subspace the bound would be about is not defined and the verdict
    is ``unresolved``.

    Refusing here is what stops the block bound from quietly certifying two
    modes with a subspace that actually holds three.
    """
    L = _operator_with_spectrum([0.0, _SLOW, 1.05 * _SLOW, 1.10 * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, 1.05 * _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)
    assert in_band.tolist() == [True, True, True, False]

    out = certified_nonzero_modes(L, candidates, in_band)
    assert [bool(out[1]), bool(out[2])] == [False, False]


def test_the_same_cluster_with_the_third_mode_moved_away_is_certified() -> None:
    """POSITIVE CONTROL for the refusal above: only the crowding is removed.

    The third mode goes from ``1.10 S`` to ``1.5 S``. Everything else -- the
    similarity transform, the cluster, the band -- is identical, so a failure
    here would mean the refusal above is measuring the fixture, not the gate.
    """
    L = _operator_with_spectrum([0.0, _SLOW, 1.05 * _SLOW, 1.5 * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, 1.05 * _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)

    out = certified_nonzero_modes(L, candidates, in_band)
    assert [bool(out[1]), bool(out[2])] == [True, True]


def test_a_cluster_the_reference_spectrum_cannot_match_stays_unresolved() -> None:
    """The D11 shape, isolated: three candidates, two reference modes.

    The candidate ladder claims ``{S, S, 1.05 S}`` while the operator's own
    spectrum carries that value only twice and puts its third mode 27 per cent
    away. One candidate therefore has no evidence of its own, and because the
    three are mutually within the agreement guard there is no way to say WHICH
    one -- so the whole group abstains rather than two of them keeping a partner
    and the third borrowing.
    """
    L = _operator_with_spectrum([0.0, _SLOW, _SLOW, 1.27 * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, _SLOW, 1.05 * _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)
    assert in_band.tolist() == [True, True, True, True, False]

    out = certified_nonzero_modes(L, candidates, in_band)
    assert not bool(out.any()), "no member of an unmatchable cluster may be certified"


def test_a_cluster_member_with_its_own_exact_partner_still_abstains() -> None:
    """All-or-nothing is the block statement, not a rounding of it.

    The candidate ladder offers ``{S, 1.09 S}``; the operator has one mode at
    ``S`` and its next at ``1.5 S``. The first candidate matches ``S`` to the
    last bit and would certify on its own -- but its neighbour, one cluster
    away, has no counterpart at all, and the two are not separately determined.
    Certifying the one with a partner would split a group the bound only speaks
    about as a group.

    This is the D11 shape again, with the roles reduced to two: a ladder
    claiming more modes at a position than the operator has.
    """
    L = _operator_with_spectrum([0.0, _SLOW, 1.5 * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, 1.09 * _SLOW, -1.0], dtype=complex)
    in_band = np.abs(candidates) <= _band(L)
    assert in_band.tolist() == [True, True, True, False]

    out = certified_nonzero_modes(L, candidates, in_band)
    assert [bool(out[1]), bool(out[2])] == [False, False]


# --------------------------------------------------------------------------
# The supplied-vector route is deliberately NOT changed
# --------------------------------------------------------------------------


def test_supplied_vectors_still_certify_by_index_identity() -> None:
    """A caller's own decomposition is evidence for itself, degenerate or not.

    Those vectors are handed to D9/D19 downstream, so the certificate has to
    vouch for THEM; a subspace certificate would clear eigenvectors that were
    never examined. Asserted with a deliberately corrupted second slow vector:
    the sound mode is certified, the corrupt one is not, and the degeneracy does
    not merge them.
    """
    lam = np.array([0.0, _SLOW, _SLOW, -1.0], dtype=complex)
    rng = np.random.default_rng(_SEED)
    W = np.eye(4, dtype=complex) + 0.25 * rng.standard_normal((4, 4))
    V = np.linalg.inv(W)
    L = V @ np.diag(lam) @ W
    vr = V.copy()
    vr[:, 2] = V[:, 3]  # the fast mode's vector, wearing the slow mode's label
    x = vr[:, 2] / np.linalg.norm(vr[:, 2])
    assert float(np.linalg.norm(L @ x - lam[2] * x)) > 0.1, "precondition: badly wrong"

    out = certified_nonzero_modes(
        L, lam, np.abs(lam) <= _band(L), right_vectors=vr, left_vectors=np.conj(W).T
    )
    assert [bool(out[1]), bool(out[2])] == [True, False]


def test_the_healthy_separated_path_is_bit_for_bit_the_old_verdict() -> None:
    """Structural control: a well-separated spectrum is untouched by the round.

    Diagonal, so every reference eigenvalue is its candidate's unique admissible
    partner and the minimum-cost assignment is the set of exact matches. If this
    ever changed, the round would have altered the healthy path while claiming
    not to.
    """
    scale = 1.0e-14
    diag = np.array([0.0, scale, 4.0 * scale, 1.0], dtype=complex)
    L = np.diag(diag)
    out = certified_nonzero_modes(L, diag, np.array([True, True, True, False]))
    assert [bool(v) for v in out] == [False, True, True, False]


@pytest.mark.parametrize("shift", [0.0, 5.0e-2, 5.0e-1])
def test_certification_is_never_manufactured_for_the_stationary_mode(
    shift: float,
) -> None:
    """Whatever the regime, the smallest in-band mode is never certified.

    Trace preservation guarantees a zero eigenvalue exists; a certificate that
    emptied the band would contradict its own precondition. Swept across the
    three regimes -- exactly degenerate, near-degenerate, separated -- because
    the subspace path is a second place this invariant could be lost.
    """
    L = _operator_with_spectrum([0.0, _SLOW, (1.0 + shift) * _SLOW, -1.0])
    candidates = np.array([0.0, _SLOW, (1.0 + shift) * _SLOW, -1.0], dtype=complex)
    out = certified_nonzero_modes(L, candidates, np.abs(candidates) <= _band(L))
    assert bool(out[0]) is False


# --------------------------------------------------------------------------
# The subspace certificate's own bookkeeping guards
#
# The tests above reach ``_certify_invariant_subspace`` through
# ``certified_nonzero_modes``, where ``ref`` is by construction the operator's
# OWN ``sla.eig`` spectrum. Two of the refusals the function documents can then
# never be the sole cause: another guard downstream refuses the same fixture
# first. That was measured, not assumed -- a retraction probe over the round-28
# repair (ledger DK-20260907T225012-a908bed7de33, 4 of 6) reported exactly those
# two as BLIND, i.e. removable without any test noticing.
#
# ``ref``, ``L_c`` and ``cluster_values`` are three independent parameters, and
# the guards exist for the case where they disagree -- a candidate ladder solved
# by one route, a reference spectrum by another. They are therefore pinned at
# the function's own contract boundary, where each is the sole cause.
#
# The operators here are diagonal on purpose, the opposite choice from
# ``_operator_with_spectrum`` above and for the same reason: the bound must NOT
# be what refuses, or the assertion would pass for the wrong reason. With an
# exact invariant subspace the residual is ~0 and every downstream check passes,
# so a verdict of ``False`` can only come from the guard under test -- which the
# shared positive control demonstrates by certifying the identical fixture.
# --------------------------------------------------------------------------

_SUB_SLOW = -1.0e-6
_SUB_CLUSTER = np.array([_SUB_SLOW, _SUB_SLOW], dtype=complex)
_SUB_OPERATOR = np.diag(np.array([0.0, _SUB_SLOW, _SUB_SLOW, -1.0], dtype=complex))


def test_a_reference_with_too_few_modes_at_that_position_stays_unresolved() -> None:
    """Reference multiplicity below the cluster's: unresolved, not certified.

    The candidate ladder reports two modes at ``-1e-6``; the reference spectrum
    carries one there and its next entry at ``-1.2e-6``. Claiming two modes
    where the reference has room for one is the D11 situation in miniature, and
    the group must stay unresolved rather than have one member certified from
    evidence gathered for the other.

    The distances are chosen so that every LATER guard would pass: the second
    reference entry sits at ``2e-7`` from the centre, above the ``0.1``
    agreement guard's ``1e-7`` but far enough below the next distance
    (``1e-6``) to clear the factor-4 separation, and the Schur reordering still
    selects exactly the two operator modes.
    """
    ref = np.array([0.0, _SUB_SLOW, 1.2 * _SUB_SLOW, -1.0], dtype=complex)
    assert (
        _certify_invariant_subspace(_SUB_OPERATOR, ref, _SUB_CLUSTER, ZERO_MODE_APOSTERIORI_MARGIN)
        is False
    )


def test_a_cluster_the_operator_carries_more_of_stays_unresolved() -> None:
    """Schur selects k+1 modes for a k-cluster: unresolved, not a partial claim.

    The operator has a THREEFOLD cluster at ``-1e-6`` while the candidate group
    names two of its members. The leading columns of a Schur factorisation span
    an invariant subspace for ANY truncation, so the residual would be ~0 and
    the bound would certify happily -- on a two-dimensional slice cut out of a
    three-dimensional cluster. That is the "splitting the group" the docstring
    forbids: the block bound puts ``k`` eigenvalues of ``T11`` in bijection with
    ``k`` of ``L``, and which two of the three were selected is not determined.
    """
    operator = np.diag(np.array([0.0, _SUB_SLOW, _SUB_SLOW, _SUB_SLOW, -1.0], dtype=complex))
    ref = np.array([0.0, _SUB_SLOW, _SUB_SLOW, -0.5, -1.0], dtype=complex)
    assert (
        _certify_invariant_subspace(operator, ref, _SUB_CLUSTER, ZERO_MODE_APOSTERIORI_MARGIN)
        is False
    )


def test_the_matching_reference_certifies_the_same_fixture() -> None:
    """Positive control for both refusals above.

    Same operator, same cluster, a reference that agrees in multiplicity: the
    certificate is granted. Without this, a ``_certify_invariant_subspace`` that
    had been broken into refusing everything would satisfy both tests above.
    """
    ref = np.array([0.0, _SUB_SLOW, _SUB_SLOW, -1.0], dtype=complex)
    assert (
        _certify_invariant_subspace(_SUB_OPERATOR, ref, _SUB_CLUSTER, ZERO_MODE_APOSTERIORI_MARGIN)
        is True
    )
