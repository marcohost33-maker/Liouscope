"""Issue #168: basis-invariant conditioning of repeated/defective zero clusters."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.numerics import conditioning as conditioning_module
from liouscope.numerics.conditioning import (
    CONDITIONING_ABSTAIN,
    CONDITIONING_BENIGN,
    CONDITIONING_CLUSTER_ONLY,
    CONDITIONING_LIMITED,
    CONDITIONING_UNAVAILABLE,
    _Resolution,
    _scalar_verdict,
    _schur_cluster_projector_condition,
    _smallest_singular_value,
    cluster_conditioning,
    eigenvalue_conditioning,
    zero_mode_conditioning,
)
from liouscope.numerics.linalg import certified_eigvals
from liouscope.numerics.traceless import trace_vector


def _basis_with_first(q: np.ndarray) -> np.ndarray:
    seed = np.eye(q.size, dtype=complex)
    seed[:, 0] = q
    basis, _ = np.linalg.qr(seed)
    phase = np.vdot(basis[:, 0], q)
    return np.asarray(basis * (phase / abs(phase)), dtype=complex)


def _issue168_pair(
    seed: int = 3, block: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The exact reproduction from issue #168 and a unitary re-expression.

    ``block`` replaces the Jordan block (in the trace-vector basis) and ``seed``
    the rotation, so the issue-#183 tests can sweep both.
    """
    q = trace_vector(2)
    basis = _basis_with_first(q)

    if block is None:
        block = np.zeros((4, 4), dtype=complex)
        block[1, 0] = 1.0
        block[2, 2] = -1.0
        block[3, 3] = -2.0
    plain = basis @ block @ basis.conj().T

    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(
        rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    )
    U = (
        np.outer(q, q.conj())
        + basis[:, 1:] @ V @ basis[:, 1:].conj().T
    )
    rotated = U @ plain @ U.conj().T
    np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=2e-15)
    np.testing.assert_allclose(U @ q, q, atol=2e-15)
    return plain, rotated


def test_issue168_defective_cluster_is_unitary_basis_invariant() -> None:
    """The same operator in two orthonormal bases must give one cluster reading."""
    plain, rotated = _issue168_pair()

    a = zero_mode_conditioning(plain, zero_tolerance=1.0e-6)
    b = zero_mode_conditioning(rotated, zero_tolerance=1.0e-6)

    assert a.available and b.available
    assert a.cluster_size == b.cluster_size == 2
    assert a.verdict == b.verdict == CONDITIONING_CLUSTER_ONLY
    assert a.reason == b.reason == "cluster_conditioning_only"
    assert a.reciprocal_condition == pytest.approx(
        b.reciprocal_condition, rel=2e-10, abs=2e-12
    )
    assert a.reciprocal_condition == pytest.approx(1.0, rel=2e-10, abs=2e-12)
    assert math.isnan(a.per_mode_reciprocal_condition)
    assert math.isnan(b.per_mode_reciprocal_condition)
    assert math.isnan(a.structural_forward_estimate)
    assert math.isnan(b.structural_forward_estimate)
    assert a.displacement_explained is None
    assert b.displacement_explained is None


def test_projector_2norm_is_not_lapack_trsen_frobenius_lower_bound() -> None:
    """Pin the distinction that was misstated in the earlier design.

    For T11=0, T22=diag(-1,-2), T12=diag(3,4), the Sylvester solution is
    R=diag(3,2).  Hence the exact reciprocal projector 2-norm is 1/sqrt(10),
    while xTRSEN's documented Frobenius lower bound is 1/sqrt(14).
    """
    T = np.zeros((4, 4), dtype=complex)
    T[0, 2] = 3.0
    T[1, 3] = 4.0
    T[2, 2] = -1.0
    T[3, 3] = -2.0

    exact = _schur_cluster_projector_condition(
        T,
        reference_eigenvalues=np.diag(T),
        cluster_indices=np.array([0, 1], dtype=np.intp),
    )
    assert exact == pytest.approx(1.0 / math.sqrt(10.0), rel=1e-12)

    schur, q = sla.schur(T, output="complex")
    selected = np.abs(np.diag(schur)) <= 1.0e-12
    trsen = sla.get_lapack_funcs("trsen", (schur,))
    _ts, _qs, _w, m, s_lower, _sep, info = trsen(
        np.asarray(selected, dtype=np.int32),
        schur,
        q,
        job="B",
        wantq=0,
        lwork=8,
    )
    assert info == 0 and m == 2
    assert s_lower == pytest.approx(1.0 / math.sqrt(14.0), rel=1e-12)
    assert exact > s_lower


def test_schur_projector_matches_eigenvector_subspace_on_semisimple_cluster() -> None:
    """Where a genuine eigenspace exists, the new and legacy formulas agree."""
    S = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.0, 1.0, 0.3, 0.2],
            [0.1, 0.0, 1.0, 0.4],
            [0.0, 0.2, 0.0, 1.0],
        ],
        dtype=complex,
    )
    L = S @ np.diag([0.0, 0.0, -1.0, -2.0]) @ np.linalg.inv(S)
    values, left, right = sla.eig(L, left=True, right=True)
    idx = np.flatnonzero(np.abs(values) <= 1.0e-8)
    assert idx.size == 2

    legacy = cluster_conditioning(right, left, idx)
    schur = _schur_cluster_projector_condition(
        L,
        reference_eigenvalues=values,
        cluster_indices=idx,
    )
    assert schur is not None
    assert schur == pytest.approx(legacy, rel=2e-10, abs=2e-12)


def test_schur_projector_refuses_same_size_different_reference_spectrum() -> None:
    """Cluster cardinality alone must not bind a projector to another spectrum."""
    T = np.diag([0.0, 0.0, -1.0, -2.0]).astype(complex)
    reference = np.array([0.0, 1.0e-8, -1.0, -2.0], dtype=complex)

    measured = _schur_cluster_projector_condition(
        T,
        reference_eigenvalues=reference,
        cluster_indices=np.array([0, 1], dtype=np.intp),
    )
    assert measured is None

def test_simple_stationary_mode_keeps_scalar_first_order_evidence() -> None:
    """The abstention is specific to multi-mode clusters, not a global retreat."""
    jump = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [jump])

    evidence = zero_mode_conditioning(L, zero_tolerance=1.0e-10)
    assert evidence.available
    assert evidence.cluster_size == 1
    assert evidence.reason == "ok"
    assert np.isfinite(evidence.per_mode_reciprocal_condition)
    assert np.isfinite(evidence.structural_forward_estimate)
    assert np.isfinite(evidence.solver_forward_estimate)
    assert evidence.verdict != CONDITIONING_CLUSTER_ONLY


# --------------------------------------------------------------------------
# PR #173 review: the DEFAULT call (no zero_tolerance) abstains
# --------------------------------------------------------------------------

#: Every field whose value depends on WHICH modes form the zero set.
_ZERO_SET_FIELDS = (
    "reciprocal_condition",
    "per_mode_reciprocal_condition",
    "spectrum_min_reciprocal_condition",
    "right_residual",
    "left_residual",
    "separation",
    "solver_forward_estimate",
    "structural_forward_estimate",
    "zero_tolerance",
)


def test_default_call_is_unitary_basis_invariant_on_a_split_defective_root() -> None:
    """Codex P2 on PR #173: the default path must not depend on the basis.

    Measured before the fix on exactly this pair: plain ``CLUSTER_ONLY`` with
    ``cluster_size=2``; the unitary re-expression ``BENIGN`` with
    ``cluster_size=1`` and ``reciprocal_condition=3.63e-08``, because ``?geev``
    splits the defective double root by ``~sqrt(eps)`` and a proximity grouping
    then sees two simple roots. Both must now give the same statement.
    """
    plain, rotated = _issue168_pair()
    a = zero_mode_conditioning(plain)
    b = zero_mode_conditioning(rotated)

    # Premise: the fixture really exercises the split -- one basis returns the
    # double root exactly, the other far outside any round-off band.
    assert a.observed_displacement == 0.0
    assert b.observed_displacement > 1.0e-9

    for x in (a, b):
        assert x.available is True
        assert x.verdict == CONDITIONING_ABSTAIN
        assert x.reason == "zero_tolerance_not_supplied"
        assert x.cluster_size == 0
        assert x.conditioning_limited is False
        assert x.displacement_explained is None
        for field in _ZERO_SET_FIELDS:
            assert math.isnan(getattr(x, field)), field
    # The one operator-level observation agrees across the bases.
    assert a.trace_defect == pytest.approx(b.trace_defect, abs=1.0e-15)
    # Serialised claims are identical, and valid JSON (RFC 8259: no NaN).
    da, db = a.as_dict(), b.as_dict()
    claim_keys = (
        "available",
        "reason",
        "verdict",
        "cluster_size",
        "conditioning_limited",
        "displacement_explained",
        *_ZERO_SET_FIELDS,
    )
    for key in claim_keys:
        assert da[key] == db[key], key
    json.dumps(da, allow_nan=False)


def test_default_call_never_reports_benign_for_a_non_trace_preserving_operator() -> None:
    """Audit finding on PR #173: ``None`` became an internal NaN cutoff.

    ``diag(0, -1, -1, -2)`` is not trace preserving (defect ``1.414``) and the
    default call returned ``BENIGN`` -- the only scalar answer a NaN cutoff
    could produce. It now abstains; with a cutoff supplied the same operator
    is reported for what it is.
    """
    L = np.diag([0.0, -1.0, -1.0, -2.0]).astype(complex)
    default = zero_mode_conditioning(L)
    assert default.verdict == CONDITIONING_ABSTAIN
    assert default.reason == "zero_tolerance_not_supplied"
    assert default.cluster_size == 0
    assert default.trace_defect == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert math.isnan(default.structural_forward_estimate)

    supplied = zero_mode_conditioning(L, zero_tolerance=1.0e-6)
    assert supplied.verdict == CONDITIONING_LIMITED
    assert supplied.conditioning_limited is True


@pytest.mark.parametrize("seed", range(12))
def test_no_default_call_is_ever_benign(seed: int) -> None:
    """Negative sweep: whatever the operator, a call without a cutoff abstains
    or is unavailable -- it never accepts."""
    rng = np.random.default_rng(seed)
    side = (4, 9, 16)[seed % 3]
    A = rng.standard_normal((side, side)) + 1j * rng.standard_normal((side, side))
    if seed % 2:
        A = A - np.diag(np.diag(A))  # push roots toward zero
    evidence = zero_mode_conditioning(A)
    assert evidence.verdict in (CONDITIONING_ABSTAIN, CONDITIONING_UNAVAILABLE)
    assert evidence.verdict != CONDITIONING_BENIGN


@pytest.mark.parametrize(
    ("budget", "tol", "expected"),
    [
        (0.5, 1.0, CONDITIONING_BENIGN),
        (1.0, 1.0, CONDITIONING_BENIGN),  # the boundary is an acceptance
        (0.0, 0.0, CONDITIONING_BENIGN),
        (2.0, 1.0, CONDITIONING_LIMITED),
        (math.inf, 1.0, CONDITIONING_LIMITED),
        (math.nan, 1.0, CONDITIONING_LIMITED),  # NaN never lands in BENIGN
        (0.5, math.nan, CONDITIONING_ABSTAIN),
        (math.nan, math.nan, CONDITIONING_ABSTAIN),
        (0.5, math.inf, CONDITIONING_ABSTAIN),
    ],
)
def test_scalar_verdict_accepts_only_on_a_true_comparison(
    budget: float, tol: float, expected: str
) -> None:
    """The ACCEPTANCE (``BENIGN``) is bound to ``budget <= tol`` being TRUE."""
    assert _scalar_verdict(budget, tol) == expected


# --------------------------------------------------------------------------
# Issue #183: a FINITE cutoff below the defective split
# --------------------------------------------------------------------------

#: The cutoffs from the issue table (layer cutoff ``4.44e-13``, ``1e-12``,
#: ``1e-9``, ``>= 1e-7``) plus the gaps between them, the exact-zero cutoff and
#: the band around the ``~1.5e-08`` split itself.
_ISSUE183_CUTOFFS = (
    0.0,
    1.0e-16,
    4.44e-13,
    1.0e-12,
    1.0e-9,
    1.0e-8,
    2.0e-8,
    3.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-3,
    0.5,
)


@pytest.mark.parametrize("tol", _ISSUE183_CUTOFFS)
def test_issue183_finite_cutoff_is_unitary_basis_invariant(tol: float) -> None:
    """The invariance probe of the default-call test, over finite cutoffs.

    Measured before the fix: for every cutoff below the split (``0`` to
    ``1e-8``) the plain basis gave ``CLUSTER_ONLY`` with two modes and the
    rotated one ``CONDITIONING_LIMITED`` with one mode and
    ``reciprocal_condition = 3.07e-08``. The rotated basis now closes its group
    over the pair the arithmetic cannot separate, as the plain basis did.
    """
    plain, rotated = _issue168_pair()
    a = zero_mode_conditioning(plain, zero_tolerance=tol)
    b = zero_mode_conditioning(rotated, zero_tolerance=tol)

    # Premise: the rotated basis really carries the split.
    assert b.observed_displacement > 1.0e-9

    da, db = a.as_dict(), b.as_dict()
    for key in (
        "available",
        "reason",
        "verdict",
        "cluster_size",
        "conditioning_limited",
        "displacement_explained",
        "per_mode_reciprocal_condition",
        "solver_forward_estimate",
        "structural_forward_estimate",
        "zero_tolerance",
    ):
        assert da[key] == db[key], key
    assert a.verdict == CONDITIONING_CLUSTER_ONLY
    assert a.cluster_size == 2
    assert a.reciprocal_condition == pytest.approx(
        b.reciprocal_condition, rel=2e-10, abs=2e-12
    )
    json.dumps(db, allow_nan=False)


@pytest.mark.parametrize("seed", range(40))
def test_issue183_every_rotation_reports_the_same_cluster(seed: int) -> None:
    """Across rotations: one statement, including from the Schur gate.

    The independent Schur solve places a defective pair's ``+-sqrt(eps)`` roots
    differently from ``?geev``; with a mode-for-mode round-off agreement the
    projector was refused (``schur_cluster_mismatch``) in 6 of 400
    rotation/cutoff pairs -- an ``UNAVAILABLE`` in one basis and a
    ``CLUSTER_ONLY`` in another. Inside a cluster the arithmetic cannot
    resolve, agreement is now required only to that resolution.
    """
    plain, rotated = _issue168_pair(seed=seed)
    for tol in (0.0, 4.44e-13, 1.0e-9, 1.0e-7):
        for evidence in (
            zero_mode_conditioning(plain, zero_tolerance=tol),
            zero_mode_conditioning(rotated, zero_tolerance=tol),
        ):
            assert evidence.verdict == CONDITIONING_CLUSTER_ONLY
            assert evidence.cluster_size == 2
            assert evidence.reciprocal_condition == pytest.approx(
                1.0, rel=2e-10, abs=2e-12
            )


def test_issue183_displaced_defective_pair_is_kept_whole() -> None:
    """The same split, away from zero, on the EMPTY-cutoff fallback path.

    A Jordan pair at ``1e-5`` with a cutoff of ``1e-12``: no eigenvalue is
    admitted, so the group is the one nearest to zero. The plain basis ties
    the pair exactly (round-off band); a rotation splits it by ``~1.5e-08``,
    which the band cannot see but the pseudospectral test does.
    """
    block = np.zeros((4, 4), dtype=complex)
    block[0, 0] = block[1, 1] = 1.0e-5
    block[1, 0] = 1.0
    block[2, 2] = -1.0
    block[3, 3] = -2.0
    plain, rotated = _issue168_pair(seed=0, block=block)
    a = zero_mode_conditioning(plain, zero_tolerance=1.0e-12)
    b = zero_mode_conditioning(rotated, zero_tolerance=1.0e-12)

    values = np.sort(np.abs(np.linalg.eigvals(rotated) - 1.0e-5))
    assert values[1] > 1.0e-9  # premise: the rotation splits the pair
    for x in (a, b):
        assert x.verdict == CONDITIONING_CLUSTER_ONLY
        assert x.cluster_size == 2


@pytest.mark.parametrize("split", [1.0e-14, 1.0e-13, 1.0e-12])
@pytest.mark.parametrize("seed", range(3))
def test_issue183_a_genuine_split_is_not_merged(split: float, seed: int) -> None:
    """Negative control: a split the OPERATOR carries stays a scalar claim.

    ``[[0, split], [1, 0]]`` has eigenvalues ``+-sqrt(split)``; at
    ``split = 1e-14`` this is the PR #127 fixture, ``18.4`` units of round-off
    above the resolution. Whatever the basis, the stationary mode is
    conditioned alone and reported conditioning-limited, with the same
    per-mode figure ``~2 sqrt(split)``.
    """
    block = np.zeros((4, 4), dtype=complex)
    block[0, 1] = split
    block[1, 0] = 1.0
    block[2, 2] = -1.0
    block[3, 3] = -2.0
    plain, rotated = _issue168_pair(seed=seed, block=block)
    for L in (plain, rotated):
        evidence = zero_mode_conditioning(L, zero_tolerance=1.0e-12)
        assert evidence.cluster_size == 1
        assert evidence.verdict == CONDITIONING_LIMITED
        assert evidence.per_mode_reciprocal_condition == pytest.approx(
            2.0 * math.sqrt(split), rel=2e-2
        )


def test_issue183_resolution_costs_no_svd_on_healthy_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pseudospectral test is reached only where first order cannot decide.

    On physical generators every pair is resolved by the first-order estimate,
    so the closure must not take a single SVD -- and must not change the answer
    the layer's own cutoff gives.
    """
    calls: list[complex] = []

    def counting(L: np.ndarray, z: complex) -> float:
        calls.append(z)
        return _smallest_singular_value(L, z)

    monkeypatch.setattr(conditioning_module, "_smallest_singular_value", counting)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    rng = np.random.default_rng(0)
    generators = [
        build_liouvillian(
            0.5 * sigma_x, [math.sqrt(g) * sigma_minus, math.sqrt(w) * sigma_z]
        )
        for g in (1.0e-3, 1.0, 1.0e3)
        for w in (1.0e-4, 1.0e-1, 10.0)
    ]
    for d in (3, 4):
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        jumps = [
            rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            for _ in range(2)
        ]
        generators.append(build_liouvillian(H + H.conj().T, jumps))
    for L in generators:
        accepted, certificate = certified_eigvals(L)
        evidence = zero_mode_conditioning(
            L,
            zero_tolerance=certificate.zero_set_tolerance(accepted),
            eigenvalues=accepted,
        )
        assert evidence.verdict == CONDITIONING_BENIGN
        assert evidence.cluster_size == 1
    assert calls == []


def test_smallest_singular_value_is_unitarily_invariant_and_scale_covariant() -> None:
    """The two invariances the grouping inherits from ``sigma_min``."""
    _plain, rotated = _issue168_pair()
    rng = np.random.default_rng(11)
    U, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    z = 0.3 - 0.2j
    base = _smallest_singular_value(rotated, z)
    assert _smallest_singular_value(U @ rotated @ U.conj().T, z) == pytest.approx(
        base, rel=1e-12
    )
    assert _smallest_singular_value(1.0e200 * rotated, 1.0e200 * z) == pytest.approx(
        1.0e200 * base, rel=1e-12
    )


def _resolvent_bound(L: np.ndarray, z: complex) -> float:
    """``1 / sum_j 1 / (s_j |lambda_j - z|)`` from the eigendecomposition."""
    values, left, right = sla.eig(L, left=True, right=True)
    s = eigenvalue_conditioning(right, left)
    return float(1.0 / np.sum(1.0 / (s * np.abs(values - z))))


@pytest.mark.parametrize("seed", range(30))
def test_resolvent_bound_never_exceeds_sigma_min(seed: int) -> None:
    """The pre-filter is a PROOF, not an estimate: bound <= sigma_min.

    Random dense, strongly non-normal (scaled upper-triangular) and GKSL
    operators, probed at points near and far from the spectrum.
    """
    rng = np.random.default_rng(seed)
    n = (4, 9, 16)[seed % 3]
    kind = (seed // 3) % 3
    if kind == 0:
        L = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    elif kind == 1:
        # Departure from normality up to ~30x: strong, yet sigma_min stays
        # above the SVD's absolute accuracy somewhere, so the bound is tested
        # rather than compared against round-off (10**4 left it nowhere).
        L = np.triu(rng.standard_normal((n, n)), 1) * 10.0 ** rng.uniform(0, 1.5)
        L = L + np.diag(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    else:
        d = math.isqrt(n)
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        jumps = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))]
        L = build_liouvillian(H + H.conj().T, jumps)
    values = np.linalg.eigvals(L)
    probes = list(rng.standard_normal(6) + 1j * rng.standard_normal(6))
    probes += [v + 1.0e-3 * (1 + 1j) for v in values[:3]]
    # Two points outside the numerical range, where sigma_min >= |z| - ||L||_2
    # is always resolvable, so no seed leaves the property untested.
    radius = 2.0 * float(np.linalg.norm(L, 2))
    probes += [radius * np.exp(1j * t) for t in (0.3, 2.1)]
    eps = float(np.finfo(float).eps)
    meaningful = 0
    for z in probes:
        shifted = L - z * np.eye(n)
        sigma = float(sla.svdvals(shifted)[-1])
        # A backward-stable SVD gets sigma_min to an ABSOLUTE accuracy of
        # ~n eps ||A||, not a relative one: CI's Python 3.10-3.14 jobs compared
        # 4.65e-23 against 4.31e-23 on a matrix of norm ~1e4, i.e. two numbers
        # eleven orders below what either computation resolves.
        floor = n * eps * float(np.linalg.norm(shifted))
        assert _resolvent_bound(L, z) <= sigma * (1.0 + 1.0e-6) + floor
        meaningful += int(sigma > 1.0e3 * floor)
    assert meaningful >= 1  # the property was tested, not only its noise


def test_first_order_estimate_is_not_a_bound() -> None:
    """Why the one-term estimate was replaced: it is not a lower bound.

    The first version of the pre-filter skipped the SVD when
    ``min(s_i, s_j) * |gap| / 2`` exceeded the resolution by 10x. That ignores
    every OTHER mode's share of the resolvent at the midpoint. Here a
    well-conditioned pair ``0, 1e-6`` sits beside an order-8 Jordan block at
    ``0.1`` -- no eigenvalue between them -- and the estimate overstates
    ``sigma_min`` at their midpoint by ``50x``, beyond its headroom. (In the
    fixtures examined the closure still reached the right group, transitively
    through the Jordan modes, whose own ``s ~ 0`` forced a probe; nothing
    guaranteed that.) The summed resolvent bound stays below ``sigma_min``.
    """
    n = 16
    L = np.zeros((n, n), dtype=complex)
    L[1, 1] = 1.0e-6
    for k in range(2, 10):
        L[k, k] = 0.1
    for k in range(2, 9):
        L[k, k + 1] = 1.0
    for k in range(10, n):
        L[k, k] = -float(k - 9)
    values, left, right = sla.eig(L, left=True, right=True)
    s = eigenvalue_conditioning(right, left)
    i = int(np.argmin(np.abs(values)))
    j = int(np.argmin(np.abs(values - 1.0e-6)))
    mid = 0.5 * (values[i] + values[j])
    gap = abs(values[i] - values[j])
    others = np.abs(np.delete(values, [i, j]) - mid)
    assert bool(np.all(others >= 0.5 * gap))  # premise: no eigenvalue between
    one_term = float(min(s[i], s[j]) * gap / 2.0)
    sigma = float(sla.svdvals(L - mid * np.eye(n))[-1])
    assert one_term > 10.0 * sigma
    assert _resolvent_bound(L, mid) <= sigma * (1.0 + 1.0e-6)


@pytest.mark.parametrize("seed", range(10))
def test_issue183_order_three_jordan_block_is_kept_whole(seed: int) -> None:
    """A defective triple (d = 3) splits into a triangle; it is one cluster."""
    d = 3
    n = d * d
    q = trace_vector(d)
    basis = _basis_with_first(q)
    block = np.zeros((n, n), dtype=complex)
    block[1, 0] = block[2, 1] = 1.0
    for k in range(3, n):
        block[k, k] = -float(k - 2)
    plain = basis @ block @ basis.conj().T
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(
        rng.standard_normal((n - 1, n - 1)) + 1j * rng.standard_normal((n - 1, n - 1))
    )
    U = np.outer(q, q.conj()) + basis[:, 1:] @ V @ basis[:, 1:].conj().T
    rotated = U @ plain @ U.conj().T
    split = float(np.sort(np.abs(np.linalg.eigvals(rotated)))[2])
    assert split > 1.0e-7  # premise: the rotation splits the triple
    for L in (plain, rotated):
        for tol in (0.0, 1.0e-12, 1.0e-8):
            evidence = zero_mode_conditioning(L, zero_tolerance=tol)
            assert evidence.verdict == CONDITIONING_CLUSTER_ONLY
            assert evidence.cluster_size == 3
            assert evidence.reciprocal_condition == pytest.approx(1.0, rel=1e-8)


def test_schur_gate_still_refuses_a_resolvably_different_spectrum() -> None:
    """The relaxed pairing is "to the arithmetic's resolution", not "any".

    ``diag(0, 0, -1, -2)`` is normal, so its resolution is ``~eps``: a report
    claiming ``1e-8`` for the second zero, or ``-2.001`` for the last mode,
    describes a spectrum the arithmetic separates from this one, and the
    projector must be refused whether the difference is inside the cluster or
    in its complement.
    """
    T = np.diag([0.0, 0.0, -1.0, -2.0]).astype(complex)
    for reference in (
        np.array([0.0, 1.0e-8, -1.0, -2.0], dtype=complex),
        np.array([0.0, 0.0, -1.0, -2.001], dtype=complex),
    ):
        resolution = _Resolution(T, reference, np.ones(4), 0.0)
        assert (
            _schur_cluster_projector_condition(
                T,
                reference_eigenvalues=reference,
                cluster_indices=np.array([0, 1], dtype=np.intp),
                resolution=resolution,
            )
            is None
        )


@pytest.mark.parametrize("factor", [1.0e-200, 1.0e200, 1.0e300])
def test_audit_lapack_calls_are_power_of_two_rescaled(factor: float) -> None:
    """Every rate-valued output follows ``c L``; nothing else moves.

    The CI's Python 3.11 job (SciPy 1.17.1, scipy-openblas 0.3.31) returned
    the eigenvalues of ``1e200 * L`` near ``1e138`` -- ``xGEEV`` scaled the
    operator down and failed to scale the eigenvalues back (``DLASCL ...
    illegal value``) -- and the resolution closure then correctly found that
    garbage spectrum unresolved. The audit now makes every LAPACK call on an
    exactly power-of-two rescaled operator, so the build's internal scaling
    path is never taken.
    """
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    base = build_liouvillian(0.5 * sigma_x, [math.sqrt(0.6) * sigma_minus])
    scaled = factor * base

    unit, exponent = conditioning_module._unit_scaled(scaled)
    assert 0.5 <= float(np.max(np.abs(np.r_[unit.real.ravel(), unit.imag.ravel()]))) < 1.0
    np.testing.assert_array_equal(
        conditioning_module._ldexp_complex(unit, exponent), scaled
    )

    a = zero_mode_conditioning(base, zero_tolerance=0.0)
    b = zero_mode_conditioning(scaled, zero_tolerance=0.0)
    assert a.verdict == b.verdict
    assert a.cluster_size == b.cluster_size
    assert b.reciprocal_condition == pytest.approx(a.reciprocal_condition, rel=1e-8)
    assert b.separation == pytest.approx(factor * a.separation, rel=1e-8)
    assert _smallest_singular_value(scaled, factor * (0.1 + 0.2j)) == pytest.approx(
        factor * _smallest_singular_value(base, 0.1 + 0.2j), rel=1e-10
    )


def test_resolution_svd_budget_bounds_the_audit_and_fails_closed() -> None:
    """A strongly non-normal operator ends as UNAVAILABLE, never as a hang.

    ``D A D^-1`` with a ``1e8`` spread leaves the resolvent bound unable to
    certify anything, so every candidate needs an SVD; the count grows with
    ``n`` (251 at ``n = 64`` without a budget). The audit runs by default in
    the spectral layer, so it stops at ``_SVD_BUDGET`` and says so.
    """
    n = 64
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    D = np.diag(np.logspace(0.0, 8.0, n))
    M = D @ A @ np.linalg.inv(D)

    calls: list[complex] = []
    original = conditioning_module._smallest_singular_value

    def counting(L: np.ndarray, z: complex) -> float:
        calls.append(z)
        return original(L, z)

    conditioning_module._smallest_singular_value = counting
    try:
        evidence = zero_mode_conditioning(M, zero_tolerance=1.0)
    finally:
        conditioning_module._smallest_singular_value = original
    assert evidence.verdict == CONDITIONING_UNAVAILABLE
    assert evidence.reason == "resolution_budget_exhausted"
    assert len(calls) <= conditioning_module._Resolution._SVD_BUDGET


@pytest.mark.parametrize("seed", range(5))
def test_lipschitz_reuse_never_contradicts_a_direct_svd(seed: int) -> None:
    """Reused decisions agree with the SVD they replace.

    ``sigma_min(L - z I)`` is 1-Lipschitz in ``z``, so an earlier probe may
    decide a nearby point. Every decision the cache makes is checked here
    against a fresh SVD at the same point.
    """
    rng = np.random.default_rng(seed)
    n = 9
    L = np.triu(rng.standard_normal((n, n)), 1) * 3.0 + np.diag(
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    values = np.linalg.eigvals(L)
    resolution = _Resolution(L, values, np.ones(n), 0.0)
    resolution.delta = 0.05 * float(np.linalg.norm(L))  # wide enough to matter
    for z in values[0] + 0.3 * (rng.standard_normal(40) + 1j * rng.standard_normal(40)):
        decided = resolution._point_in_pseudospectrum(z)
        direct = not (_smallest_singular_value(L, z) > resolution.delta)
        assert decided == direct
    assert resolution.svd_calls < 40  # the reuse actually decided some points
