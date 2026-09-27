"""Issue #168: basis-invariant conditioning of repeated/defective zero clusters."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.numerics.conditioning import (
    CONDITIONING_CLUSTER_ONLY,
    _schur_cluster_projector_condition,
    cluster_conditioning,
    zero_mode_conditioning,
)
from liouscope.numerics.traceless import trace_vector


def _basis_with_first(q: np.ndarray) -> np.ndarray:
    seed = np.eye(q.size, dtype=complex)
    seed[:, 0] = q
    basis, _ = np.linalg.qr(seed)
    phase = np.vdot(basis[:, 0], q)
    return np.asarray(basis * (phase / abs(phase)), dtype=complex)


def _issue168_pair() -> tuple[np.ndarray, np.ndarray]:
    """The exact reproduction from issue #168 and a unitary re-expression."""
    q = trace_vector(2)
    basis = _basis_with_first(q)

    jordan = np.zeros((4, 4), dtype=complex)
    jordan[1, 0] = 1.0
    jordan[2, 2] = -1.0
    jordan[3, 3] = -2.0
    plain = basis @ jordan @ basis.conj().T

    rng = np.random.default_rng(3)
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
        T, zero_tolerance=1.0e-12, expected_cluster_size=2
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
        L, zero_tolerance=1.0e-8, expected_cluster_size=2
    )
    assert schur is not None
    assert schur == pytest.approx(legacy, rel=2e-10, abs=2e-12)


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
