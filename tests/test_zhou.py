"""Tests for v0.2.1 Zhou D24 predictor."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import _zhou, build_liouvillian


def test_zhou_claim_status_is_pending():
    """S6 audit: the unverified reference must be flagged as pending.

    The D24 Zhou predictor cites arXiv:2601.06256, which could not be
    independently verified. The module must advertise claim_status='pending'
    so downstream tooling never treats D24 as publication-grade.
    """
    assert _zhou.CLAIM_STATUS == "pending"
    assert "UNVERIFIED" in _zhou.CLAIM_REFERENCE
    assert "2601.06256" in _zhou.CLAIM_REFERENCE


def test_zhou_returns_bounds(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    assert res.converged
    assert res.mixing_time_upper >= res.mixing_time_lower > 0


def test_zhou_handles_zero_gap():
    L = np.zeros((4, 4), dtype=complex)
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    assert not res.converged
    assert np.isinf(res.mixing_time_lower) or np.isinf(res.mixing_time_upper)


def test_zhou_handles_supplied_gap(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3, gap=0.3, petermann_factor=1.1)
    assert res.converged
    assert res.mixing_time_lower > 0


def test_zhou_result_carries_gap_and_petermann():
    """Result records the gap and Petermann factor so it can be rescaled."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(0.5 * sx, [sz], [0.3])
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    assert np.isfinite(res.gap) and res.gap > 0
    assert np.isfinite(res.petermann_factor) and res.petermann_factor >= 1.0


def test_zhou_rescale_matches_recompute():
    """Rescaling via :func:`mixing_time_upper_bound` matches a fresh computation."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(0.5 * sx, [sz], [0.3])
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    new_eps = 1e-6
    rescaled = _zhou.mixing_time_upper_bound(res, eps=new_eps)
    fresh = _zhou.compute_zhou_predictor(L, epsilon=new_eps)
    assert abs(rescaled - fresh.mixing_time_upper) < 1e-9
    # Tighter epsilon means longer mixing time.
    assert rescaled > res.mixing_time_upper


def test_zhou_rescale_identity():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(0.5 * sx, [sz], [0.3])
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    assert _zhou.mixing_time_upper_bound(res) == res.mixing_time_upper
    assert _zhou.mixing_time_upper_bound(res, eps=res.epsilon) == res.mixing_time_upper


def test_zhou_rescale_rejects_unconverged():
    res = _zhou.compute_zhou_predictor(np.zeros((4, 4), dtype=complex), epsilon=1e-3)
    with pytest.raises(ValueError, match="converge"):
        _zhou.mixing_time_upper_bound(res, eps=1e-6)


# --- Anchor (Zhou predictor) -------------------------------------------------
# Hand-derived oracle for the v0.2.1 D24 predictor. Pure dephasing of a single
# qubit (H = 0, single jump operator Z, rate gamma = 0.5):
#   * The GKSL dissipator gamma*(Z rho Z - rho) acts on the two coherences with
#     eigenvalue -2*gamma = -1.0 and on the two populations with eigenvalue 0.
#     Hence the Liouvillian spectrum is {0, 0, -1, -1} and the spectral gap is
#     Delta = -max Re(lambda != 0) = 1.0  (verified numerically below).
#   * The Liouvillian is diagonal in the operator basis -> normal -> every
#     Petermann factor is exactly 1, so K_max = 1.
#   * Zhou's simplified universal form then collapses both bounds onto
#       t_lower = t_upper = log(1/eps)/Delta = log(1000) = 6.907755278982137
#     for eps = 1e-3. This is a closed-form orcale independent of the impl.
def test_anchor_zhou_pure_dephasing_closed_form():
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    h0 = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(h0, [z], [0.5])
    eps = 1e-3
    res = _zhou.compute_zhou_predictor(L, epsilon=eps)

    assert res.converged
    # Gap and Petermann are exact for this normal, diagonal Liouvillian.
    np.testing.assert_allclose(res.gap, 1.0, atol=1e-9)
    np.testing.assert_allclose(res.petermann_factor, 1.0, atol=1e-9)
    # K = 1 -> sqrt(K) = 1 -> both bounds equal log(1/eps)/gap.
    expected = float(np.log(1.0 / eps) / 1.0)
    np.testing.assert_allclose(res.mixing_time_lower, expected, atol=1e-9)
    np.testing.assert_allclose(res.mixing_time_upper, expected, atol=1e-9)


def test_zhou_defective_mode_does_not_poison_kmax():
    """A near-defective mode (denom -> 0) must be skipped, not yield inf K_max.

    The Petermann-style guard in _zhou uses the canonical EPS_DIV floor and
    ``continue`` so that t_upper stays finite when a mode is (numerically)
    defective. A 2x2 Jordan-like generator has a vanishing left/right overlap;
    the predictor must still return a finite, converged upper bound.
    """
    # Jordan block embedded so it has a real spectral gap and a defective pair.
    L = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -2.0],
        ],
        dtype=complex,
    )
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    assert res.converged
    assert np.isfinite(res.mixing_time_upper)
    assert np.isfinite(res.petermann_factor)
