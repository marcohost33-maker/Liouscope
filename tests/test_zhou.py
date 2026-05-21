"""Tests for v0.2.1 Zhou D24 predictor."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import _zhou, build_liouvillian


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
