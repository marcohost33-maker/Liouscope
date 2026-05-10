"""Tests for v0.2.1 Zhou D24 predictor."""

from __future__ import annotations

import numpy as np

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
