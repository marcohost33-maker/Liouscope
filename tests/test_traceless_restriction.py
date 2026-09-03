"""Proof-oriented tests for the structural traceless restriction (#113)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.numerics.traceless import (
    restrict_to_traceless,
    trace_vector,
    traceless_basis,
)

_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _amplitude_damped_qubit(rate: float = 1.0) -> np.ndarray:
    return build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [_LOWER],
        [rate],
    )


def _classical_stiff_network(fast_rate: float) -> np.ndarray:
    pairs = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
    rates = [7.28e-6, 3.67e-5, 1.53e-5, fast_rate, 1.42e-5]
    jumps: list[np.ndarray] = []
    for to, frm in pairs:
        jump = np.zeros((4, 4), dtype=complex)
        jump[to, frm] = 1.0
        jumps.append(jump)
    return build_liouvillian(
        np.zeros((4, 4), dtype=complex),
        jumps,
        rates,
    )


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_basis_is_orthonormal_and_exactly_traceless(d: int) -> None:
    basis = traceless_basis(d)
    q = trace_vector(d)

    assert basis.shape == (d * d, d * d - 1)
    assert np.allclose(q.conj() @ basis, 0.0, rtol=0.0, atol=5.0e-16)
    assert np.allclose(
        basis.conj().T @ basis,
        np.eye(d * d - 1),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_restriction_preserves_the_nonzero_spectrum_for_unique_steady_state() -> None:
    L_super = _amplitude_damped_qubit(rate=1.0)
    reduced = restrict_to_traceless(L_super)

    full = np.linalg.eigvals(L_super)
    full_nonzero = np.delete(full, int(np.argmin(np.abs(full))))
    restricted = np.linalg.eigvals(reduced.operator)

    assert np.allclose(
        np.sort_complex(restricted),
        np.sort_complex(full_nonzero),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert reduced.invariance_defect <= 1.0e-13 * reduced.operator_scale
    assert reduced.reconstruction_defect <= 1.0e-13 * reduced.operator_scale


def test_restriction_is_rate_scale_homogeneous() -> None:
    L_super = _amplitude_damped_qubit(rate=0.7)
    base = restrict_to_traceless(L_super).operator

    for scale in (1.0e-100, 1.0, 1.0e100):
        got = restrict_to_traceless(scale * L_super).operator
        # Divide back before comparison so an absolute test tolerance cannot
        # hide a failure in the tiny-rate case.
        assert np.allclose(got / scale, base, rtol=3.0e-14, atol=3.0e-14)


def test_degenerate_stationary_manifold_keeps_its_traceless_zero_mode() -> None:
    """Over-correction control: the reduction removes one, not every, zero mode."""
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    L_super = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [sigma_z],
        [0.3],
    )
    restricted = np.linalg.eigvals(restrict_to_traceless(L_super).operator)

    # Pure dephasing has a two-dimensional stationary diagonal algebra. The
    # trace direction is removed, while one stationary traceless diagonal mode
    # must remain. A primitive that simply deletes every small eigenvalue would
    # fail this control and erase legitimate conserved structure.
    assert np.count_nonzero(np.abs(restricted) < 1.0e-12) == 1


def test_non_trace_preserving_input_is_refused_before_restriction() -> None:
    bad = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)
    with pytest.raises(ValueError, match="not trace preserving"):
        restrict_to_traceless(bad)


@pytest.mark.parametrize("fast_rate", [1.0e8, 1.0e10, 1.0e12])
def test_stiff_network_keeps_the_known_slow_mode_without_zero_filter(
    fast_rate: float,
) -> None:
    """#113 discrimination: no spectral-radius threshold can swallow the gap."""
    L_super = _classical_stiff_network(fast_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reduced = restrict_to_traceless(L_super)
        eigenvalues = np.linalg.eigvals(reduced.operator)

    # The slowest physical Liouvillian modes for this classical-jump fixture
    # are coherence modes at 1.074e-5. The value is independent of the separate
    # fast population edge. The old zero filter reports a fast mode once its
    # radius-derived cutoff grows past this scale; the restricted operator has
    # no unique stationary eigenvalue to filter in the first place.
    gap = -float(np.max(np.real(eigenvalues)))
    assert gap == pytest.approx(1.074e-5, rel=2.0e-8, abs=0.0)
    assert reduced.invariance_defect <= 1.0e-12 * reduced.operator_scale
