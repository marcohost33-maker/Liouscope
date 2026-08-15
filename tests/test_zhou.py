"""Tests for v0.2.1 Zhou D24 predictor."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import _zhou, build_liouvillian


def test_zhou_claim_status_is_reference_verified_bound_coarser():
    """S6 re-audit 2026-06-04: reference verified, but our bound is coarser.

    The D24 Zhou predictor cites arXiv:2601.06256, which was independently
    verified to exist (Yi-Neng Zhou, "Universal Predictors for Mixing Time
    more than Liouvillian Gap", v3 2026-05-20). Our implemented upper bound is
    in the same family as Zhou's Eq.(16) but is a *related, generally coarser*
    surrogate (Petermann Schatten-2 factor instead of Zhou's per-mode
    trace-norm factor C_j, a single global gap/K_max, and no N_mode factor).
    The module must advertise this exact status so downstream tooling never
    presents our bound as a verbatim implementation of Eq.(16).
    """
    assert _zhou.CLAIM_STATUS == "reference-verified-bound-coarser"
    assert "UNVERIFIED" not in _zhou.CLAIM_REFERENCE
    assert "2601.06256" in _zhou.CLAIM_REFERENCE
    assert "Yi-Neng Zhou" in _zhou.CLAIM_REFERENCE


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


# --------------------------------------------------------------------------
# Round-13 review: the D24 recomputation must use the certified eigensolve.
# --------------------------------------------------------------------------

# Local copy of the #112 stiff fixture (the ``tests`` directory is not an
# importable package on the CI runner, so no cross-test import).
_STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
_STIFF_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]
_STIFF_TRUE_GAP = 1.074e-5


def _classical_network(pairs, rates, d=4):
    jumps = []
    for (to, frm) in pairs:
        j = np.zeros((d, d), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((d, d), dtype=complex), jumps, rates)


def test_zhou_recompute_uses_the_certified_spectrum():
    """D24 must not retain the solver failure the other layers repair.

    On the stiff #112 network the raw ``zgeev`` spectrum loses the exact zero
    mode and leaves a spurious ``7.28e-6`` in its place, shifting the whole
    predicted mixing-time window by ~30%. The certified solve recovers the
    physical gap ``1.074e-5``.
    """
    lsup = _classical_network(_STIFF_PAIRS, _STIFF_RATES)
    result = _zhou.compute_zhou_predictor(lsup, epsilon=0.05)
    assert result.converged is True
    assert result.gap == pytest.approx(_STIFF_TRUE_GAP, rel=1e-3)
    assert result.gap != pytest.approx(7.28e-6, rel=1e-3)  # the pre-fix value


def test_zhou_unresolved_spectrum_returns_unconverged():
    """An unresolved eigendecomposition must not drive a mixing-time claim.

    With the fast rate pushed to ``1e8`` the spectral spread defeats every
    repair route (issues #112/#113): the honest answer is an unconverged
    record, not bounds built on a demonstrably wrong spectrum.
    """
    from liouscope.numerics.linalg import certified_eig

    lsup = _classical_network(
        _STIFF_PAIRS, [7.28e-6, 3.67e-5, 1.53e-5, 1.0e8, 1.42e-5]
    )
    _decomp, cert = certified_eig(lsup)
    if cert.resolved:  # pragma: no cover - guard against future solver changes
        pytest.skip("fixture no longer produces an unresolved spectrum")

    result = _zhou.compute_zhou_predictor(lsup, epsilon=0.05)
    assert result.converged is False
    assert np.isinf(result.mixing_time_lower)
    assert np.isinf(result.mixing_time_upper)
    assert np.isnan(result.gap)
    assert np.isnan(result.petermann_factor)

    # Caller-supplied values are honoured in the unconverged record, exactly
    # like the existing no-nonzero-modes branch.
    supplied = _zhou.compute_zhou_predictor(lsup, epsilon=0.05, gap=0.5)
    assert supplied.converged is False
    assert supplied.gap == 0.5


def test_zhou_inapplicable_certificate_falls_back_to_the_radius_filter():
    """Round-14: the operator bound is not a cutoff without trace preservation.

    A non-trace-preserving, strongly non-normal input has no guaranteed zero
    mode; its certificate bound (~2.2e3 here) exceeds the whole spectrum, and
    filtering with it discarded every eigenvalue -- D24 reported gap 0.0 /
    unconverged for a well-separated spectrum with true gap 1.
    """
    A = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(complex)
    A[0, 3] = 1.0e16
    result = _zhou.compute_zhou_predictor(A, epsilon=0.05)
    assert result.converged is True
    assert result.gap == pytest.approx(1.0, rel=1e-9)
