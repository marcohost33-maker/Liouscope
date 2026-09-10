"""Proof-oriented tests for the structural traceless restriction (#113)."""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.numerics.linalg import (
    trace_preservation_componentwise_error,
    trace_preservation_defect,
)
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


def _diluted_trace_violation() -> np.ndarray:
    """A generator whose trace violation is invisible to the normwise ratio.

    Column 2 of this 4x4 superoperator has the trace equation
    ``L[0, 2] + L[3, 2] = 1``, a relative error of one. The entry ``L[1, 0]``
    takes part in no trace equation at all, but it sets ``||L||_F``, so the
    global ratio ``||q^H L|| / ||L||`` reads 1e-300.
    """
    L_super = np.zeros((4, 4), dtype=complex)
    L_super[1, 0] = 1.0e300
    L_super[0, 2] = 1.0
    return L_super


def test_componentwise_gate_refuses_diluted_defect() -> None:
    """PR #139 review, finding B1: a normwise gate carries no bits here.

    Before the componentwise gate this input was ACCEPTED and a restriction
    was returned whose own audit numbers refuted it: ``invariance_defect``
    and ``reconstruction_defect`` were both 0.707, i.e. the traceless subspace
    -- the one property the whole construction relies on -- was not invariant
    under the generator that had just passed the gate.
    """
    L_super = _diluted_trace_violation()

    # The dilution is a fact about the two readings, not an artefact of the
    # test fixture: assert it before asserting what the gate does with it.
    defect, scale = trace_preservation_defect(L_super)
    assert defect / scale == pytest.approx(1.0e-300, rel=1.0e-12)
    assert trace_preservation_componentwise_error(L_super) == pytest.approx(1.0)

    # Caught broadly, then classified: a test that dies of an unexpected
    # exception type would otherwise read as a pass for the wrong reason.
    with pytest.raises(Exception, match="componentwise") as exc:
        restrict_to_traceless(L_super)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_the_same_large_entry_without_a_violated_equation_is_still_accepted() -> None:
    """Positive control paired with the negative one: only the violation moves.

    This is the refusal fixture with the offending ``L[0, 2] = 1`` removed and
    nothing else changed. The 1e300 entry, the stiffness and the dimension are
    identical, so a gate that rejected this too would be rejecting SIZE rather
    than a broken trace equation.
    """
    L_super = _diluted_trace_violation()
    L_super[0, 2] = 0.0

    reduced = restrict_to_traceless(L_super)

    assert reduced.trace_componentwise_error == 0.0
    assert reduced.operator.shape == (3, 3)
    # Accepted for the right reason: the subspace really is invariant.
    assert reduced.invariance_defect == 0.0
    assert reduced.reconstruction_defect == 0.0


@pytest.mark.parametrize("fast_rate", [1.0e8, 1.0e10, 1.0e12])
def test_stiff_physical_generator_passes_a_live_componentwise_gate(
    fast_rate: float,
) -> None:
    """Positive control that proves the gate it passes is not vacuous.

    A gate that accepts everything and a gate that reads a real number can
    both produce a green test here, so acceptance alone is not evidence. The
    componentwise reading of this genuine jump-network generator is nonzero
    (~1e-17, six orders of rate span notwithstanding), and tightening
    ``tp_rtol`` below that measured value rejects THIS SAME operator through
    THIS SAME gate -- which is what shows the comparison is on the code path
    and the pass at the default tolerance was earned.
    """
    L_super = _classical_stiff_network(fast_rate)

    reduced = restrict_to_traceless(L_super)

    error = reduced.trace_componentwise_error
    normwise = reduced.trace_defect / reduced.operator_scale
    assert 0.0 < error < 1.0e-10, f"expected a live sub-tolerance reading, got {error}"
    assert 0.0 < normwise < 1.0e-10, f"expected a live normwise reading, got {normwise}"
    assert reduced.invariance_defect <= 1.0e-12 * reduced.operator_scale

    # Both readings are real, nonzero numbers here, so lowering the tolerance
    # below both must refuse the very same operator. Which of the two speaks
    # first is deliberately NOT asserted: they are within a factor of two of
    # each other on this fixture, and pinning the order would make the test
    # about evaluation sequence rather than about the gate being live.
    with pytest.raises(Exception, match="not trace preserving") as exc:
        restrict_to_traceless(L_super, tp_rtol=min(error, normwise) / 2.0)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_componentwise_gate_shares_the_eigensolver_verdict() -> None:
    """Same evidence, same tolerance, same verdict as the certified paths.

    Two gates that measure the same quantity differently are a defect class of
    their own, so the agreement is asserted rather than assumed.
    """
    for L_super in (
        _diluted_trace_violation(),
        _classical_stiff_network(1.0e12),
        _amplitude_damped_qubit(1.0),
    ):
        error = trace_preservation_componentwise_error(L_super)
        rejected_by_eigensolver_rule = error > 1.0e-10

        try:
            restrict_to_traceless(L_super, tp_rtol=1.0e-10)
            rejected_here = False
        except ValueError:
            rejected_here = True

        assert rejected_here == rejected_by_eigensolver_rule


def _exactly_cancelling_large_column(d: int = 16, upper: float = 2.5e307) -> np.ndarray:
    """A legal generator whose trace equation only cancels outside float range.

    Column 0 carries ``+upper`` in the first half of the trace rows and
    ``-upper`` in the second, so the trace equation is EXACTLY zero while the
    partial sums reach ``d/2 * upper``. The Frobenius norm is
    ``sqrt(d) * upper``, which stays inside float64 for these values.
    """
    n = d * d
    L_super = np.zeros((n, n), dtype=complex)
    trace_rows = np.arange(0, n, d + 1, dtype=int)
    half = trace_rows.size // 2
    L_super[trace_rows[:half], 0] = upper
    L_super[trace_rows[half:], 0] = -upper
    return L_super


def test_exactly_cancelling_large_column_is_no_longer_refused() -> None:
    """PR #139 review, P2: the mirror image of the finding fixed in 5c151f9.

    That one was fail-open on an invalid generator. This is fail-closed on a
    valid one: the operator is exactly trace preserving and entirely
    representable, and it was refused because an intermediate product -- not
    the result -- left the float64 range.
    """
    L_super = _exactly_cancelling_large_column()
    d = 16

    # The rejected evidence, as the helper used to assemble it.
    vec_i = np.eye(d, dtype=complex).reshape(-1, order="F")
    with np.errstate(over="ignore", invalid="ignore"):
        old_reading = (vec_i.conj() @ L_super)[0]
    assert not np.isfinite(old_reading), (
        "fixture no longer reproduces the intermediate overflow it was built for"
    )

    # Asserted at the helper first, so that a regression here fails as a
    # verdict about a number and not as an exception escaping the call below.
    # A test that dies of an unexpected error type classifies nothing.
    defect, scale = trace_preservation_defect(L_super)
    assert np.isfinite(defect), f"trace defect is not a number: {defect}"
    assert defect == 0.0
    assert scale == 1.0e308

    try:
        reduced = restrict_to_traceless(L_super)
    except ValueError as exc:  # pragma: no cover - only on regression
        pytest.fail(f"exactly trace-preserving generator refused: {exc}")

    assert reduced.trace_defect == 0.0
    assert reduced.operator_scale == 1.0e308
    assert reduced.trace_componentwise_error == 0.0
    # Accepted for the right reason: the subspace really is invariant, measured
    # RELATIVE to the operator's own scale as everywhere else in this module.
    assert reduced.invariance_defect <= 1.0e-13 * reduced.operator_scale


def test_the_overflow_repair_still_measures_a_real_defect_in_that_regime() -> None:
    """Non-emptiness: the repaired reading is computed, not merely survived.

    A helper that returned zero for everything near the float ceiling would
    also make the test above pass. Perturbing one term by a representable
    amount must therefore produce that exact defect and a refusal.
    """
    L_super = _exactly_cancelling_large_column()
    trace_rows = np.arange(0, 256, 17, dtype=int)
    L_super[trace_rows[8], 0] += 1.0e306

    # Exact oracle over the REPRESENTED coefficients: the perturbation itself
    # is rounded when it is added to 2.5e307, so 1e306 is not the number the
    # matrix actually holds. Fractions are exact for float64 values.
    exact = sum(Fraction(v) for v in L_super[trace_rows, 0].real.tolist())
    defect, scale = trace_preservation_defect(L_super)

    assert exact != 0
    assert defect == float(abs(exact))
    assert defect / scale > 1.0e-10

    with pytest.raises(Exception, match="not trace preserving") as exc:
        restrict_to_traceless(L_super)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_the_overflow_repair_does_not_reopen_the_diluted_defect() -> None:
    """Guarding the previous repair against this one.

    An overflow fix that widened the normwise reading could hand the diluted
    generator of finding B1 straight back through, and that regression would
    look exactly like a green suite. The refusal is asserted together with the
    reason, so a refusal for some unrelated cause cannot stand in for it.
    """
    L_super = _diluted_trace_violation()

    defect, scale = trace_preservation_defect(L_super)
    # The normwise reading still passes -- it is the componentwise gate that
    # must do the work, exactly as before the overflow repair.
    assert defect / scale == pytest.approx(1.0e-300, rel=1.0e-12)
    assert trace_preservation_componentwise_error(L_super) == pytest.approx(1.0)

    with pytest.raises(Exception, match="componentwise") as exc:
        restrict_to_traceless(L_super)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )
