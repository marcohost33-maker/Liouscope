"""Proof-oriented tests for the structural traceless restriction (#113)."""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as scipy_linalg
from conftest import assert_spectra_match

from liouscope import build_liouvillian
from liouscope._consts import ZERO_MODE_EPS_FACTOR
from liouscope.numerics import traceless as traceless_module
from liouscope.numerics.conditioning import eigenvalue_conditioning
from liouscope.numerics.linalg import (
    trace_preservation_componentwise_error,
    trace_preservation_defect,
)
from liouscope.numerics.traceless import (
    _require_invariant_reduction,
    restrict_to_traceless,
    trace_vector,
    traceless_basis,
)

_EPS = float(np.finfo(float).eps)

_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _amplitude_damped_qubit(rate: float = 1.0) -> np.ndarray:
    return build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [_LOWER],
        [rate],
    )


#: ``(to, from)`` jumps of the stiff classical network and their rates; the
#: entry ``None`` is the fast ``2 -> 3`` transition.
_STIFF_JUMPS: tuple[tuple[int, int, float | None], ...] = (
    (0, 3, 7.28e-6),
    (0, 2, 3.67e-5),
    (1, 0, 1.53e-5),
    (3, 2, None),
    (2, 1, 1.42e-5),
)


def _stiff_rates(fast_rate: float) -> list[float]:
    return [fast_rate if rate is None else rate for _, _, rate in _STIFF_JUMPS]


def _classical_stiff_network(fast_rate: float) -> np.ndarray:
    pairs = [(to, frm) for to, frm, _ in _STIFF_JUMPS]
    rates = _stiff_rates(fast_rate)
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

    assert_spectra_match(restricted, full_nonzero, rtol=2.0e-13, atol=2.0e-13)
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


#: Slowest coherence mode of :func:`_classical_stiff_network`. Classical jumps
#: ``|to><frm|`` map a coherence ``|i><j|`` (``i != j``) only onto itself, with
#: eigenvalue ``-(Gamma_i + Gamma_j) / 2``, ``Gamma`` the total out-rate; for the
#: pair (1, 3) that is ``-(1.42e-5 + 7.28e-6) / 2``. The fast rate leaves
#: state 2 only, so this value does not depend on it.
_STIFF_COHERENCE_MODE = -1.074e-5
_STIFF_REL = 2.0e-8


def _stiff_coherence_spectrum(fast_rate: float) -> np.ndarray:
    """Closed-form coherence eigenvalues ``-(Gamma_i + Gamma_j) / 2``, i != j.

    Ordered like the off-diagonal matrix units of :func:`traceless_basis`
    (column-stacked ``|i><j|``), so it can be compared entry by entry with the
    diagonal of the restricted coherence block.
    """
    d = 4
    out_rate = [0.0] * d
    for (_, frm, _), rate in zip(_STIFF_JUMPS, _stiff_rates(fast_rate)):
        out_rate[frm] += rate
    return np.array(
        [-(out_rate[i] + out_rate[j]) / 2.0 for j in range(d) for i in range(d) if i != j]
    )


def _stiff_population_slow_pair() -> np.ndarray:
    """Slow population pair of the stiff network, fast-rate independent.

    State 2 empties into 3 at the fast rate, so the slow dynamics is the
    three-state cycle 0 -> 1 -> (2 ->) 3 -> 0 with rates ``a, b, c``; its non-zero
    eigenvalues solve ``x^2 + (a+b+c) x + (ab+bc+ca) = 0``. The neglected
    branching ``2 -> 0`` is ``3.67e-5 / fast_rate`` of the ``1 -> 3`` flux, so
    the closed form agrees with a 60-digit eigensolve of the full rate matrix
    to below 1e-12 relative already at 1e8 (``-1.839e-5 +- 9.6864803e-6 i`` at
    1e8, 1e10 and 1e12 alike).
    """
    a, b, c = 1.53e-5, 1.42e-5, 7.28e-6
    s1 = a + b + c
    s2 = a * b + b * c + c * a
    root = np.sqrt(complex(s1 * s1 - 4.0 * s2))
    return np.array([(-s1 + root) / 2.0, (-s1 - root) / 2.0])


def _lapack_eigenvalue_error_estimate(A: np.ndarray, target: complex) -> float:
    """``eps * ||A||_1 / s`` for the eigenvalue of ``A`` nearest ``target``.

    This is the first-order estimate the LAPACK Users' Guide prescribes for
    ``xGEEV`` output (section 4.8, "Error Bounds for the Nonsymmetric
    Eigenproblem": ``EERRBD = EPSMCH * ABNRM / RCONDE``) -- asymptotic, not
    rigorous. ``numpy.finfo(float).eps`` is twice LAPACK's ``EPSMCH``, and the
    unbalanced ``||A||_1`` is not smaller than LAPACK's balanced ``ABNRM`` on
    this fixture, so both choices err on the wide side. ``s = |y^H x|`` is the
    reciprocal condition number of that eigenvalue on ``A`` as written. Where
    the solve does not resolve the eigenvalue (the 1e12 case) the measured
    ``s`` is that of whatever was computed (1.0 there, against ~0.82 at the
    resolved rates). Since ``s <= 1`` the estimate never drops below
    ``eps * ||A||_1``, and an overstated ``s`` only narrows the band.
    """
    w, left, right = scipy_linalg.eig(A, left=True, right=True)
    s = eigenvalue_conditioning(right, left)
    j = int(np.argmin(np.abs(w - target)))
    if s[j] <= 0.0:
        return math.inf
    return float(_EPS * np.linalg.norm(A, 1) / s[j])


@pytest.mark.parametrize(
    ("fast_rate", "ordering_determined"),
    [(1.0e8, True), (1.0e10, False), (1.0e12, False)],
)
def test_stiff_network_keeps_the_known_slow_mode_without_zero_filter(
    fast_rate: float,
    ordering_determined: bool,
) -> None:
    """#113 discrimination: no spectral-radius threshold can swallow the gap.

    Issue #180 restated this test against the float64 backward-error level.
    It used to assert ``gap == 1.074e-5`` at every fast rate, but at 1e12 the
    float64 data do not determine that number. The competitor is the slow
    population pair at ``Re = -1.839e-5``, which lies only ``7.65e-6`` below
    the coherence mode. The first-order error of that pair is
    ``eps * ||A||_1 / s = 3.6e-4`` there, 47x the separation.

    Measured between NumPy 1.26.4 and 2.5.3, the reduced operators differ in
    5 of 225 entries. Four differ by exactly one ulp. The fifth, ``(14, 12)``,
    is pure rounding noise around an exact 0: ``-8.5e-22`` against
    ``-4.0e-22``. That was enough to put an eigenvalue at exactly ``0.0`` and
    make the "gap" ``-0.0`` on 1.26. Neither build resolves the pair. Where
    the old assertion passed, NumPy 2.5.3 returned the real values
    ``-2.24e-5`` and ``-3.52e-5`` instead of ``-1.839e-5 +- 9.69e-6 i``. Over
    400 random one-ulp perturbations, 19 % flipped the assertion. It
    therefore depended on summation order (BLAS build, SIMD path, platform),
    not on the code under test.

    What is asserted now, for every fast rate:

    1. The coherence mode is retained to ``2e-8``, for a structural reason
       rather than by float64 accuracy. The coherence block of the restricted
       operator is decoupled by exact zeros, which is a premise the test
       checks bit for bit, so LAPACK's balancing isolates those eigenvalues
       as 1x1 blocks. Hence the reported gap can never exceed the true slow
       mode, which is the #113 failure, where a fast mode was reported.
    2. Every eigenvalue that outranks the coherence mode lies within the
       error band of the exact population slow pair, so it is the unresolved
       slow pair and not a spurious mode. The band is ``n`` times the LAPACK
       estimate, with ``n = 15``. The factor covers the formation of
       ``B^H L B`` itself: its entries are length-``n`` inner products and
       carry up to ``n * u`` relative error before the eigensolver adds its
       own. Without the factor, 3 of 400 four-ulp perturbations at 1e12
       escaped the bare estimate. With it, the band is still about 1e14 times
       narrower than the distance to the fast mode at ``-fast_rate``, the
       failure #113 is about.
    3. Where that SAME band is narrower than the separation, the arithmetic
       determines the ordering, no eigenvalue may outrank the coherence mode,
       and ``gap == 1.074e-5`` is exact. One band serves both claims on
       purpose: a looser band for containment and a tighter one for the
       regime would assert an ordering the containment model does not
       guarantee. Measured: at 1e8 the band is ``6.5e-7``, 12x below the
       separation. At 1e10 it is ``6.6e-5``, 8.6x above. With the bare
       estimate, 1e10 would have counted as determined by only 1.7x, and
       16-ulp perturbations then flipped its gap in up to 59 of 400 runs. At
       1e12 the band is 700x above the separation, and even
       ``eps * ||A||_2 = 2.7e-4`` exceeds the pair's whole modulus.
       ``ordering_determined`` pins the regime of each rate, so a fixture or
       solver change that moves a case across the boundary fails loudly
       instead of silently weakening the test. In the undetermined cases the
       computed gap still equals ``1.074e-5`` on every build measured so far.
       It is not asserted there, because the data do not guarantee it.

    Measured sensitivity, by mutating the restricted operator:
    - Caught at every rate: a coherence mode off by 1 %, the slow mode
      replaced by a fast one (#113), a spurious slow coherence mode, and any
      coherence-population coupling.
    - Caught only at 1e8: a stationary zero that reappears inside the
      POPULATION block. At 1e10 and 1e12 such a zero lies inside the band, so
      no float64 test of this fixture can tell it from the unresolved pair.
      The invariance and reconstruction defect tests in this module cover the
      restriction itself against that regression at every scale.
    """
    L_super = _classical_stiff_network(fast_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reduced = restrict_to_traceless(L_super)
        eigenvalues = np.linalg.eigvals(reduced.operator)
    A = reduced.operator
    assert reduced.invariance_defect <= 1.0e-12 * reduced.operator_scale

    # Premise of claim 1, checked exactly: the first d(d-1) = 12 basis columns
    # are off-diagonal matrix units (see ``traceless_basis``), and classical
    # jumps couple no coherence to anything but itself, so the coherence block
    # is diagonal and every entry linking it to the populations is exactly 0.
    n_coh = 12
    coherence = A[:n_coh, :n_coh]
    assert np.count_nonzero(coherence - np.diag(np.diag(coherence))) == 0
    assert np.count_nonzero(A[:n_coh, n_coh:]) == 0
    assert np.count_nonzero(A[n_coh:, :n_coh]) == 0

    # The decoupled coherence block must also carry the right numbers: all 12
    # coherence eigenvalues against their closed form. Each diagonal entry is
    # a sum of at most three rates, so 1e-12 is far above its rounding and far
    # below any physical change. Without this, a regression that moved only
    # one member of the degenerate (1, 3) / (3, 1) pair would pass claim 1 on
    # the strength of its untouched partner.
    np.testing.assert_allclose(
        np.diag(coherence).real, _stiff_coherence_spectrum(fast_rate), rtol=1.0e-12, atol=0.0
    )
    assert np.count_nonzero(np.diag(coherence).imag) == 0

    # Claim 1: the known slow mode is in the spectrum at every fast rate, and
    # therefore no reported slowest mode is faster than it.
    distance = np.min(np.abs(eigenvalues - _STIFF_COHERENCE_MODE))
    assert distance <= _STIFF_REL * abs(_STIFF_COHERENCE_MODE)
    gap = -float(np.max(np.real(eigenvalues)))
    assert gap <= abs(_STIFF_COHERENCE_MODE) * (1.0 + _STIFF_REL)

    # Claim 2: whatever outranks it is the slow population pair, within the
    # LAPACK first-order estimate widened by the formation error of A.
    pair = _stiff_population_slow_pair()
    band = A.shape[0] * max(_lapack_eigenvalue_error_estimate(A, z) for z in pair)
    separation = float(np.min(np.real(_STIFF_COHERENCE_MODE - pair)))
    assert separation == pytest.approx(7.65e-6, rel=1.0e-3)
    threshold = _STIFF_COHERENCE_MODE * (1.0 - _STIFF_REL)
    outranking = eigenvalues[np.real(eigenvalues) > threshold]
    for z in outranking:
        assert np.min(np.abs(pair - z)) <= band, (z, band)

    # Claim 3: regime pin, then exactness where the arithmetic decides it.
    assert (band < separation) is ordering_determined, (band, separation)
    if ordering_determined:
        assert outranking.size == 0
        assert gap == pytest.approx(abs(_STIFF_COHERENCE_MODE), rel=_STIFF_REL, abs=0.0)


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

    # The rejected evidence, as the helper used to assemble it: the trace
    # equation of column 0 accumulated left to right, which is what the old
    # ``vec(I)^H @ L`` did under OpenBLAS. Its partial sums reach
    # ``d/2 * upper = 2e308`` before the negative half cancels them.
    #
    # Issue #180: the premise is stated for that summation order explicitly
    # rather than through the platform BLAS. On macOS arm64 (Apple Accelerate)
    # the same ``@`` keeps several partial sums and pairs ``+upper`` with
    # ``-upper`` before either overflows, so it returned ``0j`` and this guard
    # reported a fixture failure that was really a BLAS property. The hazard
    # the fix addresses -- some summation order overflows on an exactly
    # trace-preserving operator -- is what the fixture must exhibit, and
    # ``np.add.accumulate`` is sequential by definition on every platform.
    vec_i = np.eye(d, dtype=complex).reshape(-1, order="F")
    with np.errstate(over="ignore", invalid="ignore"):
        partial_sums = np.add.accumulate(vec_i.conj() * L_super[:, 0])
    assert not np.all(np.isfinite(partial_sums)), (
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


def test_a_defect_far_below_the_dominant_scale_is_still_refused() -> None:
    """#139, third round: the gate must see a defect it cannot round away.

    Column 0's trace equation is ``1e300 - 1e300 + 1e-300``, whose exact value
    is ``1e-300``. While the column sum was formed under one common scale that
    addend was flushed to zero, the defect read 0.0, and ``tp_rtol=0`` -- the
    strictest request a caller can make -- ACCEPTED a generator that is not
    trace preserving.
    """
    d = 3
    n = d * d
    L_super = np.zeros((n, n), dtype=complex)
    trace_rows = np.arange(0, n, d + 1, dtype=int)
    L_super[trace_rows[0], 0] = 1.0e300
    L_super[trace_rows[1], 0] = -1.0e300
    L_super[trace_rows[2], 0] = 1.0e-300

    exact = sum(Fraction(v) for v in L_super[trace_rows, 0].real.tolist())
    defect, _ = trace_preservation_defect(L_super)
    assert defect == float(abs(exact)) == 1.0e-300

    with pytest.raises(Exception, match="not trace preserving") as exc:
        restrict_to_traceless(L_super, tp_rtol=0.0)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def _cross_band_residual() -> np.ndarray:
    """A finite 144x144 operator whose column-0 trace defect is exactly 1e-300.

    Its twelve trace coefficients overflow ``math.fsum`` and, split at the old
    band boundary ``2**512``, lose the 1e-300 to one band's own rounding.
    """
    column = [7e307] * 3 + [-7e307] * 3 + [2.0**513] + [-(2.0**511)] * 4 + [1e-300]
    d = 12
    n = d * d
    L_super = np.zeros((n, n), dtype=complex)
    trace_rows = np.arange(0, n, d + 1, dtype=int)
    assert trace_rows.size == len(column)
    L_super[trace_rows, 0] = column

    assert np.all(np.isfinite(L_super))
    assert float(sum(Fraction(v) for v in column)) == 1.0e-300
    return L_super


def test_a_cross_band_residual_is_refused_for_its_exact_size() -> None:
    """#139 fifth round, then #151: refused, and for the right reason.

    While the two exponent bands were rounded independently the defect read
    0.0 and ``tp_rtol=0`` -- the strictest request there is -- ACCEPTED this
    operator. The interim fix refused it as "not representable", which was
    safe but did not say what was wrong. The exact accumulator measures the
    defect itself, so the refusal now comes from the trace-preservation gate
    reading the true number.
    """
    L_super = _cross_band_residual()

    defect, _ = trace_preservation_defect(L_super)
    assert defect == 1.0e-300

    with pytest.raises(Exception, match="not trace preserving") as exc:
        restrict_to_traceless(L_super, tp_rtol=0.0)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_the_same_cross_band_residual_is_admitted_within_tolerance() -> None:
    """#151: the abstention's cost, measured on the case that paid it.

    A defect of 1e-300 on an operator of norm 1.7e308 is a relative violation
    of order 1e-608 -- invisible at double precision and far inside the default
    ``tp_rtol``. The interim fix could not measure it and so refused the
    operator at EVERY tolerance, the default included. Discrimination against
    the test above: same operator, same gates, only the tolerance differs.
    """
    L_super = _cross_band_residual()

    reduced = restrict_to_traceless(L_super)

    assert reduced.trace_defect == 1.0e-300
    assert reduced.trace_defect <= 1.0e-10 * reduced.operator_scale
    assert reduced.trace_componentwise_error <= 1.0e-10


@pytest.mark.parametrize("fast_rate", [1.0e8, 1.0e10, 1.0e12])
def test_a_legitimate_stiff_generator_never_reaches_the_band_fallback(
    fast_rate: float,
) -> None:
    """Positive control: the overflow fallback is not on the physical path.

    Written when the fallback could still abstain (#139); since #151 it answers
    exactly, and the point that remains is where it runs. This measures rather
    than assumes whether a physical stiff generator's trace equations ever
    overflow ``math.fsum`` at all. They do not -- the fallback is unreachable
    for them, so #151 cannot move their numbers -- and the restriction still
    succeeds with a finite defect.
    """
    L_super = _classical_stiff_network(fast_rate)
    n = L_super.shape[0]
    d = int(round(np.sqrt(n)))

    for column in range(n):
        terms = L_super[np.arange(0, n, d + 1, dtype=int), column]
        math.fsum(terms.real.tolist())
        math.fsum(terms.imag.tolist())

    reduced = restrict_to_traceless(L_super)

    assert np.isfinite(reduced.trace_defect)
    assert reduced.trace_defect <= 1.0e-10 * reduced.operator_scale


# --------------------------------------------------------------------------
# Issue #150: the audit numbers are a postcondition, not only a report.
# --------------------------------------------------------------------------


def _random_gksl(rng: np.random.Generator, d: int, n_jumps: int = 3) -> np.ndarray:
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    jumps = [rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)) for _ in range(n_jumps)]
    return build_liouvillian((a + a.conj().T) / 2.0, jumps, list(rng.uniform(0.1, 2.0, n_jumps)))


def _within_column_dilution() -> np.ndarray:
    """The fixture of issue #150: ``1e300 - 1e300 + 1`` in one trace equation."""
    L_super = np.zeros((9, 9), dtype=complex)
    L_super[0, 2] = 1.0e300
    L_super[4, 2] = -1.0e300
    L_super[8, 2] = 1.0
    return L_super


def _arithmetic_unit(reduced: traceless_module.TracelessRestriction) -> float:
    """``eps * ||L||_F + n * d * 2**-1074``: the bound per unit ``reduction_rtol``."""
    n = reduced.basis.shape[0]
    d = math.isqrt(n)
    return _EPS * reduced.operator_scale + n * d * math.ulp(0.0)


def test_the_issue_150_fixture_is_admitted_because_its_defects_are_round_off() -> None:
    """Step 1 of #150, decided: the admission is correct.

    ``invariance_defect = 0.577`` and ``reconstruction_defect = 4.2e284`` look
    alarming only in absolute terms beside an operator of norm ``1.4e300``.
    Measured against the identities the reduction rests on they are exact:
    the invariance defect sits ON its projection bound ``||q^H L||``, and the
    reconstruction defect exceeds it by about ``1.3 * eps * ||L||_F`` -- the
    rounding of forming ``B^H L B``, not a subspace that fails to be invariant.

    Issue #180: "ON its projection bound" holds to rounding, not bit for bit.
    The trace equation ``(1e300 - 1e300 + 1) / sqrt(3)`` is the projection of
    one exact cancellation. Summed left to right without FMA (OpenBLAS) it
    yields ``0.577`` exactly. A fused multiply-add (Apple Accelerate) carries
    the rounding residue of the first product into the second, giving
    ``1.3964842080734432e+283`` -- reproduced bit for bit with ``math.fma``.
    That residue is ``0.044`` arithmetic units, and the reconstruction excess
    stays ``1.30..1.34`` units under every order measured (sequential,
    reversed, with and without FMA). The identity is therefore asserted to one
    arithmetic unit, the rounding of a single product of this operator.
    """
    reduced = restrict_to_traceless(_within_column_dilution())

    projection_bound = reduced.trace_defect / math.sqrt(3.0)
    deviation = abs(reduced.invariance_defect - projection_bound) / _arithmetic_unit(reduced)
    assert deviation <= 1.0, f"invariance defect {deviation:.3g} units off its bound"
    excess = (reduced.reconstruction_defect - reduced.invariance_defect) / _arithmetic_unit(reduced)
    assert 0.0 < excess < 10.0, f"reconstruction excess {excess:.3g} x arithmetic unit"


def test_the_reduction_gate_is_not_empty_on_that_fixture() -> None:
    """Positive control, measured: the same gate refuses the same input below it.

    A postcondition no input can fail proves nothing. The fixture's own
    reconstruction excess is read first and the tolerance set to half of it,
    so the refusal is guaranteed to come from the reconstruction comparison
    and not from a constant that happens to fit this platform's rounding.
    """
    L_super = _within_column_dilution()
    reduced = restrict_to_traceless(L_super)
    excess = (reduced.reconstruction_defect - reduced.invariance_defect) / _arithmetic_unit(reduced)
    assert excess > 0.0

    with pytest.raises(Exception, match="not invariant") as exc:
        restrict_to_traceless(L_super, reduction_rtol=0.5 * excess)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


@pytest.mark.parametrize("d", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("scale", [1.0e-300, 1.0e-150, 1.0, 1.0e150, 1.0e300])
def test_legal_generators_pass_the_reduction_gate_with_margin(d: int, scale: float) -> None:
    """Negative control across dimension and rate unit: nothing legal is refused.

    Both identities hold to well under ``10 * eps * ||L||_F`` here (measured
    worst case 0.9), so the default multiplier ``ZERO_MODE_EPS_FACTOR`` leaves
    two orders of magnitude beyond what this test tolerates.
    """
    rng = np.random.default_rng(1500 + d)
    reduced = restrict_to_traceless(scale * _random_gksl(rng, d))

    unit = _arithmetic_unit(reduced)
    projection_bound = reduced.trace_defect / math.sqrt(d)
    assert reduced.invariance_defect <= projection_bound + 10.0 * unit
    assert reduced.reconstruction_defect <= reduced.invariance_defect + 10.0 * unit
    assert ZERO_MODE_EPS_FACTOR > 10.0


def test_subnormal_scale_generators_are_still_admitted() -> None:
    """The underflow term of the bound, exercised rather than assumed.

    At scale 1e-318 the relative part ``eps * ||L||_F`` underflows to exactly
    zero, while forming ``B^H L B`` still rounds to whole subnormals. Without
    the absolute ``n * d * 2**-1074`` part the gate would refuse legal
    generators that the admission gates -- and ``main`` before #150 -- accept.
    That at least one sample actually carries such a defect is counted, so the
    test cannot pass on a sample that never needed the term.
    """
    needed_the_underflow_term = 0
    for seed in range(20):
        L_super = (_random_gksl(np.random.default_rng(seed), 2) * 1.0e-300) * 1.0e-18
        reduced = restrict_to_traceless(L_super)

        assert _EPS * reduced.operator_scale == 0.0
        if reduced.reconstruction_defect > 0.0:
            needed_the_underflow_term += 1

    assert needed_the_underflow_term > 0


def test_a_subspace_that_is_not_invariant_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fault injection: the basis spans the wrong subspace.

    The diagonal traceless generator ``(e_00 - e_11) / sqrt(2)`` is swapped for
    the trace direction ``(e_00 + e_11) / sqrt(2)``. Amplitude damping maps
    the identity to ``gamma * (|0><0| - |1><1|)``, which leaves that span, so
    ``L B`` has a component outside ``span(B)`` of order ``gamma``. The
    invariance defect cannot see this -- ``q^H L = 0`` for a trace-preserving
    generator whatever ``B`` is -- which is exactly why the reconstruction
    identity is enforced as well.
    """
    L_super = _amplitude_damped_qubit()
    restrict_to_traceless(L_super)  # control: the true basis is accepted

    def wrong_basis(d: int) -> np.ndarray:
        basis = traceless_basis(d)
        basis[:, -1] = trace_vector(d)
        return basis

    monkeypatch.setattr(traceless_module, "traceless_basis", wrong_basis)
    with pytest.raises(Exception, match="not invariant") as exc:
        restrict_to_traceless(L_super)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_a_basis_that_is_not_orthonormal_breaks_the_projection_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fault injection: ``||B|| = 1.2``, so the projection can outgrow its vector.

    The operator carries a trace defect ``q^H L = e_1^T / sqrt(2)``, admitted
    on purpose at ``tp_rtol=1``, which lies in ``q^perp``: with the true basis
    the invariance defect sits exactly ON its bound ``||q^H L|| = 1/sqrt(2)``
    and is accepted. Scaling the basis by 1.2 lifts it to ``1.2/sqrt(2)``. The
    factor is chosen below ``sqrt(2)`` so that the ``/ sqrt(d)`` normalisation
    of the bound is tested too: an unnormalised bound of 1 would admit it.
    """
    L_super = np.zeros((4, 4), dtype=complex)
    L_super[0, 1] = 1.0
    control = restrict_to_traceless(L_super, tp_rtol=1.0)
    assert control.invariance_defect == pytest.approx(1.0 / math.sqrt(2.0), rel=1.0e-15)

    monkeypatch.setattr(traceless_module, "traceless_basis", lambda d: 1.2 * traceless_basis(d))
    with pytest.raises(Exception, match="projection bound") as exc:
        restrict_to_traceless(L_super, tp_rtol=1.0)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_the_reduction_tolerance_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(Exception, match="reduction_rtol") as exc:
        restrict_to_traceless(_amplitude_damped_qubit(), reduction_rtol=bad)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_a_reduction_bound_that_overflows_is_refused_not_trusted() -> None:
    """A finite ``reduction_rtol`` can still make the bound ``inf``.

    ``inf`` would admit every reduction, so -- as ``operator_zero_tolerance``
    does for its own derived threshold -- the operator is refused instead.
    """
    with pytest.raises(Exception, match="not finite") as exc:
        restrict_to_traceless(_amplitude_damped_qubit(rate=1.0e300), reduction_rtol=1.0e300)
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


@pytest.mark.parametrize(
    ("invariance", "reconstruction", "bound", "arithmetic"),
    [
        (math.nan, 0.0, 1.0, 1.0),
        (0.0, math.nan, 1.0, 1.0),
        (0.0, math.inf, 1.0, 1.0),
        (math.inf, math.inf, 1.0, 1.0),
        (0.0, 0.0, math.nan, 1.0),
        (0.0, 0.0, 1.0, math.inf),
        # PR #169 review (Codex P2): an infinite reading against a limit that
        # has itself overflowed -- ``inf <= inf`` is true, and this passed.
        (math.inf, math.inf, 1.0e308, 1.0e308),
        (1.0, math.inf, 1.0e308, 1.0e308),
    ],
)
def test_a_non_finite_reading_refuses_instead_of_comparing_false(
    invariance: float, reconstruction: float, bound: float, arithmetic: float
) -> None:
    """A reading that is not a finite number is refused by name, never compared.

    ``x > bound`` is False for NaN, and ``not (x <= bound)`` catches that --
    but not an infinite ``x`` against a bound that overflowed to ``inf``.
    Finiteness is therefore checked explicitly before any comparison.
    """
    with pytest.raises(Exception, match="not representable") as exc:
        _require_invariant_reduction(
            invariance_defect=invariance,
            reconstruction_defect=reconstruction,
            projection_bound=bound,
            arithmetic=arithmetic,
        )
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_a_limit_that_overflows_is_refused_even_for_finite_readings() -> None:
    """Finite readings, finite inputs, and still no verdict: the limit is ``inf``.

    ``1e308 + 1e308`` overflows, so ``projection_bound + arithmetic`` admits
    every reading. That is the same situation ``operator_zero_tolerance``
    refuses for its own threshold, and it is refused here for the same
    reason: an unusable limit is not a pass.
    """
    with pytest.raises(Exception, match="overflows") as exc:
        _require_invariant_reduction(
            invariance_defect=1.0,
            reconstruction_defect=1.0,
            projection_bound=1.0e308,
            arithmetic=1.0e308,
        )
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_the_postcondition_admits_its_own_boundary() -> None:
    """Both comparisons are inclusive: equality to the bound is a pass."""
    _require_invariant_reduction(
        invariance_defect=2.0,
        reconstruction_defect=3.0,
        projection_bound=1.0,
        arithmetic=1.0,
    )


def test_the_applied_reduction_tolerance_is_recorded_with_the_result() -> None:
    """The gate's own bound travels with the numbers it admitted.

    A postcondition that is enforced but not reported would leave a reader
    of the result unable to tell how close the admitted defects came to it.
    The recorded value is the bound itself, so ``reduction_rtol`` re-derives
    it and both admitted defects verifiably sit below it.
    """
    L_super = _amplitude_damped_qubit(rate=0.3)
    reduced = restrict_to_traceless(L_super)

    assert reduced.reduction_tolerance == ZERO_MODE_EPS_FACTOR * _arithmetic_unit(reduced)
    assert reduced.reconstruction_defect <= reduced.invariance_defect + reduced.reduction_tolerance

    halved = restrict_to_traceless(L_super, reduction_rtol=ZERO_MODE_EPS_FACTOR / 2.0)
    assert halved.reduction_tolerance == reduced.reduction_tolerance / 2.0

    trivial = restrict_to_traceless(np.zeros((1, 1), dtype=complex))
    assert trivial.reduction_tolerance == 0.0
