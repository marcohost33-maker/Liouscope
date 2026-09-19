"""Issue #117: eigenvalue-conditioning evidence for the stationary mode.

These tests pin the two claims the evidence is allowed to make and, just as
deliberately, the one it is NOT allowed to make.

ALLOWED
    * a conditioning-scaled forward estimate explains a stationary eigenvalue
      displacement that the certificate band under-predicts by five orders of
      magnitude (the PR #127 non-normal fixture);
    * the reported figure is the INVARIANT-SUBSPACE condition number, so a
      degenerate or near-degenerate stationary manifold -- physical, and the
      case a per-mode number gets wrong -- is not flagged.

NOT ALLOWED
    * catching the stiff #112 deflation failure. It is invisible to
      conditioning, and the negative control below pins that so no later change
      can quietly start claiming it;
    * changing any D1-D4 value, the certificate, the verdict or the tier.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from liouscope._consts import CONDITIONING_AGREEMENT_FACTOR
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.numerics.conditioning import (
    CONDITIONING_BENIGN,
    CONDITIONING_LIMITED,
    CONDITIONING_UNAVAILABLE,
    ZeroModeConditioning,
    _agreement_band,
    _match_spectra,
    cluster_conditioning,
    eigenvalue_conditioning,
    zero_mode_conditioning,
)
from liouscope.numerics.linalg import (
    certified_eigvals,
    eig_nonhermitian,
    trace_preservation_defect,
)
from liouscope.numerics.traceless import trace_vector

SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

# The canonical #112 repro, copied from tests/test_spectral_certificate.py: a
# four-level classical jump network with a ~1e10 rate spread on which raw
# zgeev LOSES the zero mode and the certificate repairs via dgeev-real.
STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
STIFF_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]


def _classical_network(pairs, rates, d=4):
    jumps = []
    for to, frm in pairs:
        jump = np.zeros((d, d), dtype=complex)
        jump[to, frm] = 1.0
        jumps.append(jump)
    return build_liouvillian(np.zeros((d, d), dtype=complex), jumps, rates)


def _basis_with_first(q: np.ndarray) -> np.ndarray:
    """Unitary whose first column is exactly ``q``."""
    seed = np.eye(q.size, dtype=complex)
    seed[:, 0] = q
    basis, _ = np.linalg.qr(seed)
    phase = np.vdot(basis[:, 0], q)
    return np.asarray(basis * (phase / abs(phase)), dtype=complex)


def _shifted_pr127_fixture() -> np.ndarray:
    """A well-conditioned operator whose stationary mode is far from zero.

    ``s`` is O(1) and the trace defect is tiny, so neither forward estimate can
    account for a stationary eigenvalue at 1e-3: the honest answer is that the
    displacement is NOT explained by conditioning.
    """
    block = np.zeros((4, 4), dtype=complex)
    block[0, 0] = 1.0e-3
    block[1, 1] = -1.0
    block[2, 2] = -2.0
    block[3, 3] = -3.0
    unitary = _basis_with_first(trace_vector(2))
    return unitary @ block @ unitary.conj().T


def _pr127_fixture() -> np.ndarray:
    """The 4x4 non-normal operator from the 2026-09-11 external review of #127.

    A leading block ``[[0, 1e-14], [1, 0]]`` -- exact eigenvalues ``+-1e-7`` --
    written in a basis whose first vector is ``vec(I)/sqrt(2)``, so the
    trace-preservation defect reads ``1.41e-14`` while the stationary eigenvalue
    sits seven orders of magnitude further out.
    """
    block = np.zeros((4, 4), dtype=complex)
    block[0, 1] = 1.0e-14
    block[1, 0] = 1.0
    block[2, 2] = -1.0
    block[3, 3] = -2.0
    unitary = _basis_with_first(trace_vector(2))
    return unitary @ block @ unitary.conj().T


# --------------------------------------------------------------------------
# The discriminating case: conditioning explains what the band cannot
# --------------------------------------------------------------------------


def test_pr127_fixture_band_under_predicts_the_displacement() -> None:
    """The premise: the certificate band is not a forward error bound here."""
    L = _pr127_fixture()
    defect, _scale = trace_preservation_defect(L)
    _values, certificate = certified_eigvals(L)

    # ``trace_preservation_defect`` is the UNNORMALISED ||vec(I)^H L||; the
    # evidence divides it by sqrt(d) to get the perturbation norm.
    assert defect == pytest.approx(1.4145e-14, rel=1e-3)
    assert certificate.bound == pytest.approx(4.4409e-13, rel=1e-3)
    # Observed displacement of the stationary eigenvalue.
    observed = float(np.min(np.abs(np.linalg.eigvals(L))))
    assert observed == pytest.approx(1.0e-7, rel=1e-3)
    # Five orders of magnitude of under-prediction -- the reason a second
    # instrument exists at all.
    assert observed / certificate.bound > 1.0e4


def test_conditioning_scaled_estimate_explains_the_pr127_displacement() -> None:
    """``defect / s`` lands on the observed displacement; the band does not."""
    L = _pr127_fixture()
    evidence = zero_mode_conditioning(L)

    assert evidence.available
    assert evidence.reciprocal_condition == pytest.approx(2.0e-7, rel=1e-3)
    assert evidence.observed_displacement == pytest.approx(1.0e-7, rel=1e-3)
    # The perturbation is the minimum-norm correction ||q^H L|| with q a UNIT
    # vector, i.e. the raw defect over sqrt(d) (PR #166 review). d = 2 here.
    assert evidence.trace_defect == pytest.approx(1.0e-14, rel=1e-3)
    assert evidence.structural_forward_estimate == pytest.approx(5.0e-8, rel=1e-3)
    # First order, so agreement is up to a constant -- measured 2.0x.
    ratio = evidence.observed_displacement / evidence.structural_forward_estimate
    assert 1.0 <= ratio <= CONDITIONING_AGREEMENT_FACTOR
    # And the certificate's own band is nowhere near.
    _values, certificate = certified_eigvals(L)
    assert evidence.structural_forward_estimate / certificate.bound > 1.0e4


def test_pr127_fixture_is_reported_conditioning_limited() -> None:
    """With the layer's own cutoff supplied, the verdict names the cause."""
    L = _pr127_fixture()
    _values, certificate = certified_eigvals(L)
    evidence = zero_mode_conditioning(L, zero_tolerance=certificate.bound)

    assert evidence.conditioning_limited is True
    assert evidence.verdict == CONDITIONING_LIMITED
    assert evidence.displacement_explained is True


# --------------------------------------------------------------------------
# Why the reported figure is the SUBSPACE condition number
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [1.0e-2, 1.0e-6, 1.0e-10, 1.0e-14])
def test_near_degenerate_cluster_is_conditioned_by_its_subspace(delta: float) -> None:
    """Per-mode ``s`` collapses with the splitting; the subspace does not.

    A per-mode conditioning gate would fire here at ``delta = 1e-14`` on a pair
    whose invariant subspace is perfectly determined -- the shape of a
    degenerate stationary manifold, which is physical.
    """
    A = np.array(
        [[0.0, 1.0, 0.0], [0.0, delta, 0.0], [0.0, 0.0, -1.0]], dtype=complex
    )
    import scipy.linalg as sla

    values, left, right = sla.eig(A, left=True, right=True)
    idx = np.argsort(np.abs(values))[:2]
    per_mode = eigenvalue_conditioning(right, left)

    # Exactly ``delta / sqrt(1 + delta^2)`` for this pair, which is ``delta``
    # to leading order -- the collapse is the point, not the constant.
    expected = delta / math.hypot(1.0, delta)
    assert float(per_mode[idx].min()) == pytest.approx(expected, rel=1e-8)
    assert cluster_conditioning(right, left, idx) == pytest.approx(1.0, rel=1e-8)


def test_degenerate_stationary_manifold_is_not_flagged() -> None:
    """Two decoupled damped sectors: four zero modes, all benign."""
    d = 4
    jump = np.zeros((d, d), dtype=complex)
    jump[0, 1] = math.sqrt(0.7)
    jump[2, 3] = math.sqrt(0.4)
    L = build_liouvillian(np.zeros((d, d), dtype=complex), [jump])

    evidence = zero_mode_conditioning(L, zero_tolerance=1.0e-10)
    assert evidence.cluster_size > 1
    assert evidence.reciprocal_condition > 0.5
    assert evidence.conditioning_limited is False
    assert evidence.verdict == CONDITIONING_BENIGN


# --------------------------------------------------------------------------
# Semantics: whose operator is being conditioned
# --------------------------------------------------------------------------


def test_conditioning_is_a_property_of_the_operator_as_written() -> None:
    """A diagonal similarity leaves the spectrum and moves the conditioning.

    This is the LAPACK balancing caveat, settled in the direction the
    certificate needs: ``?geev`` back-transforms its eigenvectors, so
    ``|y^H x|`` describes the caller's operator, not a balanced surrogate whose
    condition numbers ``xGEEVX`` would have reported instead.
    """
    rng = np.random.default_rng(7)
    # 9x9 = d=3: ``zero_mode_conditioning`` refuses a side length that cannot be
    # a superoperator, because ``vec(I)^H L = 0`` says nothing about one.
    A = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    scaling = np.diag(
        np.float64([1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e9])
    )
    scaled = scaling @ A @ np.linalg.inv(scaling)

    plain = zero_mode_conditioning(A)
    similar = zero_mode_conditioning(scaled)
    # Same spectrum ...
    assert np.allclose(
        np.sort_complex(np.linalg.eigvals(A)),
        np.sort_complex(np.linalg.eigvals(scaled)),
        rtol=1e-6,
        atol=1e-8,
    )
    # ... radically different conditioning, and the measurement follows the
    # operator rather than the spectrum.
    assert plain.spectrum_min_reciprocal_condition > 1.0e-2
    assert similar.spectrum_min_reciprocal_condition < 1.0e-10


def test_conditioning_is_invariant_under_a_unitary_change_of_basis() -> None:
    """The control for the test above: a unitary similarity must change nothing."""
    system = build_liouvillian(
        0.5 * SIGMA_X, [math.sqrt(0.6) * SIGMA_MINUS, math.sqrt(1e-3) * SIGMA_Z]
    )
    rng = np.random.default_rng(11)
    seed = rng.standard_normal(system.shape) + 1j * rng.standard_normal(system.shape)
    unitary, _ = np.linalg.qr(seed)
    rotated = unitary @ system @ unitary.conj().T

    plain = zero_mode_conditioning(system)
    turned = zero_mode_conditioning(rotated)
    assert turned.reciprocal_condition == pytest.approx(
        plain.reciprocal_condition, rel=1e-8
    )


@pytest.mark.parametrize("factor", [1.0e-200, 1.0e-30, 1.0e30, 1.0e200])
def test_conditioning_is_invariant_under_a_change_of_rate_units(factor: float) -> None:
    """``s(cL) = s(L)``: the eigenvectors do not move when the rates rescale.

    The same scale-covariance issues #108 and #130 demand of every other
    threshold. An implementation that formed ``|y^H x|`` on unnormalised
    vectors would lose a 1e-200 generator to underflow and report ``0.0`` --
    the fail-closed value, but reached by arithmetic failure rather than by
    measurement, which is how a false ``CONDITIONING_LIMITED`` gets published.
    """
    base = build_liouvillian(
        0.5 * SIGMA_X, [math.sqrt(0.6) * SIGMA_MINUS, math.sqrt(1e-3) * SIGMA_Z]
    )
    plain = zero_mode_conditioning(base)
    scaled = zero_mode_conditioning(factor * base)
    assert scaled.available
    assert scaled.reciprocal_condition == pytest.approx(
        plain.reciprocal_condition, rel=1e-8
    )


# --------------------------------------------------------------------------
# Negative controls: what this instrument does NOT detect
# --------------------------------------------------------------------------


def test_the_stiff_112_network_really_loses_its_zero_mode() -> None:
    """Premise of the negative control below, established before it is used.

    PR #166 review: the first version of that control used a two-level family on
    which raw zgeev keeps an exact ``0.0`` zero mode at every rate spread. There
    was no wrong spectrum for conditioning to miss, so the control asserted
    nothing. The canonical #112 network does fail, and this test says so before
    anything is concluded from it.
    """
    L = _classical_network(STIFF_PAIRS, STIFF_RATES)
    raw = eig_nonhermitian(np.asarray(L, dtype=complex)).eigenvalues
    _accepted, certificate = certified_eigvals(L)

    raw_min = float(np.abs(raw).min())
    assert raw_min == pytest.approx(7.28e-6, rel=1e-3)
    assert raw_min > certificate.bound          # the zero mode is genuinely lost
    assert certificate.solver == "dgeev-real"   # and a different route repaired it
    assert float(np.abs(_accepted).min()) < certificate.bound


def test_the_lost_zero_mode_is_not_visible_as_ill_conditioning() -> None:
    """The negative control proper: conditioning cannot catch the #112 failure.

    Run deliberately WITHOUT the accepted spectrum, so the audit conditions the
    wrong spectrum -- the question being exactly what conditioning would have
    said about it. The answer is "well conditioned": the stationary mode's
    reciprocal condition number is 0.29 and the worst over the whole spurious
    spectrum is 0.040, i.e. condition numbers between 1 and 25. That is the same
    range already recorded in ``numerics/linalg.py`` from the 2026-08-25
    measurement, reached here independently.

    Pinned so no later change can start claiming conditioning detects #112.
    """
    L = _classical_network(STIFF_PAIRS, STIFF_RATES)
    evidence = zero_mode_conditioning(L)

    assert evidence.available
    # It is the SPURIOUS eigenvalue that is being conditioned.
    assert evidence.observed_displacement == pytest.approx(7.28e-6, rel=1e-3)
    # ... and nothing about its conditioning gives the failure away.
    assert evidence.reciprocal_condition > 0.25
    assert evidence.spectrum_min_reciprocal_condition > 0.04
    assert 1.0 / evidence.spectrum_min_reciprocal_condition < 25.0


def test_audit_withholds_rather_than_condition_a_repaired_spectrum() -> None:
    """The evidence must describe the eigenvalues in the same result.

    PR #166 review, finding P2. ``certified_eigvals`` repairs this generator
    through ``dgeev-real``; the audit's own ``?geev`` would otherwise report the
    conditioning of the spurious ``7.28e-6`` mode against a result whose
    stationary eigenvalue is ``4.08e-17``.
    """
    L = _classical_network(STIFF_PAIRS, STIFF_RATES)
    accepted, certificate = certified_eigvals(L)
    evidence = zero_mode_conditioning(
        L,
        zero_tolerance=certificate.zero_set_tolerance(accepted),
        eigenvalues=accepted,
    )
    assert evidence.available is False
    assert evidence.reason == "accepted_spectrum_mismatch"
    assert evidence.verdict == CONDITIONING_UNAVAILABLE


def test_an_unrepaired_spectrum_still_yields_evidence() -> None:
    """The control for the guard above: it must not withhold on healthy input."""
    L = build_liouvillian(0.5 * SIGMA_X, [math.sqrt(0.6) * SIGMA_MINUS])
    accepted, certificate = certified_eigvals(L)
    assert certificate.solver == "zgeev"
    evidence = zero_mode_conditioning(
        L,
        zero_tolerance=certificate.zero_set_tolerance(accepted),
        eigenvalues=accepted,
    )
    assert evidence.available is True
    assert evidence.verdict == CONDITIONING_BENIGN


@pytest.mark.parametrize("spread", [1.0e8, 1.0e11, 1.0e13, 1.0e15])
def test_stiff_but_correctly_solved_generators_are_not_flagged(spread: float) -> None:
    """No false alarm from stiffness alone when the solver copes.

    This is a no-false-alarm check, NOT a #112 control: on this family raw zgeev
    keeps an exact zero mode at every spread (asserted, since that is the very
    assumption the first version of this file got wrong).
    """
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [math.sqrt(spread) * SIGMA_MINUS, SIGMA_Z],
    )
    accepted, certificate = certified_eigvals(L)
    assert float(np.abs(accepted).min()) == 0.0  # nothing was lost here
    evidence = zero_mode_conditioning(
        L,
        zero_tolerance=certificate.zero_set_tolerance(accepted),
        eigenvalues=accepted,
    )
    assert evidence.available
    assert evidence.spectrum_min_reciprocal_condition > 0.5
    assert evidence.conditioning_limited is False


@pytest.mark.parametrize("gamma", [1.0e0, 1.0e3, 1.0e6, 1.0e9])
@pytest.mark.parametrize("omega", [0.0, 1.0, 1.0e3, 1.0e6])
def test_physical_gksl_generators_are_never_conditioning_limited(
    gamma: float, omega: float
) -> None:
    """No false alarm on the family the pipeline actually sees."""
    L = build_liouvillian(
        0.5 * omega * SIGMA_X,
        [math.sqrt(gamma) * SIGMA_MINUS, math.sqrt(1e-9) * SIGMA_Z],
    )
    _values, certificate = certified_eigvals(L)
    evidence = zero_mode_conditioning(
        L,
        zero_tolerance=certificate.zero_set_tolerance(_values),
        eigenvalues=_values,
    )
    assert evidence.available
    assert evidence.reciprocal_condition > 0.5
    assert evidence.conditioning_limited is False
    assert evidence.displacement_explained is True


# --------------------------------------------------------------------------
# The audit-only contract
# --------------------------------------------------------------------------


def test_audit_changes_no_reported_diagnostic() -> None:
    """Every D1-D4 value and the certificate are bit-identical with it off."""
    L = build_liouvillian(
        0.5 * SIGMA_X, [math.sqrt(0.6) * SIGMA_MINUS, math.sqrt(1e-3) * SIGMA_Z]
    )
    with_audit = compute_spectral_layer(L)
    without = compute_spectral_layer(L, conditioning_audit=False)

    assert without.zero_mode_conditioning is None
    assert with_audit.zero_mode_conditioning is not None
    for field in ("gap", "gns_gap", "kms_gap", "oscillating_gap", "spectral_spread"):
        assert getattr(with_audit, field) == getattr(without, field)
    assert with_audit.has_complex_pairs == without.has_complex_pairs
    assert with_audit.zero_mode_certificate == without.zero_mode_certificate
    np.testing.assert_array_equal(with_audit.eigenvalues, without.eigenvalues)


def test_report_view_is_json_serialisable() -> None:
    """RFC 8259 has no NaN or Infinity; every field must survive the dump."""
    payloads = [
        zero_mode_conditioning(_pr127_fixture()).as_dict(),
        ZeroModeConditioning.unavailable("eigensolve_failed").as_dict(),
        # A defective pair drives the estimates to inf; they must still dump.
        zero_mode_conditioning(_defective_superoperator()).as_dict(),
    ]
    for payload in payloads:
        text = json.dumps(payload, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["audit_only"] is True


def _defective_superoperator(defect: float = 0.0) -> np.ndarray:
    """4x4 (d=2) operator with an exactly defective stationary Jordan block.

    The dimension matters: ``trace_preservation_defect`` is defined only when
    the operator side is a perfect square, so a 2x2 Jordan block would report an
    UNDEFINED structural defect instead of exercising the ``s = 0`` branch. In
    column stacking ``vec(I)^H`` reads rows 0 and 3, so making them negatives of
    each other keeps the operator exactly trace preserving while ``e1`` maps to
    ``e0 - e3``: a nilpotent block of index two on the stationary subspace.
    """
    A = np.zeros((4, 4), dtype=complex)
    A[0, 1] = 1.0
    A[3, 1] = -1.0 + defect
    A[2, 2] = -1.0
    return A


def test_defective_pair_fails_closed_without_raising() -> None:
    """A Jordan block has no eigenvector pair; ``s -> 0`` is the honest answer."""
    evidence = zero_mode_conditioning(_defective_superoperator(), zero_tolerance=1e-13)
    assert evidence.available
    assert evidence.cluster_size == 3
    # The SUBSPACE figure is exactly zero: the three vectors do not span a
    # three-dimensional invariant subspace at all.
    assert evidence.reciprocal_condition == 0.0
    # The per-mode figure, which the scalar estimates divide by, is subnormal
    # rather than exactly zero -- LAPACK still returns a direction.
    assert 0.0 < evidence.per_mode_reciprocal_condition < 1.0e-200
    # Exactly trace preserving, so the STRUCTURAL estimate stays 0.0: the 0/0
    # branch reports no displacement for an exact backward error rather than
    # inventing one.
    assert evidence.trace_defect == 0.0
    assert evidence.structural_forward_estimate == 0.0
    # The SOLVER estimate is what is unusable here, and the verdict must read
    # it -- a defective stationary pair is conditioning-limited even when the
    # generator is exactly trace preserving. The residual over a subnormal
    # conditioning is order one, thirteen decades past the supplied cutoff.
    assert evidence.solver_forward_estimate >= 1.0
    assert evidence.solver_forward_estimate > evidence.zero_tolerance
    assert evidence.conditioning_limited is True
    assert evidence.verdict == CONDITIONING_LIMITED


def test_defective_pair_with_a_nonzero_defect_reports_unbounded_displacement() -> None:
    """A real backward error over a vanishing ``s`` means an unusable location."""
    evidence = zero_mode_conditioning(
        _defective_superoperator(defect=1.0e-12), zero_tolerance=1.0e-13
    )
    assert evidence.available
    assert evidence.trace_defect == pytest.approx(1.0e-12 / math.sqrt(2.0), rel=1e-3)
    # 1e-12 over a subnormal conditioning: enormous, and far past any tolerance
    # a caller could sensibly supply.
    assert evidence.structural_forward_estimate > 1.0e200
    assert evidence.conditioning_limited is True
    assert evidence.verdict == CONDITIONING_LIMITED


def test_a_nonrepresentable_forward_estimate_withholds_the_verdict() -> None:
    """Round-2 review: a verdict may not rest on an estimate that does not exist.

    ``diag(1.3e308, 0, -1, 1.3e308)`` is finite, but its trace-defect norm is
    not representable, so the structural estimate is NaN. Dropping it from the
    budget left a zero solver residual to carry the verdict, and the audit
    published ``BENIGN`` on a structural bound it did not have.
    """
    evidence = zero_mode_conditioning(
        np.diag([1.3e308, 0.0, -1.0, 1.3e308]).astype(complex)
    )
    assert evidence.available is False
    assert evidence.reason == "forward_estimate_unavailable"
    assert evidence.verdict == CONDITIONING_UNAVAILABLE
    assert json.dumps(evidence.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[np.nan, 0.0], [0.0, -1.0]]),
        np.array([[np.inf, 0.0], [0.0, -1.0]]),
        np.zeros((0, 0)),
        np.zeros((2, 3)),
    ],
)
def test_audit_never_raises_on_unusable_input(bad: np.ndarray) -> None:
    """An audit that can abort the analysis it audits is a liability."""
    evidence = zero_mode_conditioning(bad)
    assert evidence.available is False
    assert evidence.verdict == CONDITIONING_UNAVAILABLE
    assert evidence.reason != "ok"
    assert json.dumps(evidence.as_dict(), allow_nan=False)


def test_negative_zero_tolerance_is_rejected() -> None:
    """PR #166 review, finding P2.

    A negative cutoff admits no eigenvalue to the zero set, after which the
    ``budget > tol`` comparison labels even an exact, well-conditioned zero mode
    ``CONDITIONING_LIMITED``. Valid-looking nonsense out of an exported public
    function is worse than an error, so it is a caller contract violation --
    unlike unusable OPERATOR input, which still comes back as UNAVAILABLE.
    """
    L = build_liouvillian(0.5 * SIGMA_X, [math.sqrt(0.6) * SIGMA_MINUS])
    with pytest.raises(ValueError, match="non-negative"):
        zero_mode_conditioning(L, zero_tolerance=-1.0)
    # Zero is a legitimate cutoff and must not be swept up with it.
    assert zero_mode_conditioning(L, zero_tolerance=0.0).available


@pytest.mark.parametrize("side", [2, 3, 6, 7])
def test_a_side_that_cannot_be_a_superoperator_is_unavailable(side: int) -> None:
    """PR #166 review, finding P2.

    ``vec(I)^H L = 0`` is not a statement about a matrix whose side is not
    ``d*d``, so ``trace_preservation_defect`` is NaN there. Publishing
    ``available=True`` / ``BENIGN`` on top of that would rest a benign verdict
    on a structural estimate that does not exist.
    """
    rng = np.random.default_rng(side)
    evidence = zero_mode_conditioning(rng.standard_normal((side, side)))
    assert evidence.available is False
    assert evidence.reason == "dimension_not_a_superoperator"
    assert evidence.verdict == CONDITIONING_UNAVAILABLE


def test_forward_estimates_do_not_depend_on_the_supplied_tolerance() -> None:
    """The cutoff must not leak into quantities that are about the operator.

    Two round-2 findings meet here. ``displacement_explained`` used to take the
    tolerance into its budget, which made it true by construction for any mode
    inside the zero set. And the scalar estimates used to divide by the SUBSPACE
    condition number, which changes the moment a wider cutoff pulls a second
    eigenvalue into the cluster -- for this fixture ``s_cluster`` jumps from
    ``2e-7`` to ``1.0`` between ``tol = 1e-9`` and ``tol = 1e-6``, and the
    estimate moved with it.

    Both are now per-mode quantities, so across eight orders of magnitude of
    cutoff -- and across the cluster-size change at ``1e-6`` -- the estimates and
    the verdict are identical.
    """
    L = _pr127_fixture()
    runs = [zero_mode_conditioning(L, zero_tolerance=t) for t in (1e-13, 1e-9, 1e-6, 1e-5)]

    # The cluster really does change, so the invariant is not vacuous.
    assert {r.cluster_size for r in runs} == {1, 2}
    assert {round(r.reciprocal_condition, 6) for r in runs} == {0.0, 1.0}

    reference = runs[0]
    for run in runs[1:]:
        assert run.per_mode_reciprocal_condition == pytest.approx(
            reference.per_mode_reciprocal_condition, rel=1e-12
        )
        assert run.structural_forward_estimate == pytest.approx(
            reference.structural_forward_estimate, rel=1e-12
        )
        assert run.solver_forward_estimate == pytest.approx(
            reference.solver_forward_estimate, rel=1e-12
        )
        assert run.displacement_explained == reference.displacement_explained


def test_displacement_agreement_is_true_when_the_estimates_earn_it() -> None:
    """The positive control: a displacement a forward estimate does account for.

    The operator is well conditioned (``s = 1``) but violates trace preservation
    by ``1e-3``, and its stationary eigenvalue sits exactly that far from zero.
    The structural estimate carries the whole distance, so ``True`` here is
    earned by the evidence and not by the cutoff.
    """
    evidence = zero_mode_conditioning(_shifted_pr127_fixture(), zero_tolerance=1.0e-2)

    assert evidence.reciprocal_condition == pytest.approx(1.0, rel=1e-8)
    assert evidence.observed_displacement == pytest.approx(1.0e-3, rel=1e-6)
    assert evidence.structural_forward_estimate == pytest.approx(1.0e-3, rel=1e-6)
    assert evidence.displacement_explained is True


def test_a_wider_dtype_that_overflows_complex128_is_unavailable() -> None:
    """Round-3 review: the narrowing cast must not escape the totality contract.

    ``np.longdouble`` holding 1e400 is finite and passes validation, but casting
    it to complex128 emits ``RuntimeWarning: overflow encountered in cast``.
    Under the suite's ``filterwarnings = ["error"]`` that RAISED out of a
    function whose whole contract is that it does not. The suite's own filter
    makes this test the enforcement: no ``pytest.warns``, no suppression.
    """
    wide = np.zeros((4, 4), dtype=np.longdouble)
    wide[0, 0] = np.longdouble("1e400")
    wide[1, 1] = -1.0
    assert bool(np.all(np.isfinite(wide)))  # finite in ITS OWN dtype

    evidence = zero_mode_conditioning(wide)
    assert evidence.available is False
    assert evidence.reason == "not_representable_in_complex128"
    assert evidence.verdict == CONDITIONING_UNAVAILABLE


def test_the_two_forward_errors_add_rather_than_compete() -> None:
    """Round-3 review: they are successive perturbations, so the bounds add.

    From the exact zero of the nearest trace-preserving operator, to that
    operator's eigenvalue, to the one the solver returned. Under ``max`` a pair
    of estimates each just below the cutoff reported ``BENIGN`` while permitting
    a total displacement above it; the budget is now the sum, and the same sum
    is what ``displacement_explained`` measures against so the two fields cannot
    disagree.
    """
    evidence = zero_mode_conditioning(_pr127_fixture(), zero_tolerance=1.0e-13)
    assert evidence.available
    both = (
        evidence.structural_forward_estimate + evidence.solver_forward_estimate
    )
    # Each contribution is real and neither is negligible relative to the other
    # being dropped, so summing is not a distinction without a difference.
    assert evidence.structural_forward_estimate > 0.0
    assert evidence.solver_forward_estimate > 0.0
    assert both > max(
        evidence.structural_forward_estimate, evidence.solver_forward_estimate
    )
    # A cutoff sitting between the larger single estimate and the sum is exactly
    # the case ``max`` got wrong: it must read as conditioning-limited.
    between = 0.5 * (
        max(evidence.structural_forward_estimate, evidence.solver_forward_estimate)
        + both
    )
    straddling = zero_mode_conditioning(_pr127_fixture(), zero_tolerance=between)
    assert straddling.conditioning_limited is True
    assert straddling.verdict == CONDITIONING_LIMITED


def test_the_default_keeps_an_exactly_degenerate_stationary_manifold() -> None:
    """Round-4 review: the default must not contradict "zero SET" conditioning.

    Two-level pure dephasing has spectrum ``[0, 0, -2, -2]``. With no cutoff
    supplied the fallback used to take the single smallest mode, conditioning
    one arbitrary vector of a two-dimensional stationary subspace. Every mode
    exactly tied with the smallest is kept instead -- exact ties need no
    invented threshold, and a looser one would be a second zero-mode tolerance
    competing with the certificate's.
    """
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [SIGMA_Z])
    accepted, certificate = certified_eigvals(L)
    assert int(np.count_nonzero(np.abs(accepted) == 0.0)) == 2  # the premise

    default = zero_mode_conditioning(L)
    with_cutoff = zero_mode_conditioning(
        L, zero_tolerance=certificate.zero_set_tolerance(accepted)
    )
    assert default.cluster_size == 2
    assert default.cluster_size == with_cutoff.cluster_size


def test_spectrum_matching_is_permutation_invariant() -> None:
    """Round-4 review: a lexicographic sort mispairs reordered conjugates.

    ``[i, -i]`` against ``[i - 1e-14, -i + 1e-14]`` is the same pair of modes,
    well inside the agreement band, but the two sort into opposite orders under
    ``(real, imag)`` -- the old comparison paired ``i`` with ``-i``, measured a
    distance of 2, and discarded evidence for a spectrum it should have taken.
    """
    audit = np.array([1j, -1j], dtype=complex)
    accepted = np.array([1j - 1e-14, -1j + 1e-14], dtype=complex)

    matched = _match_spectra(audit, accepted)
    assert matched is not None
    _perm, worst = matched
    assert worst == pytest.approx(1.0e-14, rel=1e-6)
    assert worst <= _agreement_band(audit, accepted)
    # A genuinely different spectrum still fails.
    assert _match_spectra(audit, np.array([1.0, -1.0], dtype=complex))[1] > 1.0


def test_scalar_estimates_use_the_selected_mode_not_the_cluster() -> None:
    """Round-4 review: the cluster MINIMUM coupled the estimate to the cutoff.

    Trace-basis blocks ``1e-10``, ``[[1e-5, 1], [0, 1.00001e-5]]`` and ``-1``.
    Widening the cutoff leaves the selected stationary eigenvalue at ``1e-10``
    but pulls in a near-defective pair whose own ``s`` is ~1e-10. Under the
    round-2 cluster-minimum the structural estimate moved from ``1e-10`` to
    ``1.0`` -- ten orders of magnitude contributed by modes the stationary one
    has nothing to do with.
    """
    block = np.zeros((4, 4), dtype=complex)
    block[0, 0] = 1.0e-10
    block[1, 1] = 1.0e-5
    block[1, 2] = 1.0
    block[2, 2] = 1.00001e-5
    block[3, 3] = -1.0
    unitary = _basis_with_first(trace_vector(2))
    L = unitary @ block @ unitary.conj().T

    narrow = zero_mode_conditioning(L, zero_tolerance=1.0e-9)
    wide = zero_mode_conditioning(L, zero_tolerance=2.0e-5)

    # The cutoff really does change the cluster, so the test is not vacuous ...
    assert narrow.cluster_size == 1
    assert wide.cluster_size == 3
    # ... while selecting the same stationary eigenvalue ...
    assert narrow.observed_displacement == pytest.approx(
        wide.observed_displacement, rel=1e-9
    )
    # ... so its conditioning and its estimates must not move.
    assert wide.per_mode_reciprocal_condition == pytest.approx(
        narrow.per_mode_reciprocal_condition, rel=1e-9
    )
    assert wide.structural_forward_estimate == pytest.approx(
        narrow.structural_forward_estimate, rel=1e-9
    )


def test_zero_set_membership_follows_the_accepted_spectrum() -> None:
    """Round-4 review: the report filters the accepted spectrum, so must this.

    Two spectra can agree well within the band and still disagree about which
    modes fall inside the cutoff. The evidence has to describe the set the
    report filtered, not the set the audit's own solve would have.
    """
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [SIGMA_Z])
    accepted, certificate = certified_eigvals(L)
    tol = certificate.zero_set_tolerance(accepted)

    evidence = zero_mode_conditioning(L, zero_tolerance=tol, eigenvalues=accepted)
    assert evidence.available
    assert evidence.cluster_size == int(np.count_nonzero(np.abs(accepted) <= tol))
    # The reported eigenvalue is the accepted one, not the audit solve's.
    assert complex(evidence.eigenvalue) in set(np.asarray(accepted, dtype=complex))


def test_cluster_conditioning_fails_closed_on_unusable_vector_sets() -> None:
    """``0.0`` is the fail-closed value: it maximises every estimate built on it."""
    eye = np.eye(2, dtype=complex)
    # An empty cluster conditions nothing.
    assert cluster_conditioning(eye, eye, np.array([], dtype=int)) == 0.0
    # Two parallel columns do not span a two-dimensional invariant subspace.
    parallel = np.zeros((3, 2), dtype=complex)
    parallel[:, 0] = [1.0, 0.0, 0.0]
    parallel[:, 1] = [1.0, 0.0, 0.0]
    assert cluster_conditioning(parallel, parallel, np.array([0, 1])) == 0.0
    # A column with no direction at all.
    degenerate = np.zeros((3, 2), dtype=complex)
    degenerate[:, 0] = [1.0, 0.0, 0.0]
    assert cluster_conditioning(degenerate, degenerate, np.array([0, 1])) == 0.0


def test_a_spectrum_of_a_different_size_is_a_mismatch() -> None:
    """The accepted-spectrum guard must not be fooled by a truncated list."""
    L = build_liouvillian(0.5 * SIGMA_X, [math.sqrt(0.6) * SIGMA_MINUS])
    accepted, _certificate = certified_eigvals(L)
    short = zero_mode_conditioning(L, eigenvalues=accepted[:-1])
    assert short.available is False
    assert short.reason == "accepted_spectrum_mismatch"
    # A NaN in the accepted spectrum is likewise not something to agree with.
    poisoned = np.asarray(accepted, dtype=complex).copy()
    poisoned[0] = np.nan
    assert zero_mode_conditioning(L, eigenvalues=poisoned).available is False


def test_eigenvalue_conditioning_rejects_mismatched_vector_matrices() -> None:
    with pytest.raises(ValueError, match="equally shaped"):
        eigenvalue_conditioning(np.eye(3, dtype=complex), np.eye(2, dtype=complex))
