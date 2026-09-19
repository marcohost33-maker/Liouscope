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
    cluster_conditioning,
    eigenvalue_conditioning,
    zero_mode_conditioning,
)
from liouscope.numerics.linalg import certified_eigvals, trace_preservation_defect
from liouscope.numerics.traceless import trace_vector

SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _basis_with_first(q: np.ndarray) -> np.ndarray:
    """Unitary whose first column is exactly ``q``."""
    seed = np.eye(q.size, dtype=complex)
    seed[:, 0] = q
    basis, _ = np.linalg.qr(seed)
    phase = np.vdot(basis[:, 0], q)
    return np.asarray(basis * (phase / abs(phase)), dtype=complex)


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
    # First order, so agreement is up to a constant -- measured 1.41x.
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
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    scaling = np.diag(np.float64([1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9]))
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


@pytest.mark.parametrize("spread", [1.0e8, 1.0e11, 1.0e13, 1.0e15])
def test_stiff_112_family_is_benign_to_conditioning(spread: float) -> None:
    """Pinned so no later change claims conditioning catches the #112 failure.

    The Ahues-Tisseur deflation destroys the slow spectrum before any
    conditioning estimate can see it, so the WRONG answer is reported as
    perfectly well conditioned. Measured: every per-mode ``s`` stays at 0.707
    across four decades of rate spread.
    """
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [math.sqrt(spread) * SIGMA_MINUS, SIGMA_Z],
    )
    evidence = zero_mode_conditioning(L)
    assert evidence.available
    assert evidence.spectrum_min_reciprocal_condition > 0.5
    assert evidence.verdict == CONDITIONING_BENIGN


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
        L, zero_tolerance=certificate.zero_set_tolerance(_values)
    )
    assert evidence.reciprocal_condition > 0.5
    assert evidence.conditioning_limited is False


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
        zero_mode_conditioning(np.array([[0.0, 1.0], [0.0, 0.0]])).as_dict(),
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
    """A Jordan block has no eigenvector pair; ``s = 0`` is the honest answer."""
    evidence = zero_mode_conditioning(_defective_superoperator(), zero_tolerance=1e-13)
    assert evidence.available
    assert evidence.cluster_size == 3
    assert evidence.reciprocal_condition == 0.0
    # Exactly trace preserving, so the STRUCTURAL estimate stays 0.0: the 0/0
    # branch reports no displacement for an exact backward error rather than
    # inventing an infinite one.
    assert evidence.trace_defect == 0.0
    assert evidence.structural_forward_estimate == 0.0
    # The SOLVER estimate is what is unbounded here, and the verdict must read
    # it -- a defective stationary pair is conditioning-limited even when the
    # generator is exactly trace preserving.
    assert evidence.solver_forward_estimate == math.inf
    assert evidence.conditioning_limited is True
    assert evidence.verdict == CONDITIONING_LIMITED


def test_defective_pair_with_a_nonzero_defect_reports_unbounded_displacement() -> None:
    """``s = 0`` with a real backward error means the location is unbounded."""
    evidence = zero_mode_conditioning(
        _defective_superoperator(defect=1.0e-12), zero_tolerance=1.0e-13
    )
    assert evidence.available
    assert evidence.reciprocal_condition == 0.0
    assert evidence.trace_defect == pytest.approx(1.0e-12, rel=1e-3)
    assert evidence.structural_forward_estimate == math.inf
    assert evidence.conditioning_limited is True
    assert evidence.verdict == CONDITIONING_LIMITED


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


def test_eigenvalue_conditioning_rejects_mismatched_vector_matrices() -> None:
    with pytest.raises(ValueError, match="equally shaped"):
        eigenvalue_conditioning(np.eye(3, dtype=complex), np.eye(2, dtype=complex))
