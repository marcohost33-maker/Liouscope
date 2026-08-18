"""Issue #109: Hermiticity validation must not depend on the units of H.

``H`` carries energy/rate dimension, so an ABSOLUTE tolerance made the
validation verdict a function of the caller's unit choice. The tests below pin
both failure directions that were measured on ``main``, the unit-scale
compatibility claim, and -- crucially -- a DISCRIMINATION test proving the
legacy ``atol=`` opt-in genuinely restores the old defect. Without that last
test a no-op "fix" would pass every invariance assertion here.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.core.lindblad import build_liouvillian
from liouscope.numerics.linalg import (
    hermiticity_defect,
    is_density_matrix,
    is_hermitian,
)
from liouscope.sparse.build import build_sparse_liouvillian

SCALES = (1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12)


def _hermitian_base(seed: int = 7, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    h = a + a.conj().T
    return h / np.linalg.norm(h, 2)


def _antihermitian_unit(seed: int = 11, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    k = a - a.conj().T
    return k / np.linalg.norm(k, 2)


def _broken(scale: float, rel_defect: float) -> np.ndarray:
    """H with a RELATIVE Hermiticity defect -- the same broken input at every scale."""
    return scale * (_hermitian_base() + rel_defect * _antihermitian_unit())


# --------------------------------------------------------------------------
# Direction 1: fail-open. The serious one -- a non-GKSL generator gets built.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", SCALES)
def test_relative_defect_is_rejected_at_every_scale(scale: float) -> None:
    """A relative defect of 1e-6 is the same broken H in any unit system."""
    h = _broken(scale, 1e-6)
    assert not is_hermitian(h)
    with pytest.raises(ValueError, match="Hermitian"):
        build_liouvillian(h, [], [])


@pytest.mark.parametrize("scale", SCALES)
def test_sparse_builder_matches_dense_verdict(scale: float) -> None:
    """The sparse path must not accept what the dense path rejects."""
    h = _broken(scale, 1e-6)
    with pytest.raises(ValueError, match="Hermitian"):
        build_sparse_liouvillian(h, [], [])


def test_fail_open_was_real_on_the_legacy_absolute_gate() -> None:
    """DISCRIMINATION: the legacy atol really did accept this broken H.

    If this ever starts failing, the repro above no longer exercises the defect
    it claims to, and the test above becomes vacuous.
    """
    h = _broken(1e-6, 1e-6)
    defect, scale = hermiticity_defect(h)
    assert defect < 1e-9, "repro must sit below the legacy absolute floor"
    assert defect > 1e-9 * scale, "repro must sit above the new relative floor"
    assert is_hermitian(h, atol=1e-9) is True     # legacy verdict: accepted
    assert is_hermitian(h) is False               # new verdict: rejected


# --------------------------------------------------------------------------
# Direction 2: false rejection of valid input at large scale.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e9, 1e12])
def test_similarity_roundoff_is_accepted_at_every_scale(scale: float) -> None:
    """An exactly Hermitian H reconstructed by a unitary similarity."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    evals = np.linalg.eigvalsh(scale * _hermitian_base())
    h = q @ np.diag(evals) @ q.conj().T
    assert is_hermitian(h)
    build_liouvillian(h, [], [])          # must not raise


def test_false_rejection_was_real_on_the_legacy_absolute_gate() -> None:
    """DISCRIMINATION: the legacy atol really did reject this valid H."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    evals = np.linalg.eigvalsh(1e12 * _hermitian_base())
    h = q @ np.diag(evals) @ q.conj().T
    defect, scale = hermiticity_defect(h)
    assert defect > 1e-9, "repro must sit above the legacy absolute floor"
    assert defect <= 1e-9 * scale, "repro must sit below the new relative floor"
    assert is_hermitian(h, atol=1e-9) is False    # legacy verdict: rejected
    assert is_hermitian(h) is True                # new verdict: accepted


# --------------------------------------------------------------------------
# Invariance and compatibility.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rel_defect", [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3])
def test_verdict_is_invariant_under_rescaling(rel_defect: float) -> None:
    """The whole point: one verdict for one physical input, in any units."""
    verdicts = {is_hermitian(_broken(s, rel_defect)) for s in SCALES}
    assert len(verdicts) == 1, (
        f"relative defect {rel_defect:g} gave different verdicts across scales"
    )


def test_unit_scale_reproduces_the_historical_gate() -> None:
    """Compatibility claim: at max|H| ~ 1 nothing moved."""
    h = _hermitian_base()
    h = h / np.max(np.abs(h))            # max|H| == 1 exactly
    for rel in (1e-12, 1e-10, 1e-8, 1e-6):
        broken = h + rel * _antihermitian_unit()
        assert is_hermitian(broken) == is_hermitian(broken, atol=1e-9)


def test_zero_operator_semantics() -> None:
    """max|A| == 0 -> tol == 0: exactly Hermitian passes, nothing else."""
    assert is_hermitian(np.zeros((3, 3), dtype=complex))
    tiny = np.zeros((3, 3), dtype=complex)
    tiny[0, 1] = 1e-300
    tiny[1, 0] = -1e-300                 # anti-Hermitian, however small
    assert not is_hermitian(tiny)


def test_density_matrix_gate_stays_absolute() -> None:
    """is_density_matrix keeps the absolute reading (rho is trace-normalised)."""
    d = 8
    rho = np.eye(d, dtype=complex) / d   # max|rho| = 1/d, defect 0
    assert is_density_matrix(rho)
    # a defect between the relative and absolute thresholds must still pass,
    # i.e. the density-matrix path did NOT silently tighten by a factor d
    rho_perturbed = rho.copy()
    rho_perturbed[0, 1] += 5e-10j
    rho_perturbed[1, 0] += 5e-10j        # makes A - A^H = 1e-9 in those slots
    assert hermiticity_defect(rho_perturbed)[0] < 1e-9 * d
    assert is_density_matrix(rho_perturbed)


def test_non_square_and_shape_guards() -> None:
    assert not is_hermitian(np.zeros((2, 3), dtype=complex))
    assert not is_hermitian(np.zeros((2, 2, 2), dtype=complex))


def test_hermiticity_defect_is_homogeneous() -> None:
    """defect and scale are both degree-one homogeneous, so the ratio is invariant."""
    h = _broken(1.0, 1e-7)
    d0, s0 = hermiticity_defect(h)
    for c in (1e-8, 1e-3, 1e5, 1e11):
        dc, sc = hermiticity_defect(c * h)
        assert dc == pytest.approx(c * d0, rel=1e-12)
        assert sc == pytest.approx(c * s0, rel=1e-12)
