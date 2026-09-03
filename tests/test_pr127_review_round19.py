"""PR #127 round-19 review findings (external).

Three findings, all of the same family that PR #121 has been working through:
a quantity whose verdict changes when nothing but the UNITS change, or a
substitute value returned where a refusal was owed.

* **prony.py** the whole-grid uniformity test used NumPy's default
  ``atol=1e-8``, an ABSOLUTE time. On a nanosecond-scale two-scale grid it
  called steps that differ by four orders of magnitude uniform, so the
  two-scale prefix path never ran.
* **relaxation.py** an unrepresentable gap-scaled window fell back to the
  absolute legacy window and the fits ran on it anyway -- producing a full
  set of finite rates, a class label and, worst of all, the provenance tag
  ``gap_scaled`` for a grid that was nothing of the kind.
* **lindblad.py / sparse/build.py** a Hamiltonian that is numerically a pure
  gauge term has no traceless scale left after gauge fixing, so the relative
  Hermiticity gate compared round-off against round-off and rejected the
  exact Hamiltonian ``I``.

Each test is paired with a mutation in the round-19 discrimination spec.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.relaxation import (
    UnrepresentableRelaxationWindowError,
    default_relaxation_grid,
)
from liouscope.fitting.prony import _uniform_prefix_length
from liouscope.sparse.build import build_sparse_liouvillian

_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# Finding 1 -- the Prony uniformity gate must be scale-relative.
# ---------------------------------------------------------------------------


def test_nanosecond_two_scale_grid_is_not_called_uniform() -> None:
    """The reviewer's units: ``gap=1e8``, ``fast_rate=1e12``.

    Measured on this grid the steps span a factor of 1.03e4 -- as
    non-uniform as a grid gets -- and every one of them is below NumPy's
    default absolute ``atol=1e-8``, which is why the default declared them
    all equal.
    """
    grid = default_relaxation_grid(1.0e8, fast_rate=1.0e12)
    steps = np.diff(grid)

    # The fixture has to be a genuinely two-scale grid, or it proves nothing.
    assert steps.max() / steps.min() > 1.0e3
    assert steps.max() < 1.0e-8, "fixture no longer sits under the default atol"

    # This is the comparison the code performs.
    assert not np.allclose(steps, steps[0], rtol=1e-4, atol=0.0)
    # ... and this is what it used to perform. Kept as a live assertion so the
    # test states WHY atol=0.0 is load-bearing rather than merely asserting
    # the fixed behaviour.
    assert np.allclose(steps, steps[0], rtol=1e-4), (
        "NumPy's default atol no longer masks this grid; the finding's "
        "premise has changed and this test needs rereading"
    )

    # The consequence the finding is actually about: the prefix path runs.
    head = _uniform_prefix_length(grid)
    assert 6 <= head < grid.size


def test_a_genuinely_uniform_grid_is_still_uniform() -> None:
    """Negative control: ``atol=0.0`` must not reclassify honest grids.

    A linspace carries ulp-level variation in its diffs; ``rtol=1e-4`` covers
    that with eleven orders of magnitude to spare. If this ever turns red the
    tightening has become a regression rather than a repair.
    """
    for span in (5.0, 1.0e-9, 1.0e9):
        steps = np.diff(np.linspace(0.0, span, 80))
        assert np.allclose(steps, steps[0], rtol=1e-4, atol=0.0)


# ---------------------------------------------------------------------------
# Finding 2 -- an unrepresentable window is withheld, not substituted.
# ---------------------------------------------------------------------------


def test_unrepresentable_window_is_withheld() -> None:
    """``horizon/gap`` overflows: refuse rather than answer about [0, 10]."""
    with pytest.raises(UnrepresentableRelaxationWindowError):
        default_relaxation_grid(5.0e-308)


def test_no_usable_decay_scale_keeps_its_documented_fallback() -> None:
    """The other state, and it must stay distinguishable from the first.

    "Not determinable" (no gap was resolved) and "determined but not
    representable" are different answers to different questions. Collapsing
    them is what round 18 did; this control pins that round 19 did not
    collapse them the other way.
    """
    for gap in (0.0, -1.0, float("nan"), float("inf")):
        grid = default_relaxation_grid(gap)
        assert grid.size == 80
        assert np.all(np.isfinite(grid))
        assert grid.max() == pytest.approx(10.0)


def test_a_representable_window_is_untouched() -> None:
    """Positive control: ordinary rate units keep the gap-scaled window."""
    grid = default_relaxation_grid(0.5)
    assert np.all(np.isfinite(grid))
    assert grid.max() == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Finding 3 -- round-off allowance near a pure-gauge Hamiltonian.
# ---------------------------------------------------------------------------


def _numerically_pure_gauge_hamiltonian() -> np.ndarray:
    """``Q @ I @ Q^H`` for a numerical unitary ``Q``: exactly ``I``, in theory.

    In floating point it is ``I`` plus round-off, and the round-off is all
    that survives gauge fixing.
    """
    rng = np.random.default_rng(0)
    q = sla.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    return q @ np.eye(2, dtype=complex) @ q.conj().T


def test_pure_gauge_hamiltonian_is_accepted_by_both_builders() -> None:
    """THE regression, and it must hold for the sparse twin as well.

    The finding names the mirrored sparse calculation explicitly. Two
    builders that disagree about whether the same Hamiltonian is valid is a
    worse defect than either verdict on its own.
    """
    h = _numerically_pure_gauge_hamiltonian()

    # The fixture must actually sit in the regime under repair.
    defect = float(np.max(np.abs(h - h.conj().T)))
    gauge = h - (np.trace(h).real / 2.0) * np.eye(2, dtype=complex)
    gauge_scale = float(np.max(np.abs(gauge)))
    assert defect > 0.0
    assert defect > 1.0e-9 * gauge_scale, (
        "fixture no longer trips the relative gate; the finding's premise "
        "has changed"
    )

    assert build_liouvillian(h, [_SIGMA_MINUS]).shape == (4, 4)
    assert build_sparse_liouvillian(h, [_SIGMA_MINUS]).shape == (4, 4)


def test_a_real_hermiticity_defect_is_still_rejected() -> None:
    """Negative control: the allowance must not become a blanket pass."""
    h_bad = np.array([[1.0, 1.0e-3], [0.0, -1.0]], dtype=complex)
    with pytest.raises(ValueError, match="Hermitian"):
        build_liouvillian(h_bad, [_SIGMA_MINUS])
    with pytest.raises(ValueError, match="Hermitian"):
        build_sparse_liouvillian(h_bad, [_SIGMA_MINUS])


def test_the_twelfth_round_gauge_hole_stays_closed() -> None:
    """The allowance must not undo the reason gauge fixing exists.

    Twelfth-round review: ``H + 1e9*I`` passed with the same non-Hermitian
    part that ``H`` alone rejected. Measured here, the new allowance is
    4.44e-7 and the defect is 1e-6, so the rejection survives -- but the
    margin is a factor of two, which is exactly why this control is pinned
    rather than argued.
    """
    h_hole = np.array([[1.0, 1.0e-6], [0.0, -1.0]], dtype=complex) + 1.0e9 * np.eye(2)
    with pytest.raises(ValueError, match="Hermitian"):
        build_liouvillian(h_hole, [_SIGMA_MINUS])
    with pytest.raises(ValueError, match="Hermitian"):
        build_sparse_liouvillian(h_hole, [_SIGMA_MINUS])


def test_a_traceless_hamiltonian_keeps_a_bit_identical_verdict() -> None:
    """For traceless ``H`` the allowance is exactly zero, by construction.

    This is the statement that bounds the blast radius of the change: every
    Hamiltonian without an identity component is gated exactly as before.
    """
    h = np.array([[1.0, 1.0e-8], [0.0, -1.0]], dtype=complex)
    gauge_shift = float(np.trace(h).real) / 2.0
    assert gauge_shift == 0.0
    assert 2 * float(np.finfo(float).eps) * abs(gauge_shift) == 0.0
    with pytest.raises(ValueError, match="Hermitian"):
        build_liouvillian(h, [_SIGMA_MINUS])
