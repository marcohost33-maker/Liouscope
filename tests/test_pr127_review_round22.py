"""PR #127 round-22 regression: extreme resolution and propagation limits.

The original review finding mixed two numerical questions:
1) rate * blind_interval can overflow while the sampling-ratio representation
   is correctly 0.0, so formatting its disclosure must use the mathematical
   100% decay limit rather than divide by zero;
2) at the same extreme scale the dense matrix exponential can itself become
   numerically unrepresentable even when L*t is elementwise finite.

These tests separate the two contracts so each failure mode is measured rather
than hidden behind whichever one happens first.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.diagnostics.relaxation import (
    UnrepresentableTrajectoryError,
    _decay_fraction_from_resolution,
    _evolve,
    _resolution_detail,
)

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_SP = _SM.conj().T
_RHO_PLUS = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)


def _extreme_thermalising_qubit() -> tuple[np.ndarray, float]:
    rate = 8.5e307
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.sqrt(rate) * _SM, np.sqrt(rate) * _SP],
    )
    assert np.all(np.isfinite(L))
    return L, rate


def test_zero_resolution_ratio_has_the_complete_decay_limit() -> None:
    """The 0.0 sampling-ratio representation maps to 100% decay, not 1/0."""
    L, rate = _extreme_thermalising_qubit()
    grid = np.array([0.0, 1.1])

    # Exact spectrum of equal raising/lowering on a qubit.  Supplying it here
    # isolates the resolution arithmetic from a dense eigensolve at 1e308 scale.
    eigenvalues = np.array([0.0, -rate, -rate, -2.0 * rate], dtype=float)
    resolution, worst_rate, blind, blind_start = _resolution_detail(
        L, grid, eigenvalues=eigenvalues
    )

    assert np.all(np.isfinite(L * grid[-1]))
    assert np.isfinite(worst_rate)
    assert blind == pytest.approx(1.1)
    assert blind_start == pytest.approx(0.0)
    assert resolution == 0.0
    assert _decay_fraction_from_resolution(resolution) == 1.0


@pytest.mark.parametrize("samples", [0.25, 1.0, 4.0, 1.0e6])
def test_decay_fraction_matches_the_direct_formula_away_from_zero(samples: float) -> None:
    expected = 1.0 - float(np.exp(-1.0 / samples))
    assert _decay_fraction_from_resolution(samples) == pytest.approx(
        expected, rel=2.0e-15, abs=0.0
    )


def test_extreme_finite_Lt_fails_closed_when_expm_is_nonfinite() -> None:
    """Finite L*t is necessary but not sufficient for a usable trajectory."""
    L, _ = _extreme_thermalising_qubit()
    grid = np.array([0.0, 1.1])
    assert np.all(np.isfinite(L * grid[-1]))

    with pytest.raises(UnrepresentableTrajectoryError, match="scipy.linalg.expm"):
        _evolve(L, _RHO_PLUS, grid)
