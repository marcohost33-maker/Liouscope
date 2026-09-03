"""Regression tests for observable-amplitude invariance in the GLS optimiser.

Issue #124: the mathematical fit is unchanged by ``y -> c*y`` for ``c > 0``,
but feeding the raw residuals to SciPy lets the gradient termination criterion
accept the initial seed when ``c`` is tiny.  These tests pin both directions:
the tiny-amplitude counterexample must fit, and the ordinary-scale positive
control must remain unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.fitting.gls import fit_gls_ar1

_TRUE_RATE = 1.3
_SEED_RATE = 0.2
_T = np.linspace(0.0, 5.0, 64)


def _fit_exponential(scale: float):
    def model(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        return scale * np.exp(-params[0] * t)

    y = scale * np.exp(-_TRUE_RATE * _T)
    return fit_gls_ar1(
        model,
        _T,
        y,
        np.array([_SEED_RATE]),
        bounds=(np.array([0.0]), np.array([5.0])),
        n_iters=1,
    )


def test_tiny_amplitude_cannot_turn_the_seed_into_a_measurement() -> None:
    """The exact #124 counterexample must move away from its initial seed."""
    tiny = _fit_exponential(1.0e-40)

    assert tiny.success
    assert not tiny.degenerate
    assert tiny.params[0] == pytest.approx(_TRUE_RATE, rel=1.0e-7, abs=1.0e-10)
    assert abs(tiny.params[0] - _SEED_RATE) > 0.5


def test_amplitude_rescaling_preserves_the_fitted_rate() -> None:
    """Positive control: ordinary and tiny amplitudes represent the same fit."""
    ordinary = _fit_exponential(1.0)
    tiny = _fit_exponential(1.0e-40)

    assert ordinary.success and tiny.success
    assert ordinary.params[0] == pytest.approx(_TRUE_RATE, rel=1.0e-7, abs=1.0e-10)
    assert tiny.params[0] == pytest.approx(ordinary.params[0], rel=1.0e-9, abs=1.0e-12)
