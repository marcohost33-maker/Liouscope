"""Regression for PR #127 review finding: zero resolution ratio warning.

The fixture keeps every generator entry finite, while the fastest decay rate
times the caller's blind interval exceeds float64. Before the fix,
_resolution_detail correctly returned 0.0 samples per e-folding and the
warning formatter then divided by zero.
"""

from __future__ import annotations

import warnings

import numpy as np

from liouscope import build_liouvillian
from liouscope.diagnostics.relaxation import (
    UnderResolvedTransientWarning,
    compute_relaxation_layer,
)

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_SP = _SM.conj().T
_RHO_PLUS = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)


def test_zero_resolution_ratio_warns_as_complete_decay_instead_of_crashing() -> None:
    """Overflow in rate*blind is the 100% decay limit, not an exception."""
    rate = 8.5e307
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.sqrt(rate) * _SM, np.sqrt(rate) * _SP],
    )
    grid = np.array([0.0, 1.1])

    # The generator itself remains representable. The failing quantity is the
    # dimensionless product of the fastest decay rate and the blind interval.
    assert np.all(np.isfinite(L))
    assert np.all(np.isfinite(L * grid[-1]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = compute_relaxation_layer(
            L,
            rho_initial=_RHO_PLUS,
            t_grid=grid,
            bootstrap_B=5,
            seed=1,
        )

    hits = [
        w for w in caught if issubclass(w.category, UnderResolvedTransientWarning)
    ]
    assert report.samples_per_fast_efolding == 0.0
    assert hits, "the zero-resolution limit must still emit the disclosure"
    assert "100.0%" in str(hits[0].message)
