"""Smoke tests for diagnostics/transient.py."""
from __future__ import annotations

import numpy as np
import pytest

from liouscope.diagnostics.transient import (
    compute_transient_layer,
    kappa_trans,
    numerical_abscissa,
    trans_amplitude_ratio,
)


def _toy_lindblad_super(gamma: float = 0.5) -> np.ndarray:
    """Minimal 4x4 Liouvillian super-operator for a dephased qubit.

    Built directly in vec-form so the tests stay independent of the rest
    of the package. The dynamics decays the off-diagonal coherences.
    """
    L = np.zeros((4, 4), dtype=complex)
    L[1, 1] = -gamma
    L[2, 2] = -gamma
    return L


def test_numerical_abscissa_self_adjoint_part():
    L = _toy_lindblad_super(gamma=0.3)
    omega = numerical_abscissa(L)
    assert omega == pytest.approx(0.0, abs=1.0e-12), (
        "Hermitian part of a purely dissipative diagonal L is non-positive; max eig is 0."
    )


def test_trans_amplitude_ratio_decays():
    L = _toy_lindblad_super(gamma=0.5)
    ratio = trans_amplitude_ratio(L, t_grid=np.linspace(0.01, 2.0, 10))
    assert 0.0 < ratio <= 1.0 + 1.0e-9, "Propagator norm of a dissipative L is bounded by 1."


def test_kappa_trans_basic():
    assert kappa_trans(0.5, 0.1) == pytest.approx(5.0)
    assert kappa_trans(0.0, 0.5) == pytest.approx(0.0)
    assert np.isinf(kappa_trans(1.0, 0.0)), "Zero gap -> infinite ratio."
    assert np.isinf(kappa_trans(1.0, -1.0)), "Non-positive gap -> infinite (defensive)."


def test_compute_transient_layer_fields_present():
    L = _toy_lindblad_super(gamma=0.4)
    res = compute_transient_layer(L, gap=0.4)
    assert hasattr(res, "trans_amplitude_ratio")
    assert hasattr(res, "kappa_trans")
    assert hasattr(res, "numerical_abscissa")
    assert np.isfinite(res.numerical_abscissa)
    assert np.isfinite(res.trans_amplitude_ratio)
