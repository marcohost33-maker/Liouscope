"""V3 amplitude-damped qubit reference (three gamma regimes)."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import diagnose
from liouscope.examples import v3_amplitude_damped_qubit


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1.5])
def test_v3_runs_at_multiple_rates(gamma):
    sys = v3_amplitude_damped_qubit(gamma=gamma)
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    assert report.spectral.gap > 0
    assert np.isfinite(report.relaxation.beta_D)


def test_v3_aicc_picks_model():
    sys = v3_amplitude_damped_qubit(gamma=0.5)
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=30, seed=42)
    assert report.relaxation.aicc_model in ("M0", "M1", "M2", "M3a", "M3b")
