"""V2 pure-dephasing qubit reference."""

from __future__ import annotations

import numpy as np

from liouscope import diagnose
from liouscope.examples import v2_dephasing_qubit


def test_v2_runs():
    sys = v2_dephasing_qubit()
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    assert report.spectral.gap > 0
    assert np.isfinite(report.relaxation.beta_D)


def test_v2_gap_matches_dephasing_rate():
    sys = v2_dephasing_qubit()
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    # gamma = 0.25 from v2_dephasing_qubit definition
    # Gap of Pauli oscillation is ~ gamma = 0.25 (from complex pair real part)
    assert 0.1 < report.spectral.gap < 0.6
