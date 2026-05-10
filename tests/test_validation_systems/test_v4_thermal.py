"""V4 thermal two-level reference: KMS = GNS at detailed balance."""

from __future__ import annotations

from liouscope import diagnose
from liouscope.examples import v4_thermal_two_level


def test_v4_kms_equals_gns_at_detailed_balance():
    sys = v4_thermal_two_level(beta=1.0, omega=1.0)
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    diff = abs(report.spectral.kms_gap - report.spectral.gns_gap)
    assert diff < 0.05, f"KMS - GNS = {diff}, expected ~0 at detailed balance"


def test_v4_at_higher_temperature():
    sys = v4_thermal_two_level(beta=0.2, omega=1.0)
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    assert report.spectral.gap > 0
