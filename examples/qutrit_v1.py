"""V1 dissipative qutrit worked example."""

from __future__ import annotations

import liouscope as ls
from liouscope.examples import v1_qutrit


def main() -> None:
    sys = v1_qutrit()
    print(f"System: {sys.name} -- {sys.description}")
    print(f"  Liouvillian dim: {sys.L.shape}")
    report = ls.diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=100, seed=42)
    delta = report.spectral.gap
    delta_s = report.spectral.gns_gap
    kms = report.spectral.kms_gap
    ratio = kms / delta_s if delta_s > 0 else float("inf")
    print(f"  Delta            = {delta:.4f}")
    print(f"  Delta_s (GNS)    = {delta_s:.4f}")
    print(f"  KMS gap          = {kms:.4f}")
    print(f"  KMS / GNS ratio  = {ratio:.4f}")
    print(f"  Petermann max    = {report.nonnorm.petermann_max:.3e}")
    print(f"  Verdict          = {report.classification.verdict}")
    print(f"  A-class          = {report.classification.a_class}")


if __name__ == "__main__":
    main()
