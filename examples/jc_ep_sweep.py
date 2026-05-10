"""V5 Jaynes-Cummings exceptional-point Petermann-divergence sweep.

Approaches the EP at ``g = kappa / 4`` and tracks ``D9`` (Petermann max).
Expected behaviour: K diverges several orders of magnitude near the EP.
"""

from __future__ import annotations

import numpy as np

import liouscope as ls
from liouscope.examples import v5_jaynes_cummings


def main() -> None:
    kappa = 1.0
    g_values = np.linspace(0.05, kappa / 4.0 - 1.0e-3, 6)
    print("Jaynes-Cummings EP sweep (target EP at g = kappa/4 = 0.25)")
    print("  g       Delta     PetermannMax    Kreiss")
    for g in g_values:
        sys = v5_jaynes_cummings(g=g, kappa=kappa, n_max=2)
        # Lightweight diagnose (no bootstrap, no mpemba)
        report = ls.diagnose(
            sys.L,
            rho_initial=sys.rho_initial,
            bootstrap_B=20,
            include_mpemba=False,
            seed=42,
        )
        print(
            f"  {g:.3f}   {report.spectral.gap:.4e}   "
            f"{report.nonnorm.petermann_max:.3e}   "
            f"{report.nonnorm.kreiss:.3e}"
        )


if __name__ == "__main__":
    main()
