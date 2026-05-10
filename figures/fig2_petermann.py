"""Figure 2: Petermann-factor divergence near the Jaynes-Cummings EP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import liouscope as ls
from liouscope.examples import v5_jaynes_cummings


def generate(out_path: Path) -> None:
    kappa = 1.0
    gs = np.linspace(0.02, 0.249, 12)
    Ks = []
    deltas = []
    for g in gs:
        sys = v5_jaynes_cummings(g=float(g), kappa=kappa, n_max=2)
        report = ls.diagnose(
            sys.L, rho_initial=sys.rho_initial, bootstrap_B=10,
            include_mpemba=False, seed=42,
        )
        Ks.append(float(report.nonnorm.petermann_max))
        deltas.append(float(report.spectral.gap))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; data table for fig 2:")
        for g, K, d in zip(gs, Ks, deltas):
            print(f"  g={g:.3f}  K_max={K:.3e}  Delta={d:.3e}")
        return

    fig, (ax_K, ax_d) = plt.subplots(1, 2, figsize=(10, 4))
    ax_K.semilogy(gs, Ks, "o-")
    ax_K.axvline(kappa / 4, ls="--", color="r", label="EP at kappa/4")
    ax_K.set_xlabel("coupling g")
    ax_K.set_ylabel("Petermann max K (D9)")
    ax_K.set_title("Petermann divergence")
    ax_K.legend()

    ax_d.plot(gs, deltas, "o-")
    ax_d.axvline(kappa / 4, ls="--", color="r")
    ax_d.set_xlabel("coupling g")
    ax_d.set_ylabel("Liouvillian gap Delta (D1)")
    ax_d.set_title("Gap collapse at EP")

    fig.suptitle("Figure 2: JC exceptional-point sweep")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
