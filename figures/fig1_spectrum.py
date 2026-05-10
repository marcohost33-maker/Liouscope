"""Figure 1: Spectral overview of V1-V5 (D1-D4)."""

from __future__ import annotations

from pathlib import Path

import liouscope as ls
from liouscope.examples import all_systems


def generate(out_path: Path) -> None:
    rows = []
    for sys in all_systems():
        report = ls.diagnose(
            sys.L, rho_initial=sys.rho_initial,
            bootstrap_B=20, include_mpemba=False, seed=42,
        )
        rows.append((
            sys.name,
            report.spectral.gap,
            report.spectral.gns_gap,
            report.spectral.kms_gap,
        ))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; data table for fig 1:")
        for r in rows:
            print(f"  {r[0]:<20}  Delta={r[1]:.4f}  GNS={r[2]:.4f}  KMS={r[3]:.4f}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(rows))
    ax.bar([i - 0.25 for i in x], [r[1] for r in rows], width=0.25, label="Delta (D1)")
    ax.bar(list(x),               [r[2] for r in rows], width=0.25, label="Delta_s (D2)")
    ax.bar([i + 0.25 for i in x], [r[3] for r in rows], width=0.25, label="KMS (D2b)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("Spectral gap")
    ax.set_title("Figure 1: Spectral overview of V1-V5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
