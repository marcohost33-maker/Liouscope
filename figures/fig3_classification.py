"""Figure 3: A1-A12 verdict distribution over V1-V5."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import liouscope as ls
from liouscope.examples import all_systems


def generate(out_path: Path) -> None:
    counts: Counter[str] = Counter()
    confidences: dict[str, float] = {}
    for sys in all_systems():
        report = ls.diagnose(
            sys.L, rho_initial=sys.rho_initial,
            bootstrap_B=20, include_mpemba=True, seed=42,
        )
        counts[report.classification.a_class] += 1
        confidences[sys.name] = report.classification.confidence

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; data table for fig 3:")
        for cls, cnt in counts.most_common():
            print(f"  {cls}: {cnt}")
        print("Confidences per system:")
        for k, v in confidences.items():
            print(f"  {k}: {v:.3f}")
        return

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(10, 4))
    if counts:
        classes, freqs = zip(*counts.most_common())
        ax_h.bar(classes, freqs)
    ax_h.set_xlabel("A-class")
    ax_h.set_ylabel("count")
    ax_h.set_title("Mechanism distribution over V1-V5")

    sys_names = list(confidences)
    sys_confs = [confidences[k] for k in sys_names]
    ax_c.bar(range(len(sys_names)), sys_confs)
    ax_c.set_xticks(range(len(sys_names)))
    ax_c.set_xticklabels(sys_names, rotation=15)
    ax_c.set_ylabel("confidence")
    ax_c.set_ylim(0, 1)
    ax_c.set_title("Per-system classification confidence")

    fig.suptitle("Figure 3: Mechanism classification")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
