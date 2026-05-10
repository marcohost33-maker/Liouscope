"""Heisenberg-XXZ alpha-scaling benchmark.

Fits ``beta_D ~ N^{-alpha}`` for N = 2..5 (small to keep the dense pipeline
fast). The full paper benchmark uses N = 3..7 with sparse path; this script
showcases the orchestration.
"""

from __future__ import annotations

import time

import numpy as np

import liouscope as ls
from liouscope.core import (
    boundary_dephasing_jumps,
    heisenberg_xxz_hamiltonian,
    one_d_chain,
)


def main() -> None:
    Ns = list(range(2, 6))
    betas: list[float] = []
    print("Heisenberg-XXZ scaling benchmark")
    print("  N   dim   beta_D    wall(s)")
    for N in Ns:
        lat = one_d_chain(N)
        H = heisenberg_xxz_hamiltonian(lat, J=1.0, Delta=1.0)
        jumps = boundary_dephasing_jumps(N)
        rates = [0.25] * len(jumps)
        L = ls.build_liouvillian(H, jumps, rates)
        t0 = time.perf_counter()
        report = ls.diagnose(L, bootstrap_B=20, include_mpemba=False, seed=42)
        dt = time.perf_counter() - t0
        beta = float(report.relaxation.beta_D)
        betas.append(beta)
        print(f"  {N}   {L.shape[0]:>3}   {beta: .4f}   {dt:.2f}")
    if all(b > 0 for b in betas):
        log_betas = np.log(np.asarray(betas))
        log_Ns = np.log(np.asarray(Ns, dtype=float))
        slope, intercept = np.polyfit(log_Ns, log_betas, 1)
        print(f"\nLog-log slope alpha = {-slope:.3f}")


if __name__ == "__main__":
    main()
