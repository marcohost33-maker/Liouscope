"""Extended benchmark covering the four lattices and three dissipator families."""

from __future__ import annotations

import time

import liouscope as ls
from liouscope.core import (
    boundary_dephasing_jumps,
    bulk_amplitude_damping_jumps,
    bulk_dephasing_jumps,
    honeycomb_lattice,
    ising_hamiltonian,
    one_d_chain,
    square_lattice,
    triangular_lattice,
)


def main() -> None:
    geometries = {
        "1D-chain-4":  one_d_chain(4),
        "square-2x2":  square_lattice(2, 2),
        "honeycomb-2": honeycomb_lattice(2),
        "tri-2x2":     triangular_lattice(2, 2),
    }
    dissipators = {
        "bulk_dephasing":  bulk_dephasing_jumps,
        "boundary":        boundary_dephasing_jumps,
        "bulk_amplitude":  bulk_amplitude_damping_jumps,
    }
    print(f"Geometry           Dissipator        Delta     model  beta_D    a_class")
    for geom_name, lat in geometries.items():
        H = ising_hamiltonian(lat, J=1.0, h=0.5)
        N = lat.n_sites
        for diss_name, diss_fn in dissipators.items():
            jumps = diss_fn(N)
            rates = [0.2] * len(jumps)
            L = ls.build_liouvillian(H, jumps, rates)
            t0 = time.perf_counter()
            report = ls.diagnose(
                L, bootstrap_B=20, include_mpemba=False, seed=42,
            )
            dt = time.perf_counter() - t0
            print(
                f"{geom_name:<18} {diss_name:<17} "
                f"{report.spectral.gap:.4f}    "
                f"{report.relaxation.aicc_model:>3}   "
                f"{report.relaxation.beta_D: .3f}   "
                f"{report.classification.a_class}  "
                f"({dt:.1f}s)"
            )


if __name__ == "__main__":
    main()
