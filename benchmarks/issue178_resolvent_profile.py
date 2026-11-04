"""Resolvent / non-normality sweeps at d = 8 and d = 16 (issue #178, item 2).

The two layers evaluate ``sigma_min(zI - L)``, or its reciprocal, on grids of
one operator: D10 (600 points), D10b (1200), D11b (2 x 201) and D13
(2 x 625). For ``n = d**2 > 128`` every sweep now shares one complex Schur
decomposition and runs inverse Lanczos on the shifted triangle
(``liouscope.numerics.resolvent.ShiftedSchurResolvent``).

For each damped transverse-field Ising chain this script times every sweep and
then repeats it with the dense formulas forced (the explicit inverse + SVD,
or the SVD, per point) to print the largest relative deviation, plus the
number of Schur evaluations and dense fallbacks.

Run with single-threaded BLAS, the convention of issue #178::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/issue178_resolvent_profile.py

Measured 2026-10-28 (Linux, OpenBLAS 0.3.31, SciPy 1.17.1, single thread),
``diagnose(bootstrap_B=20)`` on the d = 16 chain: 81.5 s before, 19.2 s after;
the five sweeps timed here 7.2 s against 55.8 s with the dense formulas.
Before, the sweeps were ~90 % of the call: 1602 ``resolvent_norm`` calls
(SuperLU of the dense matrix, then on average 106 power-iteration steps, 272
of them stopped at the 200-step cap with a ResolventConvergenceWarning), 600
explicit inverses and 1250 SVDs. After, the values agree with the dense
formulas to <= 1.8e-13 relative, the D13 decisions are identical, and nothing
below n = 128 changes at all.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import ClassVar

import numpy as np

import liouscope.numerics.resolvent as resolvent
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.nonnormality import kreiss_constant, kreiss_grid_lower_bound
from liouscope.diagnostics.resolvent import (
    pseudospectral_radius_diag,
    resolvent_peak_curve,
    resolvent_peak_curve_scaled,
)

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _site(op: np.ndarray, k: int, n_sites: int) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for j in range(n_sites):
        out = np.kron(out, op if j == k else np.eye(2))
    return out


def tfim(n_sites: int, g: float = 1.0, coupling: float = 1.0, gamma: float = 0.1) -> np.ndarray:
    """Damped transverse-field Ising chain, ``d = 2**n_sites``."""
    H = sum(
        -coupling * _site(_SZ, k, n_sites) @ _site(_SZ, k + 1, n_sites)
        for k in range(n_sites - 1)
    ) + sum(-g * _site(_SX, k, n_sites) for k in range(n_sites))
    jumps = [math.sqrt(gamma) * _site(_SM, k, n_sites) for k in range(n_sites)]
    return build_liouvillian(H, jumps)


def _sweeps(L: np.ndarray) -> dict[str, object]:
    return {
        "D10 kreiss": lambda: kreiss_constant(L),
        "D10b kreiss_scaled": lambda: kreiss_grid_lower_bound(L).value,
        "D11b peak curve": lambda: resolvent_peak_curve(L, 1e-3)[1],
        "D11b peak curve (scaled)": lambda: resolvent_peak_curve_scaled(L, 1e-3)[1],
        "D13 radius": lambda: pseudospectral_radius_diag(L, 1e-3),
    }


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return np.asarray(value, dtype=float), time.perf_counter() - start


class _Counting(resolvent.ShiftedSchurResolvent):
    built: ClassVar[list[resolvent.ShiftedSchurResolvent]] = []

    def __init__(self, L: np.ndarray) -> None:
        super().__init__(L)
        _Counting.built.append(self)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--no-dense", action="store_true", help="skip the dense comparison")
    args = parser.parse_args()

    resolvent.ShiftedSchurResolvent = _Counting  # type: ignore[misc]
    cutoff = resolvent._DENSE_NORM_CUTOFF
    for n_sites in args.sites:
        L = tfim(n_sites)
        n = L.shape[0]
        path = "dense (n <= 128, unchanged)" if n <= cutoff else "shared Schur form"
        print(f"\nTFIM N={n_sites}: d={2 ** n_sites}, n={n}, path: {path}")
        print(f"{'sweep':26s} {'time':>8s} {'dense':>8s} {'max rel dev':>12s} {'evals':>6s} {'fallb':>5s}")
        for name, fn in _sweeps(L).items():
            _Counting.built.clear()
            value, t_new = _timed(fn)
            evals = sum(s.evaluations for s in _Counting.built)
            fallbacks = sum(s.dense_fallbacks for s in _Counting.built)
            if args.no_dense or n <= cutoff:
                print(f"{name:26s} {t_new:7.2f}s {'-':>8s} {'-':>12s} {evals:6d} {fallbacks:5d}")
                continue
            resolvent._DENSE_NORM_CUTOFF = 10**9
            try:
                ref, t_dense = _timed(fn)
            finally:
                resolvent._DENSE_NORM_CUTOFF = cutoff
            dev = float(np.max(np.abs(value - ref) / np.maximum(np.abs(ref), 1e-300)))
            print(f"{name:26s} {t_new:7.2f}s {t_dense:7.2f}s {dev:12.1e} {evals:6d} {fallbacks:5d}")


if __name__ == "__main__":
    main()
