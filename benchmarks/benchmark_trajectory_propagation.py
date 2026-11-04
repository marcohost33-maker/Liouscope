"""Dense ``expm`` vs exponential-action trajectory propagation (issue #162).

Times both backends of :mod:`liouscope.numerics.propagation` on a damped
transverse-field Ising chain (``N = 1..5`` sites, ``d = 2**N``,
superoperator dimension ``n = d**2``) over the default 80-point relaxation
grid, and reports what the ``"auto"`` rule chooses, the action's
matrix-vector-product bound, the peak extra memory each backend needs and the
dense-vs-action deviation.

Run with single-threaded BLAS for comparable numbers::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/benchmark_trajectory_propagation.py

Memory column: the dense backend materialises ``L*t`` and ``expm(L*t)``
(two ``n x n`` complex arrays, plus SciPy's Pade work arrays); the action
backend holds one scaled copy of ``L`` and a handful of length-``n`` vectors.
``tracemalloc`` measures the Python-visible peak of each.

Timings are descriptive for this machine only. The claim the library relies
on is structural -- dense cost grows like ``n**3`` per grid point, action cost
like ``nnz(L) * ||L - mu I||_1 * t_max`` -- and the rule in
``select_trajectory_backend`` uses those structures, never timings.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from liouscope.core.lindblad import build_liouvillian
from liouscope.numerics.kronecker import vec
from liouscope.numerics.propagation import (
    BACKEND_ACTION,
    BACKEND_DENSE,
    propagate_trajectory,
    select_trajectory_backend,
)

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _site(op: np.ndarray, i: int, n_sites: int) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for k in range(n_sites):
        out = np.kron(out, op if k == i else np.eye(2, dtype=complex))
    return out


def damped_ising_chain(
    n_sites: int, *, gamma: float = 0.7, J: float = 1.0, h: float = 0.3
) -> np.ndarray:
    """Dense GKSL generator of the benchmark fixture."""
    dim = 2**n_sites
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_sites - 1):
        H += J * _site(_SZ, i, n_sites) @ _site(_SZ, i + 1, n_sites)
    for i in range(n_sites):
        H += h * _site(_SX, i, n_sites)
    jumps = [np.sqrt(gamma) * _site(_SM, i, n_sites) for i in range(n_sites)]
    return build_liouvillian(H, jumps)


def _measure(L: np.ndarray, v0: np.ndarray, grid: np.ndarray, backend: str):
    tracemalloc.start()
    start = time.perf_counter()
    result = propagate_trajectory(L, v0, grid, backend=backend)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result.states, elapsed, peak


def run(max_sites: int, t_max: float, points: int) -> list[dict[str, float | int | str]]:
    grid = np.linspace(0.0, t_max, points)
    rows: list[dict[str, float | int | str]] = []
    for n_sites in range(1, max_sites + 1):
        L = damped_ising_chain(n_sites)
        dim = 2**n_sites
        rho0 = np.zeros((dim, dim), dtype=complex)
        rho0[-1, -1] = 1.0
        v0 = vec(rho0)
        auto, bound = select_trajectory_backend(L, grid)
        dense, t_dense, m_dense = _measure(L, v0, grid, BACKEND_DENSE)
        action, t_action, m_action = _measure(L, v0, grid, BACKEND_ACTION)
        rows.append(
            {
                "d": dim,
                "n": L.shape[0],
                "auto": auto,
                "matvec_bound": bound,
                "dense_s": t_dense,
                "action_s": t_action,
                "dense_peak_MiB": m_dense / 2**20,
                "action_peak_MiB": m_action / 2**20,
                "max_abs_dev": float(np.max(np.abs(dense - action))),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-sites", type=int, default=5)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=80)
    args = parser.parse_args()
    header = (
        f"{'d':>4} {'n':>6} {'auto':>12} {'bound':>9} {'dense s':>9} "
        f"{'action s':>9} {'dense MiB':>10} {'action MiB':>11} {'max |dev|':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in run(args.max_sites, args.t_max, args.points):
        print(
            f"{row['d']:>4} {row['n']:>6} {row['auto']:>12} {row['matvec_bound']:>9.0f} "
            f"{row['dense_s']:>9.3f} {row['action_s']:>9.3f} "
            f"{row['dense_peak_MiB']:>10.2f} {row['action_peak_MiB']:>11.2f} "
            f"{row['max_abs_dev']:>10.1e}"
        )


if __name__ == "__main__":
    main()
