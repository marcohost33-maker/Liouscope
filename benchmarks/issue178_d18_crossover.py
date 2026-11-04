"""D18 block propagation: dense ``expm`` vs exponential action (issue #178).

D18 evolves a block of ``n_rhs`` Haar states to one time ``t``. The dense path
forms one ``expm(L t)`` and multiplies; the action path runs
``expm_multiply`` on the block. ``select_block_backend`` decides between them
by comparing the two cost estimates of :mod:`liouscope.numerics.propagation`,
with the action charged ``block_action_efficiency(n_rhs)`` times its counted
matrix-vector products. This script measures what that factor has to be and
how close the rule comes to choosing the faster backend.

For every case it prints both timings, the action's product bound, the dense
estimate, the observed ratio ``kappa = (t_action / bound) / (t_dense / est)``
and the backend ``"auto"`` chooses, then summarises agreement with the faster
backend and the time left on the table against an oracle. ``--holdout`` runs
the independent validation set (random GKSL generators, other ``t`` and block
widths) instead of the calibration set.

Run with single-threaded BLAS, the convention of issue #178::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/issue178_d18_crossover.py

Timings are descriptive for this machine. Measured 2026-10-27 (Linux,
OpenBLAS, single thread): calibration set 65/72 agreement, 1.2 % excess time,
worst case 1.58x; hold-out 29/32, 3.6 %, 1.38x. With several BLAS threads the
dense path usually gains more than the action's thin block products, which
would favour dense; on an oversubscribed container it can instead lose by
orders of magnitude, so multi-threaded numbers are not a calibration source.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from liouscope.core.lindblad import build_liouvillian
from liouscope.numerics import propagation as prop

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


def ladder(d: int, gamma: float = 0.5) -> np.ndarray:
    """Driven, damped and dephased ``d``-level ladder."""
    a = np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)
    H = a.conj().T @ a + 0.3 * (a + a.conj().T)
    return build_liouvillian(H, [math.sqrt(gamma) * a, math.sqrt(0.05) * a.conj().T @ a])


def _median_time(L: np.ndarray, B: np.ndarray, t: float, backend: str, reps: int) -> float:
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        prop.propagate_block_at_time(L, B, t, backend=backend)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _cases(holdout: bool):
    if not holdout:
        families = [
            ("tfim3", tfim(3)), ("ladder12", ladder(12)), ("tfim4", tfim(4)),
            ("ladder20", ladder(20)), ("ladder24", ladder(24)), ("tfim5", tfim(5)),
        ]
        for name, L0 in families:
            for scale in (0.3, 1.0, 3.0, 10.0):
                for n_rhs in (1, 10, 30):
                    yield f"{name} x{scale}", L0 * scale, 1.0, n_rhs
        return
    rng = np.random.default_rng(42)
    for d in (9, 14, 18, 22):
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        jumps = [
            (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / d
            for _ in range(2)
        ]
        L0 = build_liouvillian((H + H.conj().T) / np.sqrt(d), jumps)
        for scale in (0.5, 2.0):
            for t in (0.5, 2.0):
                for n_rhs in (5, 20):
                    yield f"gksl{d} x{scale}", L0 * scale, t, n_rhs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout", action="store_true", help="run the validation set")
    parser.add_argument("--reps", type=int, default=3, help="timing repetitions (median)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    agree = cases = 0
    oracle = excess = 0.0
    worst = 1.0
    print(f"{'case':16s} {'n':>5s} {'t':>4s} {'rhs':>3s} {'bound':>8s} {'est':>7s} "
          f"{'dense':>9s} {'action':>9s} {'kappa':>6s}  auto")
    for name, L, t, n_rhs in _cases(args.holdout):
        n = L.shape[0]
        bound = prop.block_action_matvec_bound(L, t, n_rhs)
        if bound > 2e6:
            continue
        estimate = prop.dense_block_matvec_estimate(L, t, n_rhs)
        B = rng.standard_normal((n, n_rhs)) + 1j * rng.standard_normal((n, n_rhs))
        t_dense = _median_time(L, B, t, prop.BACKEND_DENSE, args.reps)
        t_action = _median_time(L, B, t, prop.BACKEND_ACTION, args.reps)
        choice, _ = prop.select_block_backend(L, t, n_rhs)
        chosen = t_action if choice == prop.BACKEND_ACTION else t_dense
        best = min(t_action, t_dense)
        kappa = (t_action / bound) / (t_dense / estimate)
        cases += 1
        agree += (choice == prop.BACKEND_ACTION) == (t_action < t_dense)
        oracle += best
        excess += chosen - best
        worst = max(worst, chosen / best)
        print(f"{name:16s} {n:5d} {t:4.1f} {n_rhs:3d} {bound:8.0f} {estimate:7.0f} "
              f"{t_dense * 1e3:7.1f}ms {t_action * 1e3:7.1f}ms {kappa:6.2f}  {choice}")
    print(f"\nauto vs faster backend: {agree}/{cases} agree, "
          f"{excess / oracle * 100:.1f} % excess time, worst case {worst:.2f}x")


if __name__ == "__main__":
    main()
