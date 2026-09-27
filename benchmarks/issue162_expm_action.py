"""Issue #162 benchmark: full dense expm vs exponential action.

Run manually:
    python benchmarks/issue162_expm_action.py

The script is evidence generation, not a performance gate. It prints elapsed
time, peak RSS (where ``resource`` is available), and dense/action agreement.
Each backend runs in a fresh child process so peak RSS is not contaminated by
the backend that ran first.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla


def _matrix(n: int, seed: int = 162) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    # Shift left so exp(A) is not dominated by explosive growth.
    A -= (float(n) + 1.0) * np.eye(n)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return np.asarray(A, dtype=complex), np.asarray(b, dtype=complex)


def _peak_rss_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. This repo CI/benchmark target
        # is Linux, but keep the output approximately useful on macOS.
        return rss / (1024.0 if rss < 1.0e8 else 1024.0**2)
    except ImportError:
        return float("nan")


def _worker(kind: str, n: int, q: mp.Queue) -> None:
    A, b = _matrix(n)
    started = time.perf_counter()
    if kind == "dense":
        out = sla.expm(A) @ b
    elif kind == "action":
        out = spla.expm_multiply(A, b)
    else:
        raise ValueError(kind)
    q.put((kind, time.perf_counter() - started, _peak_rss_mb(), out))


def _run(kind: str, n: int) -> tuple[float, float, np.ndarray]:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(kind, n, q))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"{kind} child failed with exit code {proc.exitcode}")
    _kind, elapsed, rss, out = q.get()
    return float(elapsed), float(rss), np.asarray(out)


def main() -> None:
    for n in (64, 128, 256):
        dense_t, dense_rss, dense = _run("dense", n)
        action_t, action_rss, action = _run("action", n)
        rel = float(
            np.linalg.norm(action - dense)
            / max(np.linalg.norm(dense), np.finfo(float).tiny)
        )
        print(
            f"n={n:4d}  dense={dense_t:9.4f}s/{dense_rss:9.1f}MB  "
            f"action={action_t:9.4f}s/{action_rss:9.1f}MB  relerr={rel:.3e}"
        )


if __name__ == "__main__":
    main()
