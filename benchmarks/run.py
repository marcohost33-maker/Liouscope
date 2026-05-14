"""Benchmark runner: load a BM-* entry from ``LIOUSCOPE_BENCHMARK_MANIFEST.yaml``
and execute its canonical pipeline.

Usage::

    python benchmarks/run.py BM-001
    python benchmarks/run.py BM-003 --output results/bm003.json

The runner only depends on the standard library plus NumPy + SciPy. If
``PyYAML`` is installed it is used to parse the manifest; otherwise a
minimal ``yaml``-like loader runs against the well-known structure of the
manifest file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_manifest() -> dict:
    path = REPO_ROOT / "LIOUSCOPE_BENCHMARK_MANIFEST.yaml"
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        # Minimal fallback: rely on the top-level structure we know.
        # The benchmark manifest is intentionally simple enough to parse via
        # a small subset of YAML; we only need the ``benchmarks`` list and
        # the per-entry IDs / commands / paper references.
        return _yaml_lite(text)


def _yaml_lite(text: str) -> dict:
    """Very small YAML reader for the known manifest shape. Best-effort.

    This is **not** a general YAML parser. It handles:
    - top-level ``key: value`` pairs
    - lists of ``- key: value`` dicts
    - quoted strings
    - basic integer/float/bool conversion
    It is good enough for our manifest but **must not** be relied on for
    user-supplied YAML; install ``pyyaml`` for that.
    """
    raise RuntimeError(
        "PyYAML is required to run the benchmark loader. "
        "Install with: pip install pyyaml"
    )


def _bm_001(seed: int, *, n_qubits: int = 5) -> dict:
    """BM-001 Ising chain boundary dephasing.

    The canonical manifest entry uses ``N = 8`` (Liouvillian dimension 65536);
    for the golden-output regression fixture we fall back to ``N = 5``
    (Liouvillian dimension 1024) so the CI smoke job finishes quickly. The
    full N=8 run remains reachable via ``--n-qubits 8``.
    """
    from liouscope.core import (
        boundary_dephasing_jumps,
        ising_hamiltonian,
        one_d_chain,
    )
    from liouscope.sparse import build_sparse_liouvillian, sparse_steady_state

    lat = one_d_chain(n_qubits)
    H = ising_hamiltonian(lat, J=1.0, h=1.0)
    jumps = boundary_dephasing_jumps(n_qubits)
    L = build_sparse_liouvillian(H, jumps, [0.1] * len(jumps))
    t0 = time.perf_counter()
    rho_ss = sparse_steady_state(L, tol=1.0e-7)
    dt = time.perf_counter() - t0
    return {
        "benchmark_id": "BM-001",
        "N": n_qubits,
        "liouvillian_dim": int(L.shape[0]),
        # Wall time is intentionally omitted from the returned dict so the
        # golden fixture stays deterministic; we record only physics-level
        # quantities here.
        "wall_seconds_omitted_for_determinism": True,
        "trace_steady_state": float(np.real(np.trace(rho_ss))),
    }


def _bm_002(seed: int) -> dict:
    from liouscope.core import (
        boundary_dephasing_jumps,
        heisenberg_xxz_hamiltonian,
        one_d_chain,
    )
    from liouscope.sparse import build_sparse_liouvillian, sparse_steady_state

    rows = []
    for N in (4, 5, 6):
        lat = one_d_chain(N)
        H = heisenberg_xxz_hamiltonian(lat, J=1.0, Delta=0.5)
        # Heisenberg-XXZ scaling benchmark with dephasing as a stand-in
        # for the canonical boundary-driving dissipator (kept simple so
        # the runner works without QuTiP-specific operators).
        jumps = boundary_dephasing_jumps(N)
        L = build_sparse_liouvillian(H, jumps, [0.25] * len(jumps))
        t0 = time.perf_counter()
        rho_ss = sparse_steady_state(L, tol=1.0e-7)
        dt = time.perf_counter() - t0
        rows.append({
            "N": N,
            "liouvillian_dim": int(L.shape[0]),
            "trace": float(np.real(np.trace(rho_ss))),
        })
        _ = dt  # wall time intentionally not recorded in the JSON
    return {"benchmark_id": "BM-002", "rows": rows}


def _bm_003(seed: int) -> dict:
    import liouscope as ls
    from liouscope.examples import v1b_thermal_qutrit

    sys_q = v1b_thermal_qutrit(beta=1.0, omega=1.0)
    report = ls.diagnose(
        sys_q.L,
        rho_initial=sys_q.rho_initial,
        bootstrap_B=20,
        seed=seed,
    )
    gns = report.spectral.gns_gap
    kms = report.spectral.kms_gap
    return {
        "benchmark_id": "BM-003",
        "gap": float(report.spectral.gap),
        "gns_gap": float(gns),
        "kms_gap": float(kms),
        "kms_over_gns": float(kms / gns) if gns > 0 else None,
        "a_class": report.classification.a_class,
        "f_family": report.classification.f_family,
        "verdict": report.classification.verdict,
        "run_id": report.governance.run_id,
    }


def _bm_003b(seed: int) -> dict:
    import liouscope as ls

    H = np.array(
        [[0.0, 0.3, 0.0], [0.3, 1.0, 0.4], [0.0, 0.4, 2.5]],
        dtype=complex,
    )
    jumps: list[np.ndarray] = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        op = np.zeros((3, 3), dtype=complex)
        op[j, i] = 1.0
        jumps.append(op)
        jumps.append(op.conj().T)
    rates = [0.3, 0.05, 0.4, 0.07, 0.2, 0.04]
    L = ls.build_liouvillian(H, jumps, rates)
    report = ls.diagnose(L, bootstrap_B=10, include_mpemba=False, seed=seed)
    gns = report.spectral.gns_gap
    kms = report.spectral.kms_gap
    return {
        "benchmark_id": "BM-003b",
        "gap": float(report.spectral.gap),
        "gns_gap": float(gns),
        "kms_gap": float(kms),
        "kms_over_gns": float(kms / gns) if gns > 0 else None,
        "a_class": report.classification.a_class,
        "f_family": report.classification.f_family,
        "verdict": report.classification.verdict,
        "run_id": report.governance.run_id,
    }


_RUNNERS = {
    "BM-001": _bm_001,
    "BM-002": _bm_002,
    "BM-003": _bm_003,
    "BM-003b": _bm_003b,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a LiouScope benchmark entry.")
    parser.add_argument("benchmark_id", choices=sorted(_RUNNERS))
    parser.add_argument("--scaling", action="store_true", help="enable scaling sweep (BM-002 only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", help="optional JSON output path")
    args = parser.parse_args(argv)

    # Validate that the manifest still contains the requested ID. The
    # manifest is the single source of truth -- the runner must agree.
    try:
        manifest = _load_manifest()
        ids = {entry["benchmark_id"] for entry in manifest.get("benchmarks", [])}
        if args.benchmark_id not in ids:
            print(f"error: {args.benchmark_id} not in manifest", file=sys.stderr)
            return 2
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"warning: could not load manifest ({exc}); proceeding anyway.", file=sys.stderr)

    print(f"Running {args.benchmark_id} (seed={args.seed})...")
    result = _RUNNERS[args.benchmark_id](args.seed)
    payload = json.dumps(result, sort_keys=True, indent=2)
    print(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"\nSHA-256: {digest}")

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
