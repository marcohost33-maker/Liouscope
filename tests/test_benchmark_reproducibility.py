"""Regression tests for the two benchmark fixes.

B2: ``benchmarks/reproduce_paper.py`` must hash only run-invariant fields, so
    wall-clock timing can never change the digest (was: every run differed).
B1: ``benchmarks/benchmark_heisenberg_scaling.py`` must handle the degenerate,
    U(1)-symmetric NESS of the boundary-dephasing Heisenberg model correctly
    (kernel dim == N+1; sector-projected steady state is exactly stationary)
    instead of crashing with ``DegenerateSteadyStateError``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _BENCH_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# B2 -- deterministic reproduce_paper hash (timing excluded)
# --------------------------------------------------------------------------- #

def test_reproduce_hash_excludes_timing():
    rp = _load("reproduce_paper")
    rows = [{"name": "sys", "dim": 4, "delta": 0.5, "beta_D": 1.25}]
    payload = rp.build_payload(rows, seed=42)
    # Timing is NOT part of the hashed payload.
    assert "wall_seconds" not in payload
    assert all("wall_seconds" not in r for r in payload["rows"])
    # Same deterministic inputs -> byte-identical digest.
    assert rp.digest_payload(payload) == rp.digest_payload(
        rp.build_payload(rows, seed=42)
    )


def test_reproduce_hash_invariant_to_timing_metadata():
    """Attaching perf metadata must not perturb the digest."""
    rp = _load("reproduce_paper")
    rows = [{"name": "sys", "dim": 4, "delta": 0.5, "beta_D": 1.25}]
    payload = rp.build_payload(rows, seed=42)
    base = rp.digest_payload(payload)
    # Simulate two runs with different wall times attached as artefact perf.
    for wall in (0.001, 999.0):
        artefact = {**payload, "sha256": base, "perf": {"wall_seconds_total": wall}}
        # Digest is computed over `payload`, never over `artefact`.
        assert rp.digest_payload(payload) == base
        assert artefact["sha256"] == base


# --------------------------------------------------------------------------- #
# B1 -- degenerate NESS handled, not crashed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N", [2, 3, 4])
def test_boundary_dephasing_kernel_is_N_plus_1(N):
    import liouscope as ls
    from liouscope.core import (
        boundary_dephasing_jumps,
        heisenberg_xxz_hamiltonian,
        one_d_chain,
    )

    bench = _load("benchmark_heisenberg_scaling")
    H = heisenberg_xxz_hamiltonian(one_d_chain(N), J=1.0, Delta=1.0)
    jumps = boundary_dephasing_jumps(N)
    L = ls.build_liouvillian(H, jumps, [0.25] * len(jumps))
    # Expected U(1) invariant: one steady state per total-Sz sector.
    assert bench._kernel_dim(L) == N + 1


@pytest.mark.parametrize("N", [2, 3, 4])
def test_sector_ness_is_exactly_stationary(N):
    import liouscope as ls
    from liouscope.core import (
        boundary_dephasing_jumps,
        heisenberg_xxz_hamiltonian,
        one_d_chain,
    )
    from liouscope.numerics.kronecker import vec

    bench = _load("benchmark_heisenberg_scaling")
    H = heisenberg_xxz_hamiltonian(one_d_chain(N), J=1.0, Delta=1.0)
    jumps = boundary_dephasing_jumps(N)
    L = ls.build_liouvillian(H, jumps, [0.25] * len(jumps))
    _rho0, rho_ss = bench._sector_states(N)
    assert np.isclose(np.trace(rho_ss), 1.0)
    # Independent oracle: the sector-projected state is an exact NESS.
    assert float(np.linalg.norm(L @ vec(rho_ss))) < 1.0e-9


def test_diagnose_runs_on_degenerate_model_without_crash():
    """The whole point of B1: diagnose no longer raises on this model."""
    import liouscope as ls
    from liouscope.core import (
        boundary_dephasing_jumps,
        heisenberg_xxz_hamiltonian,
        one_d_chain,
    )

    bench = _load("benchmark_heisenberg_scaling")
    N = 2
    H = heisenberg_xxz_hamiltonian(one_d_chain(N), J=1.0, Delta=1.0)
    jumps = boundary_dephasing_jumps(N)
    L = ls.build_liouvillian(H, jumps, [0.25] * len(jumps))
    rho0, rho_ss = bench._sector_states(N)
    report = ls.diagnose(
        L,
        rho_initial=rho0,
        rho_steady_state=rho_ss,
        bootstrap_B=10,
        include_mpemba=False,
        seed=42,
    )
    assert np.isfinite(float(report.relaxation.beta_D))
