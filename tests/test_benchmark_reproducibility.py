"""Regression tests for the benchmark reproducibility and NESS fixes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _BENCH_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _boundary_dephasing_fixture(N: int) -> np.ndarray:
    import liouscope as ls
    from liouscope.core import (
        boundary_dephasing_jumps,
        heisenberg_xxz_hamiltonian,
        one_d_chain,
    )

    H = heisenberg_xxz_hamiltonian(one_d_chain(N), J=1.0, Delta=1.0)
    jumps = boundary_dephasing_jumps(N)
    return ls.build_liouvillian(H, jumps, [0.25] * len(jumps))


# --------------------------------------------------------------------------- #
# Deterministic summary-payload hash
# --------------------------------------------------------------------------- #


def test_reproduce_hash_excludes_performance_metadata() -> None:
    reproduce = _load("reproduce_paper")
    rows = [{"name": "sys", "dim": 4, "delta": 0.5, "beta_D": 1.25}]
    payload = reproduce.build_payload(rows, seed=42)

    assert "perf" not in payload
    assert "wall_seconds" not in payload
    assert all("wall_seconds" not in row for row in payload["rows"])
    assert reproduce.digest_payload(payload) == reproduce.digest_payload(
        reproduce.build_payload(rows, seed=42)
    )


def test_reproduce_digest_changes_when_any_hashed_result_changes() -> None:
    reproduce = _load("reproduce_paper")
    first = reproduce.build_payload(
        [{"name": "sys", "dim": 4, "delta": 0.5, "beta_D": 1.25}],
        seed=42,
    )
    second = reproduce.build_payload(
        [{"name": "sys", "dim": 4, "delta": 0.5, "beta_D": 1.2500000000001}],
        seed=42,
    )
    assert reproduce.digest_payload(first) != reproduce.digest_payload(second)


def test_reproduce_canonical_serialization_rejects_nonfinite_values() -> None:
    reproduce = _load("reproduce_paper")
    payload = reproduce.build_payload(
        [{"name": "sys", "dim": 4, "delta": float("nan")}],
        seed=42,
    )
    with pytest.raises(ValueError):
        reproduce.digest_payload(payload)


def test_performance_metadata_is_outside_authenticated_payload() -> None:
    reproduce = _load("reproduce_paper")
    payload = reproduce.build_payload(
        [{"name": "sys", "dim": 4, "delta": 0.5}], seed=42
    )
    digest = reproduce.digest_payload(payload)

    artefact_fast = {
        "payload": payload,
        "sha256": digest,
        "perf": {"wall_seconds_total": 0.001},
    }
    artefact_slow = {
        "payload": payload,
        "sha256": digest,
        "perf": {"wall_seconds_total": 999.0},
    }
    assert artefact_fast["sha256"] == artefact_slow["sha256"] == digest
    assert reproduce.digest_payload(artefact_fast["payload"]) == digest
    assert reproduce.digest_payload(artefact_slow["payload"]) == digest


# --------------------------------------------------------------------------- #
# Degenerate full NESS and unique symmetry-resolved attractor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N", [2, 3, 4])
@pytest.mark.parametrize("scale", [1.0e-10, 1.0, 1.0e10])
def test_fixture_nullity_is_rate_scale_invariant(N: int, scale: float) -> None:
    benchmark = _load("benchmark_heisenberg_scaling")
    L = _boundary_dephasing_fixture(N)
    assert benchmark._nullity(scale * L) == N + 1


@pytest.mark.parametrize("N", [2, 3, 4])
def test_selected_charge_block_is_invariant_and_has_unique_ness(N: int) -> None:
    benchmark = _load("benchmark_heisenberg_scaling")
    L = _boundary_dephasing_fixture(N)
    rho0, _rho_ss = benchmark._sector_states(N)
    mask = benchmark._sector_mask(N, rho0)
    inside = benchmark._sector_operator_indices(mask)
    outside = np.setdiff1d(np.arange(L.shape[0]), inside)

    leakage = np.asarray(L)[np.ix_(outside, inside)]
    relative_leakage = float(np.linalg.norm(leakage)) / max(
        float(np.linalg.norm(L)), np.finfo(float).tiny
    )
    assert relative_leakage < 1.0e-12

    L_sector = benchmark._sector_liouvillian(L, mask)
    assert benchmark._nullity(L_sector) == 1
    benchmark._assert_sector_relaxes(L_sector)


@pytest.mark.parametrize("N", [2, 3, 4])
def test_sector_ness_has_scale_relative_stationarity_residual(N: int) -> None:
    benchmark = _load("benchmark_heisenberg_scaling")
    L = _boundary_dephasing_fixture(N)
    _rho0, rho_ss = benchmark._sector_states(N)

    assert np.isclose(np.trace(rho_ss), 1.0)
    assert benchmark._relative_stationarity_residual(L, rho_ss) < 1.0e-10
    for scale in (1.0e-10, 1.0e10):
        assert (
            benchmark._relative_stationarity_residual(scale * L, rho_ss)
            < 1.0e-10
        )


def test_diagnose_runs_with_explicit_symmetry_resolved_ness() -> None:
    import liouscope as ls

    benchmark = _load("benchmark_heisenberg_scaling")
    N = 2
    L = _boundary_dephasing_fixture(N)
    rho0, rho_ss = benchmark._sector_states(N)
    report = ls.diagnose(
        L,
        rho_initial=rho0,
        rho_steady_state=rho_ss,
        bootstrap_B=10,
        include_mpemba=False,
        seed=42,
    )
    assert np.isfinite(float(report.relaxation.beta_D))
