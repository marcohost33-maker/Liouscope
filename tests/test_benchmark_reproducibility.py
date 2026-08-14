"""Regression tests for the benchmark integrity and symmetry-resolved NESS fixes."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


@cache
def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _BENCH_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@cache
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
# Deterministic summary-payload integrity digest
# --------------------------------------------------------------------------- #


def _summary_rows(beta: float = 1.25) -> list[dict[str, object]]:
    return [{"name": "sys", "dim": 4, "delta": 0.5, "beta_D": beta}]


def test_summary_payload_retains_legacy_flat_fields() -> None:
    reproduce = _load("reproduce_paper")
    payload = reproduce.build_payload(_summary_rows(), seed=42)

    assert payload["schema_version"]
    assert payload["seed"] == 42
    assert payload["rows"] == _summary_rows()
    assert payload["summary_schema_version"] == "1.1.0"
    assert payload["parameters"] == {
        "bootstrap_B": 100,
        "include_mpemba": True,
    }


def test_summary_digest_excludes_only_performance_metadata() -> None:
    reproduce = _load("reproduce_paper")
    payload = reproduce.build_payload(_summary_rows(), seed=42)
    artifact_fast = reproduce.build_artifact(
        payload,
        perf={"wall_seconds_total": 0.001},
    )
    artifact_slow = reproduce.build_artifact(
        payload,
        perf={"wall_seconds_total": 999.0},
    )

    assert artifact_fast["rows"] == payload["rows"]
    assert artifact_fast["sha256"] == artifact_slow["sha256"]
    assert reproduce.verify_artifact(artifact_fast)
    assert reproduce.verify_artifact(artifact_slow)


def test_summary_digest_changes_with_result_or_run_parameter() -> None:
    reproduce = _load("reproduce_paper")
    base = reproduce.build_payload(_summary_rows(), seed=42, bootstrap_B=100)
    changed_result = reproduce.build_payload(
        _summary_rows(beta=1.2500000000001),
        seed=42,
        bootstrap_B=100,
    )
    changed_parameter = reproduce.build_payload(
        _summary_rows(),
        seed=42,
        bootstrap_B=101,
    )

    assert reproduce.digest_payload(base) != reproduce.digest_payload(changed_result)
    assert reproduce.digest_payload(base) != reproduce.digest_payload(changed_parameter)


def test_summary_artifact_verification_detects_payload_or_contract_tampering() -> None:
    reproduce = _load("reproduce_paper")
    payload = reproduce.build_payload(_summary_rows(), seed=42)
    artifact = reproduce.build_artifact(payload, perf={})
    assert reproduce.verify_artifact(artifact)

    changed_row = dict(artifact)
    changed_row["rows"] = _summary_rows(beta=9.0)
    assert not reproduce.verify_artifact(changed_row)

    changed_contract = dict(artifact)
    changed_contract["hash_contract"] = {
        **artifact["hash_contract"],
        "numeric_semantics": "rounded",
    }
    assert not reproduce.verify_artifact(changed_contract)


def test_summary_canonical_serialization_rejects_nonfinite_values() -> None:
    reproduce = _load("reproduce_paper")
    payload = reproduce.build_payload(
        [{"name": "sys", "dim": 4, "delta": float("nan")}],
        seed=42,
    )
    with pytest.raises(ValueError):
        reproduce.digest_payload(payload)


def test_summary_payload_rejects_invalid_bootstrap_count() -> None:
    reproduce = _load("reproduce_paper")
    with pytest.raises(ValueError, match="bootstrap_B"):
        reproduce.build_payload(_summary_rows(), seed=42, bootstrap_B=0)


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

    assert benchmark._relative_block_leakage(L, mask) < 1.0e-12
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


def test_sector_selection_rejects_state_spanning_multiple_charges() -> None:
    benchmark = _load("benchmark_heisenberg_scaling")
    # Equal mixture of |00> and |11>: expectation <Sz_tot>=0, but the state has
    # support in the +1 and -1 sectors and none in the m=0 sector.
    rho = np.diag([0.5, 0.0, 0.0, 0.5]).astype(complex)
    with pytest.raises(ValueError, match="exactly one total-magnetisation"):
        benchmark._sector_mask(2, rho)


def test_sector_relaxation_rejects_defective_zero_mode() -> None:
    benchmark = _load("benchmark_heisenberg_scaling")
    jordan_zero = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(AssertionError, match="defective zero mode"):
        benchmark._assert_sector_relaxes(jordan_zero)


def test_sector_relaxation_accepts_simple_zero_with_decaying_mode() -> None:
    benchmark = _load("benchmark_heisenberg_scaling")
    stable = np.diag([0.0, -1.0]).astype(complex)
    benchmark._assert_sector_relaxes(stable)


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
