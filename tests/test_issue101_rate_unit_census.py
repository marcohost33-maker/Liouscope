"""Issue #101 slice C, phase 0: the rate-unit census harness measures what it says.

``claim_status: pending`` (KANON ``RESEARCH-LIOUSCOPE``).

``benchmarks/issue101_rate_unit_census.py`` is a measurement, not a gate. These
tests pin the harness itself: the families are what their design labels say,
the two splits never share a system, the result digest is reproducible, and
the metamorphic relation ``L -> cL``, ``t -> t / c`` is applied exactly. They
assert nothing about how stable the classifier is. That number is the
census's output, and fixing it is the calibration study's job.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import liouscope as ls

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
_NAME = "issue101_rate_unit_census"


@cache
def _census() -> Any:
    spec = importlib.util.spec_from_file_location(_NAME, _BENCH_DIR / f"{_NAME}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
    return module


@cache
def _report(family: str, index: int) -> ls.DiagnosticReport:
    system = _census().FAMILIES[family](index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ls.diagnose(
            system.L,
            rho_initial=system.rho_initial,
            rho_steady_state=system.rho_steady,
            bootstrap_B=20,
            seed=1,
        )


@cache
def _tiny_census() -> dict[str, Any]:
    return _census().run_census(
        1, families=("N4", "P1"), rate_factors=(1.0, 10.0)
    )


def test_calibration_and_holdout_never_share_a_system() -> None:
    census = _census()
    calibration = census.split_indices(50, "calibration")
    holdout = census.split_indices(50, "holdout")
    assert not set(calibration) & set(holdout)
    # Enlarging a split only appends: no system moves between splits.
    assert census.split_indices(80, "calibration")[:50] == calibration
    assert census.split_indices(80, "holdout")[:50] == holdout
    with pytest.raises(ValueError, match="unknown split"):
        census.split_indices(3, "test")


@pytest.mark.parametrize("family", ["N1", "N2", "N3", "N4", "P1"])
@pytest.mark.parametrize("index", [0, 1, 6])
def test_family_systems_are_deterministic_gksl_generators(
    family: str, index: int
) -> None:
    make = _census().FAMILIES[family]
    a, b = make(index), make(index)
    assert np.array_equal(a.L, b.L) and np.array_equal(a.rho_initial, b.rho_initial)
    d = a.rho_initial.shape[0]
    assert a.L.shape == (d * d, d * d)
    # Trace preservation: vec(I)^H L = 0, to the rounding of forming L.
    trace_row = np.eye(d).reshape(-1, order="F") @ a.L
    assert np.abs(trace_row).max() <= 1.0e-12 * np.abs(a.L).max()
    # A density matrix: Hermitian, unit trace, positive semidefinite.
    rho = a.rho_initial
    assert np.allclose(rho, rho.conj().T)
    assert np.trace(rho).real == pytest.approx(1.0)
    assert np.linalg.eigvalsh(rho).min() >= -1.0e-12
    assert a.f5_label == ("positive" if family == "P1" else "negative")
    # Different indices are different systems.
    assert not np.array_equal(make(index).L, make(index + 2).L)


def test_the_positive_control_is_the_anchor_suites_phantom_ladder() -> None:
    """P1 index 0 is the operator the anchor suite classifies A10/F5."""
    d = 4
    low = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        low[i, i + 1] = 1.0
    expected = ls.build_liouvillian(
        np.zeros((d, d), dtype=complex),
        [low, low.conj().T, np.diag(np.arange(d, dtype=complex))],
        [1.5, 0.1, 0.15],
    )
    assert np.array_equal(_census().directional_ladder(0).L, expected)
    rep = _report("P1", 0)
    fired = [h["rule_id"] for h in rep.classification.triggered_hypotheses]
    assert "F5_PSEUDOSPECTRAL" in fired


@pytest.mark.parametrize(("family", "gapless"), [
    ("N1", False), ("N2", False), ("N3", True), ("N4", False), ("P1", False),
])
def test_family_premises_hold_at_unit_rate(family: str, gapless: bool) -> None:
    """The gapless family has an unresolved gap, the others a resolved one."""
    spectral = _report(family, 0).spectral
    if gapless:
        assert abs(spectral.gap) <= spectral.zero_cutoff
    else:
        assert spectral.gap > 1.0e6 * spectral.zero_cutoff


@pytest.mark.parametrize("index", [0, 2, 4])
def test_pure_dephasing_has_a_degenerate_stationary_manifold(index: int) -> None:
    """N2: one zero mode per conserved population, and a resolved gap above."""
    system = _census().pure_dephasing(index)
    spectral = _report("N2", index).spectral
    cert = spectral.zero_mode_certificate
    assert cert is not None
    assert cert["zero_mode_count"] == system.rho_initial.shape[0] >= 2


def test_pure_dephasing_is_normal() -> None:
    """N2 is the NORMAL control: no departure from normality."""
    ev = _report("N2", 0).classification.evidence
    assert ev["henrici_relative"] <= 1.0e-12


def test_the_metamorphic_relation_is_applied_exactly() -> None:
    """``c L`` on the grid ``t_ref / c``: rates scale by ``c``, the rest holds.

    Checked on the gapped, non-normal control, where every quantity is
    resolved. Tolerances are relative and of the size of the rounding in
    forming ``c * L`` and solving: 1e-9 is about 1e6 ulps.
    """
    census = _census()
    system = census.driven_damped(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reports = {
            c: ls.diagnose(
                c * system.L,
                rho_initial=system.rho_initial,
                t_grid=np.linspace(*census.REFERENCE_GRID) / c,
                bootstrap_B=20,
                seed=1,
            )
            for c in (0.1, 1.0, 10.0)
        }
    base = reports[1.0]
    for c, rep in reports.items():
        # Rate-valued: homogeneous of degree one.
        assert rep.spectral.gap == pytest.approx(c * base.spectral.gap, rel=1e-9)
        assert rep.nonnorm.henrici_eta == pytest.approx(
            c * base.nonnorm.henrici_eta, rel=1e-9
        )
        # Dimensionless by construction (slice A).
        ev, ev0 = rep.classification.evidence, base.classification.evidence
        assert ev["henrici_relative"] == pytest.approx(ev0["henrici_relative"], rel=1e-9)
        assert ev["kreiss_scaled"] == pytest.approx(ev0["kreiss_scaled"], rel=1e-9)
        # Same curve in rescaled time: the trajectory end point agrees.
        curve, curve0 = (
            rep.relaxation.trace_distance_curve, base.relaxation.trace_distance_curve
        )
        assert curve is not None and curve0 is not None
        assert curve[-1] == pytest.approx(curve0[-1], rel=1e-6, abs=1e-12)


def test_census_payload_is_complete_and_reproducible() -> None:
    census = _census()
    payload = _tiny_census()
    # Determinism: a second run of one family reproduces its rows exactly.
    again = census.run_census(1, families=("N4",), rate_factors=(1.0, 10.0))
    assert again["systems"][0] == payload["systems"][0]
    assert payload["claim_status"] == "pending"
    assert payload["split"] == "calibration"
    assert payload["time_grid"] == "t_ref / c"
    assert [s["family"] for s in payload["systems"]] == ["N4", "P1"]
    for system in payload["systems"]:
        assert [r["c"] for r in system["runs"]] == [1.0, 10.0]
        assert set(system["key_range"]) == set(census.DECISION_KEYS)
        classes = {(r["a_class"], r["f_family"]) for r in system["runs"]}
        assert system["class_stable"] == (len(classes) == 1)
        assert system["n_distinct_classes"] == len(classes)
    # RFC 8259: the payload serialises without NaN or Infinity tokens.
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["result_sha256"] == payload["result_sha256"]


def test_summary_counts_match_the_per_system_rows() -> None:
    payload = _tiny_census()
    for family, row in payload["summary"].items():
        rows = [s for s in payload["systems"] if s["family"] == family]
        assert row["n"] == len(rows)
        assert row["class_stable"] == sum(s["class_stable"] for s in rows)
        assert row["verdict_stable"] == sum(s["verdict_stable"] for s in rows)
        for i, c in enumerate(payload["rate_factors"]):
            assert row["f5_fired_by_rate_factor"][repr(c)] == sum(
                s["runs"][i]["f5_fired"] for s in rows
            )


def test_the_digest_covers_the_result_not_the_platform() -> None:
    """Two platforms that agree on every number produce the same digest."""
    payload = dict(_tiny_census())
    digest = payload.pop("result_sha256")
    payload.pop("provenance")
    import hashlib

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == digest


def test_relative_range_is_zero_for_an_invariant_key() -> None:
    census = _census()
    assert census.relative_range([2.0, 2.0, 2.0]) == 0.0
    assert census.relative_range([None, 2.0, 8.0]) == pytest.approx(0.75)
    # Withheld at all but one rate factor: nothing to compare.
    assert census.relative_range([None, 2.0]) is None
    assert census.relative_range([]) is None
    # Vanishes at one rate factor and not at another.
    assert census.relative_range([0.0, 3.0]) == 1.0


def test_relative_range_does_not_read_round_off_zeros_as_variation() -> None:
    """Measured in the first census run: seven values between 2e-17 and 2e-16
    gave max / min = 13 for a key that is zero throughout."""
    census = _census()
    values = [1.5e-16, 1.2e-16, 2.5e-17, 3.9e-17, 1.6e-16, 1.3e-16, 8.0e-17]
    assert max(values) / min(values) > 6.0
    assert census.relative_range(values) == 0.0
    assert census.relative_range([0.0, 1.0e-13, 0.0]) == 0.0
