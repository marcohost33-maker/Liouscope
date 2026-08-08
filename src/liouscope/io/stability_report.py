"""StabilityReport v2.1 contract (LIOU-RPT-001).

A claim-safe, machine-auditable projection of a :class:`DiagnosticReport`. It is
*additive* over the run manifest (:mod:`liouscope.io.manifest`): it adds an
explicit ``claim_level`` (SAFE/REVIEW/BLOCK) and ``direction``
(slowdown/acceleration/...), the CP evidence level, the three invariant
residuals and an ``evidence_bundle`` + ``provenance`` block. It does **not**
replace or modify the existing manifest or :class:`DiagnosticReport` -- this is
a new contract layer, so older artefacts stay valid (backward-compatible).

New diagnostics that anchors have not yet confirmed are stamped
``claim_status="pending"`` (AGENTS.md "Do": reference ``claim_status:pending``
for new diagnostics until anchors confirm them).

The invariant residuals are computed from the generator and steady state, an
independent re-derivation rather than trusting upstream flags:

* ``trace_preservation_residual = || <<I| M_L ||``  (TP generator identity)
* ``hermiticity_residual        = || rho_ss - rho_ss^dag ||``
* ``stationarity_residual       = || M_L vec(rho_ss) ||``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._consts import DIAGNOSTIC_SCHEMA_VERSION, TAXONOMY_VERSION
from .._types import DiagnosticReport
from ..numerics.kronecker import vec

SCHEMA_VERSION = "liouscope.stability_report.v2.1"

ClaimLevel = Literal["SAFE", "REVIEW", "BLOCK"]
Direction = Literal[
    "slowdown", "acceleration", "protocol-dependent", "ambiguous", "not_applicable"
]
CPEvidenceLevel = Literal[
    "choi_psd_proof", "construction_only", "proxy_sampling", "none"
]

_CLAIM_LEVELS = ("SAFE", "REVIEW", "BLOCK")
_DIRECTIONS = (
    "slowdown",
    "acceleration",
    "protocol-dependent",
    "ambiguous",
    "not_applicable",
)
_CP_LEVELS = ("choi_psd_proof", "construction_only", "proxy_sampling", "none")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Benchmarks, oracles and external references backing a claim."""

    benchmarks: tuple[str, ...] = ()
    oracles: tuple[str, ...] = ()
    external_refs: tuple[str, ...] = ()


def invariant_residuals(
    L_super: np.ndarray,
    rho_ss: np.ndarray,
) -> dict[str, float]:
    """Independently recompute the three invariant residuals."""
    L_super = np.asarray(L_super, dtype=complex)
    rho_ss = np.asarray(rho_ss, dtype=complex)
    d = rho_ss.shape[0]
    iden_vec = vec(np.eye(d, dtype=complex))
    tp = float(np.linalg.norm(iden_vec.conj() @ L_super))
    herm = float(np.linalg.norm(rho_ss - rho_ss.conj().T))
    stat = float(np.linalg.norm(L_super @ vec(rho_ss)))
    return {
        "trace_preservation_residual": tp,
        "hermiticity_residual": herm,
        "stationarity_residual": stat,
    }


def build_stability_report(
    report: DiagnosticReport,
    *,
    L_super: np.ndarray,
    rho_ss: np.ndarray,
    claim_level: ClaimLevel,
    direction: Direction,
    cp_evidence_level: CPEvidenceLevel = "construction_only",
    cp_choi_min_eig: float | None = None,
    evidence_bundle: EvidenceBundle | None = None,
    pending_diagnostics: tuple[str, ...] = (),
    system_label: str | None = None,
    null_dimension: int = 1,
) -> dict[str, Any]:
    """Project a :class:`DiagnosticReport` into a StabilityReport v2.1 payload.

    Parameters
    ----------
    report
        The diagnostic report to project.
    L_super, rho_ss
        Generator and steady state, used to recompute the invariant residuals
        independently of upstream flags.
    claim_level
        SAFE / REVIEW / BLOCK -- the audit verdict for the run.
    direction
        Relaxation direction (slowdown / acceleration / ...).
    cp_evidence_level
        How complete positivity was established: ``choi_psd_proof`` (dense Choi
        PSD gate, :mod:`liouscope.numerics.cptp`), ``construction_only`` (GKSL
        build guarantee), ``proxy_sampling`` (randomised positivity sanity, NOT
        a proof) or ``none``.
    cp_choi_min_eig
        The Choi minimum eigenvalue when a dense Choi gate was run, else None.
    evidence_bundle
        Benchmarks / oracles / external references backing the claim.
    pending_diagnostics
        Diagnostic ids that are not yet anchor-confirmed; each is stamped
        ``claim_status="pending"`` in the ``diagnostics`` block.
    system_label
        Optional human label for the system.
    null_dimension
        Dimension of the Liouvillian null space (1 = unique steady state).

    Raises
    ------
    ValueError
        If ``claim_level``, ``direction`` or ``cp_evidence_level`` is outside
        its allowed set, or ``null_dimension < 1``. Fail-closed so a typo in a
        verdict cannot silently produce an out-of-contract report.
    """
    if claim_level not in _CLAIM_LEVELS:
        raise ValueError(f"claim_level must be one of {_CLAIM_LEVELS}, got {claim_level!r}")
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if cp_evidence_level not in _CP_LEVELS:
        raise ValueError(
            f"cp_evidence_level must be one of {_CP_LEVELS}, got {cp_evidence_level!r}"
        )
    if null_dimension < 1:
        raise ValueError(f"null_dimension must be >= 1, got {null_dimension}")

    bundle = evidence_bundle or EvidenceBundle()
    g = report.governance
    inv: dict[str, float | None] = dict(invariant_residuals(L_super, rho_ss))
    inv["cp_choi_min_eig"] = None if cp_choi_min_eig is None else float(cp_choi_min_eig)

    spectral = report.spectral
    diagnostics: dict[str, Any] = {
        "D1_gap": float(spectral.gap),
        "D2_gns_gap": float(spectral.gns_gap),
        "D2b_kms_gap": float(spectral.kms_gap),
        "D3_oscillating_gap": float(spectral.oscillating_gap),
        "D8_henrici": float(report.nonnorm.henrici_eta),
        "D9_petermann_max": float(report.nonnorm.petermann_max),
        "D10_kreiss": float(report.nonnorm.kreiss),
    }
    # Issue #101 slice A: scale-relative variants are surfaced with an explicit
    # pending claim status (not yet anchor-confirmed) and their value attached,
    # so audits see both the number and its evidentiary standing. NaN (e.g. the
    # zero-operator case) is serialised as None -- dump uses allow_nan=False.
    for did, val in (
        ("D8b_henrici_relative", float(report.nonnorm.henrici_relative)),
        ("D10b_kreiss_scaled", float(report.nonnorm.kreiss_scaled)),
    ):
        diagnostics[did] = {
            "value": None if np.isnan(val) else val,
            "claim_status": "pending",
        }
    for did in pending_diagnostics:
        diagnostics[did] = {"claim_status": "pending"}

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": g.run_id,
        "system": {
            "dimension": int(rho_ss.shape[0]),
            **({"label": system_label} if system_label is not None else {}),
        },
        "generator": {
            "kind": "GKSL",
            "solver_path": str(g.solver_path),
        },
        "invariants": inv,
        "stationary_structure": {
            "is_unique": null_dimension == 1,
            "null_dimension": int(null_dimension),
        },
        "diagnostics": diagnostics,
        "classification": {
            "mechanism": report.classification.a_class,
            "direction": direction,
            "claim_level": claim_level,
            "cp_evidence_level": cp_evidence_level,
        },
        "evidence_bundle": {
            "benchmarks": list(bundle.benchmarks),
            "oracles": list(bundle.oracles),
            "external_refs": list(bundle.external_refs),
        },
        "provenance": {
            "framework_version": g.framework_version,
            "timestamp": g.timestamp,
            "input_hash": g.input_hash,
            "taxonomy_version": TAXONOMY_VERSION,
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        },
    }


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    """Load the bundled ``STABILITY_REPORT_SCHEMA.json`` once."""
    with resources.files("liouscope").joinpath("STABILITY_REPORT_SCHEMA.json").open(
        "r", encoding="utf-8"
    ) as fh:
        schema: dict[str, Any] = json.load(fh)
    return schema


@lru_cache(maxsize=1)
def _compiled_validator() -> Any:
    """Return a cached ``Draft202012Validator`` for the schema, or None."""
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    except ImportError:
        return None
    return Draft202012Validator(_load_schema())


def validate_stability_report(payload: dict[str, Any]) -> None:
    """Validate ``payload`` against the StabilityReport v2.1 schema.

    Runs a built-in subset check on the required fields and value spaces so the
    function never silently passes when :mod:`jsonschema` is absent. When
    :mod:`jsonschema` *is* installed, additionally runs the cached
    ``Draft202012Validator`` for full structural conformance (fail-closed,
    mirroring :func:`liouscope.io.manifest.validate_manifest`).
    """
    required_top = {
        "schema_version",
        "run_id",
        "system",
        "generator",
        "invariants",
        "stationary_structure",
        "diagnostics",
        "classification",
        "evidence_bundle",
        "provenance",
    }
    missing = required_top.difference(payload)
    if missing:
        raise ValueError(
            f"stability report is missing required fields: {sorted(missing)}"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {payload['schema_version']!r}"
        )
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or len(run_id) != 64 or any(
        c not in "0123456789abcdef" for c in run_id
    ):
        raise ValueError("run_id must be a 64-character lowercase hex SHA-256 string")
    classification = payload["classification"]
    for key in ("mechanism", "direction", "claim_level"):
        if key not in classification:
            raise ValueError(f"classification is missing required key {key!r}")
    if classification["claim_level"] not in _CLAIM_LEVELS:
        raise ValueError(f"unknown claim_level {classification['claim_level']!r}")
    if classification["direction"] not in _DIRECTIONS:
        raise ValueError(f"unknown direction {classification['direction']!r}")
    invariants = payload["invariants"]
    for key in (
        "trace_preservation_residual",
        "hermiticity_residual",
        "stationarity_residual",
    ):
        if key not in invariants:
            raise ValueError(f"invariants is missing required key {key!r}")
    bundle = payload["evidence_bundle"]
    for key in ("benchmarks", "oracles", "external_refs"):
        if key not in bundle:
            raise ValueError(f"evidence_bundle is missing required key {key!r}")

    validator = _compiled_validator()
    if validator is not None:
        validator.validate(payload)


def dump_stability_report(payload: dict[str, Any], path: str | Path) -> None:
    """Write a (validated) StabilityReport payload to ``path`` as JSON.

    Validates before writing so an out-of-contract payload fails closed instead
    of producing a non-conforming artefact on disk.
    """
    validate_stability_report(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


__all__ = [
    "SCHEMA_VERSION",
    "EvidenceBundle",
    "build_stability_report",
    "dump_stability_report",
    "invariant_residuals",
    "validate_stability_report",
]
