"""Reproduce the V1-V5 numerical summary table from the paper.

Run::

    python benchmarks/reproduce_paper.py

The emitted SHA-256 is an integrity/determinism digest of the exact summary
payload for a fixed code/environment/seed/parameter set. Performance metadata
is outside the hash domain. The digest is not a substitute for per-run
LiouScope manifests and is not an approximate cross-platform equivalence test:
scientific reproduction across BLAS/LAPACK builds must compare the reported
metrics with declared numerical tolerances.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import liouscope as ls
from liouscope.examples import all_systems

OUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_SCHEMA_VERSION = "1.1.0"
_HASH_DOMAIN = b"liouscope:reproduce-paper-summary:v1\x00"
_HASH_CONTRACT = {
    "algorithm": "sha256",
    "domain": _HASH_DOMAIN.rstrip(b"\x00").decode("ascii"),
    "canonicalization": "python-json-sort-keys-utf8-v1",
    "numeric_semantics": "exact serialized finite float values",
}
_HASHED_FIELDS = (
    "hash_contract",
    "summary_schema_version",
    "framework_version",
    "taxonomy_version",
    "schema_version",
    "seed",
    "parameters",
    "environment",
    "rows",
)


def build_payload(
    rows: list[dict[str, Any]],
    seed: int,
    *,
    bootstrap_B: int = 100,
    include_mpemba: bool = True,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic, hash-eligible summary payload.

    The legacy top-level keys ``schema_version``, ``seed`` and ``rows`` are
    retained so existing consumers of the old flat JSON artefact continue to
    work. New provenance fields are additive.
    """
    if bootstrap_B < 1:
        raise ValueError("bootstrap_B must be >= 1")
    return {
        "hash_contract": dict(_HASH_CONTRACT),
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "framework_version": ls.__version__,
        "taxonomy_version": ls.TAXONOMY_VERSION,
        # Backward-compatible key name: this is the diagnostic schema version.
        "schema_version": ls.DIAGNOSTIC_SCHEMA_VERSION,
        "seed": int(seed),
        "parameters": {
            "bootstrap_B": int(bootstrap_B),
            "include_mpemba": bool(include_mpemba),
        },
        "environment": dict(environment or {}),
        "rows": rows,
    }


def canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize the supported payload deterministically and fail on NaN/Inf.

    This is a deliberately narrow Python/JSON contract, not a claim of full
    RFC 8785 JSON Canonicalization Scheme conformance.
    """
    missing = [field for field in _HASHED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"summary payload missing hashed fields: {missing}")
    projected = {field: payload[field] for field in _HASHED_FIELDS}
    return json.dumps(
        projected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_payload(payload: Mapping[str, Any]) -> str:
    """Return a domain-separated SHA-256 of the deterministic payload."""
    digest_input = _HASH_DOMAIN + canonical_payload_bytes(payload)
    return hashlib.sha256(digest_input).hexdigest()


def build_artifact(
    payload: Mapping[str, Any],
    *,
    perf: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a backward-compatible flat artefact with authenticated payload."""
    digest = digest_payload(payload)
    return {
        **dict(payload),
        "sha256": digest,
        "perf": dict(perf),
    }


def payload_from_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the authenticated projection from a written summary artefact."""
    missing = [field for field in _HASHED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"summary artefact missing hashed fields: {missing}")
    return {field: artifact[field] for field in _HASHED_FIELDS}


def verify_artifact(artifact: Mapping[str, Any]) -> bool:
    """Return whether the stored SHA-256 matches the authenticated projection."""
    stored = artifact.get("sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        return False
    expected = digest_payload(payload_from_artifact(artifact))
    return hmac.compare_digest(stored, expected)


def _environment_from_report(report: Any) -> dict[str, str]:
    governance = report.governance
    return {
        "python_version": str(governance.python_version),
        "numpy_version": str(governance.numpy_version),
        "scipy_version": str(governance.scipy_version),
        "platform": str(governance.platform),
    }


def main(seed: int = 42, bootstrap_B: int = 100) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ls.seed_everything(seed)
    rows: list[dict[str, Any]] = []
    perf_per_system: dict[str, float] = {}
    environment: dict[str, str] | None = None
    include_mpemba = True
    total_started = time.perf_counter()

    for system in all_systems():
        started = time.perf_counter()
        report = ls.diagnose(
            system.L,
            rho_initial=system.rho_initial,
            bootstrap_B=bootstrap_B,
            seed=seed,
            include_mpemba=include_mpemba,
        )
        elapsed = time.perf_counter() - started

        report_environment = _environment_from_report(report)
        if environment is None:
            environment = report_environment
        elif report_environment != environment:
            raise RuntimeError(
                "inconsistent governance environments across summary rows: "
                f"{environment!r} != {report_environment!r}"
            )

        # Exact finite values are hashed for artefact-integrity checks. They are
        # intentionally not rounded into a false cross-platform equality claim.
        rows.append(
            {
                "name": system.name,
                "dim": int(system.L.shape[0]),
                "delta": float(report.spectral.gap),
                "delta_s": float(report.spectral.gns_gap),
                "kms": float(report.spectral.kms_gap),
                "petermann_max": float(report.nonnorm.petermann_max),
                "kreiss": float(report.nonnorm.kreiss),
                "model": report.relaxation.aicc_model,
                "beta_D": float(report.relaxation.beta_D),
                "a_class": report.classification.a_class,
                "f_family": report.classification.f_family,
                "verdict": report.classification.verdict,
                "tier": report.classification.tier,
                "confidence": float(report.classification.confidence),
                "run_id": report.governance.run_id,
                "input_hash": report.governance.input_hash,
                "solver_path": report.governance.solver_path,
                "quality_label": report.governance.quality_label,
            }
        )
        perf_per_system[system.name] = round(elapsed, 3)
        print(
            f"  {system.name:<22} {elapsed:6.2f}s  "
            f"Delta={report.spectral.gap:.4f}  "
            f"a={report.classification.a_class:>3}  "
            f"verdict={report.classification.verdict}"
        )

    total_elapsed = time.perf_counter() - total_started
    print(f"\nTotal wall time: {total_elapsed:.2f}s")

    payload = build_payload(
        rows,
        seed,
        bootstrap_B=bootstrap_B,
        include_mpemba=include_mpemba,
        environment=environment,
    )
    artifact = build_artifact(
        payload,
        perf={
            "wall_seconds_total": round(total_elapsed, 3),
            "wall_seconds_per_system": perf_per_system,
            "note": "performance metadata; excluded from sha256 by design",
        },
    )
    if not verify_artifact(artifact):
        raise RuntimeError("internal error: summary artefact digest did not verify")

    out_path = OUT_DIR / "reproduce_paper.json"
    out_path.write_text(
        json.dumps(
            artifact,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SHA-256: {artifact['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
