"""Reproduce the V1-V5 numerical summary table from the paper.

Run::

    python benchmarks/reproduce_paper.py

The emitted SHA-256 authenticates the deterministic *summary payload* only.
Performance metadata is intentionally outside that hash domain.  The digest is
not a substitute for the per-run LiouScope manifest and does not claim that
floating-point results are bit-identical across BLAS/LAPACK implementations.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import liouscope as ls
from liouscope.examples import all_systems

OUT_DIR = Path(__file__).resolve().parent / "output"
PAYLOAD_SCHEMA_VERSION = "1.0.0"
_HASH_DOMAIN = b"liouscope:reproduce-paper-summary:v1\x00"


def build_payload(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Assemble the deterministic, hash-eligible summary payload.

    The payload contains result values and the version identifiers that define
    their semantics.  Timing, host and platform measurements are excluded.
    """
    return {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "framework_version": ls.__version__,
        "taxonomy_version": ls.TAXONOMY_VERSION,
        "diagnostic_schema_version": ls.DIAGNOSTIC_SCHEMA_VERSION,
        "seed": int(seed),
        "rows": rows,
    }


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize the supported payload deterministically and fail on NaN/Inf.

    This is a deliberately narrow Python/JSON contract, not a claim of full
    RFC 8785 JSON Canonicalization Scheme conformance.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_payload(payload: dict[str, Any]) -> str:
    """Return a domain-separated SHA-256 of the deterministic payload."""
    digest_input = _HASH_DOMAIN + canonical_payload_bytes(payload)
    return hashlib.sha256(digest_input).hexdigest()


def main(seed: int = 42, bootstrap_B: int = 100) -> int:
    OUT_DIR.mkdir(exist_ok=True)
    ls.seed_everything(seed)
    rows: list[dict[str, Any]] = []
    perf_per_system: dict[str, float] = {}
    total_started = time.perf_counter()

    for system in all_systems():
        started = time.perf_counter()
        report = ls.diagnose(
            system.L,
            rho_initial=system.rho_initial,
            bootstrap_B=bootstrap_B,
            seed=seed,
            include_mpemba=True,
        )
        elapsed = time.perf_counter() - started

        # Hash full Python float values, not presentation-rounded values.  A
        # digest change therefore exposes any change visible in this summary.
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

    payload = build_payload(rows, seed)
    digest = digest_payload(payload)

    # The outer artefact is intentionally not byte-reproducible because it
    # records performance.  Only `payload` is authenticated by `sha256`.
    artefact = {
        "payload": payload,
        "sha256": digest,
        "sha256_domain": _HASH_DOMAIN.rstrip(b"\x00").decode("ascii"),
        "perf": {
            "wall_seconds_total": round(total_elapsed, 3),
            "wall_seconds_per_system": perf_per_system,
            "note": "performance metadata; excluded from sha256 by design",
        },
    }
    out_path = OUT_DIR / "reproduce_paper.json"
    out_path.write_text(
        json.dumps(
            artefact,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
