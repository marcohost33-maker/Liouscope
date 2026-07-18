"""Reproduce the V1-V5 numerical summary table from the paper.

Run::

    python benchmarks/reproduce_paper.py

Emits a deterministic SHA-256 hash of the result table when given a fixed
seed; rerunning on the same platform must produce the same hash.

Determinism contract
--------------------
The hash covers **only** run-invariant fields: rounded result arrays, the
seed, and the framework/taxonomy/schema versions. Wall-clock timing is a
platform-dependent performance metric, NOT a physics result, so it is recorded
separately under ``perf`` and is **excluded** from the hash. Including timing in
the hashed payload (the previous behaviour) made every run produce a different
digest, defeating the reproducibility check. See :func:`build_payload` /
:func:`digest_payload`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import liouscope as ls
from liouscope.examples import all_systems

OUT_DIR = Path(__file__).resolve().parent / "output"


def build_payload(rows: list[dict], seed: int) -> dict:
    """Assemble the deterministic, hash-eligible payload.

    Contains only run-invariant fields. No timing, no platform metadata.
    """
    return {
        "framework_version": ls.__version__,
        "taxonomy_version": ls.TAXONOMY_VERSION,
        "schema_version": ls.DIAGNOSTIC_SCHEMA_VERSION,
        "seed": seed,
        "rows": rows,
    }


def digest_payload(payload: dict) -> str:
    """SHA-256 over the canonical JSON of the deterministic payload."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def main(seed: int = 42, bootstrap_B: int = 100) -> int:
    OUT_DIR.mkdir(exist_ok=True)
    ls.seed_everything(seed)
    rows: list[dict] = []
    perf_per_system: dict[str, float] = {}  # non-hashed timing metadata
    t_total = time.perf_counter()
    for sys_obj in all_systems():
        t0 = time.perf_counter()
        report = ls.diagnose(
            sys_obj.L,
            rho_initial=sys_obj.rho_initial,
            bootstrap_B=bootstrap_B,
            seed=seed,
            include_mpemba=True,
        )
        dt = time.perf_counter() - t0
        # Result row: run-invariant physics only. Timing lives in `perf`.
        rows.append({
            "name": sys_obj.name,
            "dim": sys_obj.L.shape[0],
            "delta": round(report.spectral.gap, 6),
            "delta_s": round(report.spectral.gns_gap, 6),
            "kms": round(report.spectral.kms_gap, 6),
            "petermann_max": round(report.nonnorm.petermann_max, 6),
            "kreiss": round(report.nonnorm.kreiss, 6),
            "model": report.relaxation.aicc_model,
            "beta_D": round(float(report.relaxation.beta_D), 6),
            "a_class": report.classification.a_class,
            "f_family": report.classification.f_family,
            "verdict": report.classification.verdict,
            "tier": report.classification.tier,
            "confidence": round(report.classification.confidence, 3),
        })
        perf_per_system[sys_obj.name] = round(dt, 3)
        print(f"  {sys_obj.name:<22} {dt:6.2f}s  "
              f"Delta={report.spectral.gap:.4f}  "
              f"a={report.classification.a_class:>3}  "
              f"verdict={report.classification.verdict}")

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal wall time: {elapsed:.2f}s")

    payload = build_payload(rows, seed)
    digest = digest_payload(payload)

    # Written artefact = deterministic payload + its own hash + a clearly
    # separated, NON-hashed `perf` block. The digest is computed over `payload`
    # BEFORE perf is attached, so it stays stable across runs.
    out_path = OUT_DIR / "reproduce_paper.json"
    artefact = {
        **payload,
        "sha256": digest,
        "perf": {
            "wall_seconds_total": round(elapsed, 3),
            "wall_seconds_per_system": perf_per_system,
            "note": "performance metadata; excluded from sha256 by design",
        },
    }
    out_path.write_text(json.dumps(artefact, sort_keys=True, indent=2))
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
