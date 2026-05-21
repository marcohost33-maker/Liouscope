"""Run manifest construction with SHA-256 reproducibility id."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .._consts import (
    DIAGNOSTIC_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    QUALITY_EXPLORATORY,
    QUALITY_MODERATE,
    QUALITY_STABLE,
    TAXONOMY_VERSION,
)
from .._types import DiagnosticReport, GovernanceMetadata, QualityLabel, SolverPath
from .._version import __version__


def compute_input_hash(*objects: object) -> str:
    """Return a deterministic SHA-256 hash of the given input objects.

    NumPy arrays are hashed via their byte content (cast to C-contiguous
    canonical-dtype form). Other objects are hashed via repr.
    """
    h = hashlib.sha256()
    for obj in objects:
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            h.update(arr.dtype.str.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        else:
            h.update(repr(obj).encode())
    return h.hexdigest()


def _classify_quality(report_tier: str) -> QualityLabel:
    if report_tier == "PUBLICATION_GRADE":
        return QUALITY_STABLE  # type: ignore[return-value]
    if report_tier == "CONFIRMATION":
        return QUALITY_MODERATE  # type: ignore[return-value]
    return QUALITY_EXPLORATORY  # type: ignore[return-value]


def make_run_id(input_hash: str, seed: int, framework_version: str) -> str:
    """Combine input hash, seed and framework version into a 64-char SHA-256."""
    h = hashlib.sha256()
    h.update(input_hash.encode())
    h.update(str(seed).encode())
    h.update(framework_version.encode())
    return h.hexdigest()


def _utc_now_iso() -> str:
    """Timezone-aware UTC timestamp.

    ``datetime.utcnow()`` is deprecated in 3.12 and slated for removal in
    3.14. We format ``now(tz=utc)`` and rewrite the trailing ``+00:00``
    offset to the canonical ``Z`` suffix so manifest hashes stay stable.
    """
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


def build_manifest(
    *,
    input_hash: str,
    seed: int,
    solver_path: SolverPath,
    tier: str,
) -> GovernanceMetadata:
    """Build the per-run :class:`GovernanceMetadata`."""
    framework_version = __version__
    run_id = make_run_id(input_hash, seed, framework_version)
    return GovernanceMetadata(
        run_id=run_id,
        framework_version=framework_version,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        seed=seed,
        solver_path=solver_path,
        quality_label=_classify_quality(tier),
        timestamp=_utc_now_iso(),
        input_hash=input_hash,
    )


def manifest_payload(report: DiagnosticReport) -> dict[str, Any]:
    """Return a dict matching :file:`MANIFEST_SCHEMA.json` exactly.

    This is the schema-compliant projection of a :class:`DiagnosticReport`
    used for governance / archival. It is intentionally a small subset of
    the full report -- the full structured result is what
    :func:`liouscope.io.dump_report` writes.
    """
    g = report.governance
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "framework_version": g.framework_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "run_id": g.run_id,
        "timestamp": g.timestamp,
        "python_version": g.python_version,
        "platform": g.platform,
        "numpy_version": g.numpy_version,
        "scipy_version": g.scipy_version,
        "seed": int(g.seed),
        "solver_path": str(g.solver_path),
        "quality_label": str(g.quality_label),
        "input_hash": g.input_hash,
    }


def dump_manifest(report: DiagnosticReport, path: str | Path) -> None:
    """Write the schema-compliant manifest JSON for ``report`` to ``path``."""
    Path(path).write_text(json.dumps(manifest_payload(report), indent=2, sort_keys=True))


def validate_manifest(payload: dict[str, Any]) -> None:
    """Validate ``payload`` against :file:`MANIFEST_SCHEMA.json`.

    Uses :mod:`jsonschema` when available; falls back to a built-in subset
    check on the required fields so the function never silently passes.
    """
    required_fields = {
        "schema_version",
        "framework_version",
        "taxonomy_version",
        "diagnostic_schema_version",
        "run_id",
        "timestamp",
        "python_version",
        "platform",
        "numpy_version",
        "scipy_version",
        "seed",
        "quality_label",
    }
    missing = required_fields.difference(payload)
    if missing:
        raise ValueError(f"manifest payload is missing required fields: {sorted(missing)}")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}, "
            f"got {payload['schema_version']!r}"
        )
    if payload["taxonomy_version"] != TAXONOMY_VERSION:
        raise ValueError(
            f"taxonomy_version must be {TAXONOMY_VERSION!r}, "
            f"got {payload['taxonomy_version']!r}"
        )
    if payload["diagnostic_schema_version"] != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError(
            f"diagnostic_schema_version must be {DIAGNOSTIC_SCHEMA_VERSION!r}, "
            f"got {payload['diagnostic_schema_version']!r}"
        )
    if not isinstance(payload["seed"], int) or payload["seed"] < 0:
        raise ValueError("seed must be a non-negative int")
    if len(payload["run_id"]) != 64 or any(c not in "0123456789abcdef" for c in payload["run_id"]):
        raise ValueError("run_id must be a 64-character lowercase hex SHA-256 string")
    if payload["quality_label"] not in {QUALITY_STABLE, QUALITY_MODERATE, QUALITY_EXPLORATORY}:
        raise ValueError(f"unknown quality_label {payload['quality_label']!r}")
    if payload.get("solver_path") not in (None, "dense", "sparse_arpack"):
        raise ValueError(f"unknown solver_path {payload['solver_path']!r}")

    # Optional strict JSON Schema validation when jsonschema is installed.
    try:
        import jsonschema  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        return
    schema_path = Path(__file__).resolve().parents[3] / "MANIFEST_SCHEMA.json"
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(instance=payload, schema=schema)


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "TAXONOMY_VERSION",
    "build_manifest",
    "compute_input_hash",
    "dump_manifest",
    "make_run_id",
    "manifest_payload",
    "validate_manifest",
]
