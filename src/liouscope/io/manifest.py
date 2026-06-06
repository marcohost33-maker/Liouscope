"""Run manifest construction with SHA-256 reproducibility id."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys
from functools import lru_cache
from importlib import resources
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
    """Deterministic timezone-aware UTC timestamp.

    ``datetime.utcnow()`` is deprecated in 3.12 and removed in 3.14. We
    use ``now(tz=utc)``, pin ``timespec="microseconds"`` so the string
    length is identical on every call, and rewrite the trailing
    ``+00:00`` offset to the canonical ``Z`` suffix (RFC 3339 / ISO 8601
    "Zulu" form) so manifest hashes stay stable across platforms.

    ``datetime.timezone.utc`` is used rather than ``datetime.UTC`` because
    the latter only exists in Python 3.11+; the package supports 3.10.

    **Reproducible-builds override.** When the environment variable
    ``SOURCE_DATE_EPOCH`` is set (the cross-ecosystem reproducible-builds
    standard, <https://reproducibility.org/specs/source-date-epoch/>), the
    wall-clock read is replaced by that fixed Unix timestamp. This makes the
    *entire* manifest — timestamp included — byte-identical across runs, which
    is what the README's strong reproducibility claim relies on. An unparseable
    or negative value is rejected fail-closed (no silent fallback to wall clock,
    which would re-introduce non-determinism unnoticed).
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None and epoch.strip() != "":
        try:
            ts = int(epoch)
        except ValueError as exc:
            raise ValueError(
                f"SOURCE_DATE_EPOCH must be an integer Unix timestamp, got {epoch!r}"
            ) from exc
        if ts < 0:
            raise ValueError(f"SOURCE_DATE_EPOCH must be non-negative, got {ts}")
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    else:
        dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


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
    """Write the schema-compliant manifest JSON for ``report`` to ``path``.

    Parent directories are created if missing so a nested artefact path does
    not fail with a bare ``FileNotFoundError`` on first write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(manifest_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    """Load the bundled ``MANIFEST_SCHEMA.json`` once.

    Uses :mod:`importlib.resources` so the lookup works whether the
    package is installed editable, from a wheel, or from a zipfile. The
    schema is checked into ``src/liouscope/MANIFEST_SCHEMA.json`` and
    listed in :file:`pyproject.toml` under ``[tool.setuptools.package-data]``.
    """
    with resources.files("liouscope").joinpath("MANIFEST_SCHEMA.json").open(
        "r", encoding="utf-8"
    ) as fh:
        schema: dict[str, Any] = json.load(fh)
    return schema


@lru_cache(maxsize=1)
def _compiled_validator() -> Any:
    """Return a cached ``Draft202012Validator`` for the manifest schema.

    Per the python-jsonschema docs the high-throughput pattern is to
    instantiate a specific draft validator once and reuse it, instead of
    calling :func:`jsonschema.validate` which auto-detects the draft on
    every call. Returns ``None`` when :mod:`jsonschema` is not installed.
    """
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    except ImportError:
        return None
    return Draft202012Validator(_load_schema())


def validate_manifest(payload: dict[str, Any]) -> None:
    """Validate ``payload`` against :file:`MANIFEST_SCHEMA.json`.

    Runs a built-in subset check on the required fields and value spaces
    so the function never silently passes when :mod:`jsonschema` is
    absent. When :mod:`jsonschema` *is* installed, additionally runs the
    cached ``Draft202012Validator`` against the bundled schema for full
    structural conformance.
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

    validator = _compiled_validator()
    if validator is not None:
        validator.validate(payload)


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
