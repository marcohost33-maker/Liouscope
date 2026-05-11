"""Run manifest construction with SHA-256 reproducibility id."""

from __future__ import annotations

import datetime
import hashlib
import platform
import sys

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
from .._types import GovernanceMetadata, QualityLabel, SolverPath
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
        timestamp=datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        input_hash=input_hash,
    )


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "TAXONOMY_VERSION",
    "build_manifest",
    "compute_input_hash",
    "make_run_id",
]
