"""Manifest <-> golden-output integrity gate.

For every benchmark entry that ships a golden JSON fixture, the manifest's
``reproduce.output_hash`` must equal the SHA-256 of that file. If a
maintainer regenerates the golden but forgets to bump the manifest hash,
this test fires.

Pair with ``tests/test_benchmark_outputs.py`` which goes the other way:
re-run the runner and compare against the golden file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "LIOUSCOPE_BENCHMARK_MANIFEST.yaml"


def _parse_manifest() -> list[dict]:
    """Tiny YAML-less parser for the benchmarks list.

    We deliberately do not import PyYAML here so the integrity test runs
    even when the optional dependency is missing. The grammar of the
    manifest is intentionally constrained to a few keys per benchmark
    entry; we extract those keys with a regex sweep.
    """
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    entries: list[dict] = []
    for match in re.finditer(
        r"- benchmark_id: \"(?P<id>BM-[0-9a-z]+)\"(?P<body>.*?)(?=\n  - benchmark_id:|\n[a-z_]+:|\Z)",
        text,
        flags=re.DOTALL,
    ):
        body = match.group("body")
        entry = {"benchmark_id": match.group("id")}
        for key in ("golden", "output_hash"):
            kv = re.search(rf"{key}: \"(?P<val>[^\"]+)\"", body)
            if kv:
                entry[key] = kv.group("val")
            else:
                if re.search(rf"{key}: null", body):
                    entry[key] = None
        entries.append(entry)
    return entries


@pytest.mark.parametrize("entry", _parse_manifest(), ids=lambda e: e["benchmark_id"])
def test_manifest_golden_hash_matches(entry):
    """``reproduce.output_hash`` must match the SHA-256 of the golden file."""
    golden_rel = entry.get("golden")
    expected = entry.get("output_hash")
    if not golden_rel:
        pytest.skip(f"{entry['benchmark_id']}: no golden fixture declared")
    if not expected:
        pytest.skip(f"{entry['benchmark_id']}: no output_hash declared")
    golden_path = REPO_ROOT / golden_rel
    assert golden_path.exists(), f"golden file missing: {golden_path}"
    actual = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"{entry['benchmark_id']}: manifest declares {expected[:12]}..., "
        f"golden file {golden_rel} hashes to {actual[:12]}..."
    )
