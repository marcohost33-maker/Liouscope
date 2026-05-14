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


def _parse_manifest_yaml() -> list[dict] | None:
    """Parse the manifest with PyYAML if available. Returns ``None`` otherwise.

    PyYAML is in the ``dev`` extra so this branch runs on CI and on any
    developer machine that has installed dev dependencies.
    """
    try:
        import yaml
    except ImportError:
        return None
    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries: list[dict] = []
    for entry in raw.get("benchmarks", []):
        repro = entry.get("reproduce") or {}
        entries.append({
            "benchmark_id": entry["benchmark_id"],
            "golden": repro.get("golden"),
            "output_hash": repro.get("output_hash"),
        })
    return entries


def _parse_manifest_regex() -> list[dict]:
    """Regex fallback when PyYAML is not installed.

    The manifest format is intentionally constrained; this parser handles
    the ``- benchmark_id``, ``golden`` and ``output_hash`` fields. It is
    deliberately conservative.
    """
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    entries: list[dict] = []
    for match in re.finditer(
        r"- benchmark_id: \"(?P<id>BM-[0-9a-z]+)\"(?P<body>.*?)"
        r"(?=\n  - benchmark_id:|\n[a-z_]+:|\Z)",
        text,
        flags=re.DOTALL,
    ):
        body = match.group("body")
        entry: dict = {"benchmark_id": match.group("id")}
        for key in ("golden", "output_hash"):
            kv = re.search(rf"{key}: \"(?P<val>[^\"]+)\"", body)
            if kv:
                entry[key] = kv.group("val")
            elif re.search(rf"{key}: null", body):
                entry[key] = None
        entries.append(entry)
    return entries


def _parse_manifest() -> list[dict]:
    parsed = _parse_manifest_yaml()
    if parsed is not None:
        return parsed
    return _parse_manifest_regex()


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


def test_yaml_parser_used_in_dev():
    """When PyYAML is installed, the integrity test should use it, not regex.

    This is a meta-test that catches the regression of the YAML branch
    silently falling back to regex (e.g. broken import).
    """
    pytest.importorskip("yaml")
    yaml_entries = _parse_manifest_yaml()
    assert yaml_entries is not None, "PyYAML installed but _parse_manifest_yaml returned None"
    # Sanity: the YAML parser must find every benchmark id present in regex.
    regex_ids = {e["benchmark_id"] for e in _parse_manifest_regex()}
    yaml_ids = {e["benchmark_id"] for e in yaml_entries}
    assert yaml_ids == regex_ids, (
        f"YAML parser missed entries: regex_ids - yaml_ids = {regex_ids - yaml_ids}; "
        f"yaml_ids - regex_ids = {yaml_ids - regex_ids}"
    )
