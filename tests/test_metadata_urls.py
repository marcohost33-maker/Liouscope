"""Static checks on the repository's metadata URLs.

These tests fire when the cumulative search-and-replace passes produce
nonsensical URLs (the most common failure being a bulk rename of a
package-namespaced placeholder into something like
``https://arxiv.org/abs/org/repo``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that may contain URLs and that we cover here. Limiting to a
# concrete list keeps the test fast and prevents stray hits from
# auto-generated tooling.
METADATA_FILES = [
    "README.md",
    "CITATION.cff",
    "codemeta.json",
    "SECURITY.md",
    "SECURITY-INSIGHTS.yml",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    ".zenodo.json",
    "MANIFEST_SCHEMA.json",
    "pyproject.toml",
    "LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml",
    "LIOUSCOPE_RELEASE_CHECKLIST.md",
    "LIOUSCOPE_NEGATIVE_RESULTS_REGISTER.md",
    "LIOUSCOPE_BENCHMARK_MANIFEST.yaml",
    "LIOUSCOPE_EVIDENCE_LOCK_REGISTER.csv",
    "LIOUSCOPE_DRIVE_ATTESTATION.csv",
    "ROADMAP_FLOQUET.md",
    "REPRODUCIBILITY.md",
    "CHANGELOG.md",
]

# Bad URL patterns: each tuple is (regex, human-readable reason).
BAD_PATTERNS: list[tuple[str, str]] = [
    # arxiv URLs that look like ``arxiv.org/abs/<org>/<repo>`` -- the
    # arxiv path is ``arxiv.org/abs/<id>`` so an org+repo segment is
    # always wrong. This is exactly the sed-induced bug uncovered in
    # the v1.7 review.
    (r"arxiv\.org/abs/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+",
     "arxiv URL has org/repo path; use a numeric arXiv id"),
    # GitHub URLs pointing at a coworker-research namespace that does
    # not own this repository. The canonical URL is marcohost33-maker.
    (r"https?://github\.com/coworker-research(/|$|\s|\")",
     "stale coworker-research GitHub URL; canonical is marcohost33-maker"),
    # Generic broken-URL placeholders.
    (r"OWNER/(liouscope|Liouscope)", "literal OWNER placeholder"),
]


def _files_to_check():
    return [(name, REPO_ROOT / name) for name in METADATA_FILES if (REPO_ROOT / name).exists()]


@pytest.mark.parametrize("filename, path", _files_to_check())
def test_metadata_file_has_no_bad_url(filename, path):
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    for pattern, reason in BAD_PATTERNS:
        for m in re.finditer(pattern, text):
            failures.append(f"{pattern!r} matched {m.group(0)!r} ({reason})")
    assert not failures, "\n".join([f"{filename}:", *failures])


def test_at_least_one_file_uses_canonical_repo_url():
    """Sanity check: someone must reference the actual GitHub repo URL."""
    needle = "github.com/marcohost33-maker/Liouscope"
    hits = sum(
        1 for _, p in _files_to_check() if needle in p.read_text(encoding="utf-8")
    )
    assert hits >= 5, f"Only {hits} files reference the canonical repo URL"
