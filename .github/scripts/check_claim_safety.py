#!/usr/bin/env python3
"""Documentation claim-safety check for LiouScope.

The goal is not to ban strong claims forever; the goal is to prevent public-facing
status, release, certification, and deployment claims from appearing without
explicit qualifiers or linked evidence.

Release-audit files are intentionally excluded: they are the evidence ledger where
PyPI/DOI/release gaps and evidence are discussed in detail. Public status claims
should point to those files instead of duplicating unchecked wording.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_PATHS = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "docs"]
SKIP_NAMES = {"QUALITY_WORKFLOW_OS.md"}
SKIP_PREFIXES = ("RELEASE_AUDIT_",)

RISK_PATTERNS = [
    re.compile(r"\bproduction[- ]ready\b", re.IGNORECASE),
    re.compile(r"\bexternally certified\b", re.IGNORECASE),
    re.compile(r"\bclinical(?:ly)? validated\b", re.IGNORECASE),
    re.compile(r"\boperational(?:ly)? validated\b", re.IGNORECASE),
    re.compile(r"\bPyPI[- ]published\b", re.IGNORECASE),
    re.compile(r"\bDOI\b.*\b(complete|published|archived|released)\b", re.IGNORECASE),
    re.compile(r"\bZenodo\b.*\b(complete|published|archived|released)\b", re.IGNORECASE),
    re.compile(r"\brelease[- ]complete\b", re.IGNORECASE),
]

ALLOW_MARKERS = [
    "not ",
    "no ",
    "does not ",
    "must not ",
    "unless ",
    "until ",
    "unverified",
    "open gap",
    "missing evidence",
    "not for diagnostic or operational use",
    "does not certify",
    "placeholder",
]


def _should_skip(path: Path) -> bool:
    name = path.name
    return name in SKIP_NAMES or any(name.startswith(prefix) for prefix in SKIP_PREFIXES)


def _iter_docs() -> list[Path]:
    paths: list[Path] = []
    for root in DOC_PATHS:
        if root.is_file() and not _should_skip(root):
            paths.append(root)
        elif root.is_dir():
            paths.extend(
                sorted(
                    path
                    for path in root.rglob("*.md")
                    if path.is_file() and not _should_skip(path)
                )
            )
    return paths


def _line_is_allowed(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in ALLOW_MARKERS)


def main() -> int:
    errors: list[str] = []
    docs = _iter_docs()

    for path in docs:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _line_is_allowed(line):
                continue
            for pattern in RISK_PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(ROOT)
                    errors.append(
                        f"{rel}:{line_number}: risky unsupported claim wording: {line.strip()}"
                    )
                    break

    if errors:
        print("Claim-safety check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(
            "\nFix by adding evidence, linking a release audit, or marking the status as not/UNVERIFIED.",
            file=sys.stderr,
        )
        return 1

    print(f"Claim-safety check passed for {len(docs)} public-facing markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
