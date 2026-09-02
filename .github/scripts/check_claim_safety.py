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
# Scope hardening 2026-09-02. The previous scope was README + CHANGELOG + docs/.
# A negative control showed that `paper/paper.md` -- the JOSS submission, i.e.
# the single most public-facing scientific claim surface this repository has --
# accepted "production-ready and externally certified" with the gate green.
# Governance-facing prose (AGENTS/CONTRIBUTING/SECURITY) and the citation
# abstract are equally public and equally unguarded.
DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "AGENTS.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "CITATION.cff",
    ROOT / "docs",
    ROOT / "paper",
]
DOC_SUFFIXES = (".md", ".cff")
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

# Qualifiers that make a risky phrase acceptable. Two rules, both learned from
# a negative control on 2026-09-02:
#
#   1. The qualifier must sit in a WINDOW around the matched phrase, not merely
#      somewhere on the same line. The old line-wide test accepted
#      "LiouScope is production-ready and there are no open gaps." -- the bare
#      substring "no " matched and disabled the whole line. That is the same
#      failure class PR #129 found in the sibling workflow gate: a regex that
#      misses the commonest way to write the thing it guards.
#   2. Bare negations are matched on WORD BOUNDARIES and only count when they
#      precede the risky phrase (a trailing negation about something else is
#      not a qualification of the claim).
#
# Window size: qualifiers in real prose sit adjacent to the claim
# ("not production-ready", "production-ready only after ..."). 60 characters is
# wide enough for "is not yet considered production-ready" and narrow enough to
# exclude an unrelated clause later in the sentence.
QUALIFIER_WINDOW = 60

# Matched anywhere in the window (before or after the risky phrase): these are
# unambiguous evidence-status qualifiers, not generic negations.
STATUS_QUALIFIERS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bunverified\b",
        r"\bopen gap\b",
        r"\bmissing evidence\b",
        r"\bnot for diagnostic or operational use\b",
        r"\bdoes not certify\b",
        r"\bplaceholder\b",
        r"\bnot yet\b",
        r"\bUNVERIFIED\b",
        # "Directly verified against <named record>" is linked evidence, which
        # is exactly what this gate asks for. Without it the widened scope
        # flagged CITATION.cff's DOI provenance note -- a line that carries its
        # evidence in the very next clause.
        r"\bdirectly verified\b",
        r"\bverified against\b",
    )
]

# Matched only in the window BEFORE the risky phrase: bare negations qualify a
# claim only when they govern it.
LEADING_NEGATIONS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bdoes not\b",
        r"\bmust not\b",
        r"\bcannot\b",
        r"\bunless\b",
        r"\buntil\b",
        r"\bwithout\b",
    )
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
                    for path in root.rglob("*")
                    if path.is_file()
                    and path.suffix in DOC_SUFFIXES
                    and not _should_skip(path)
                )
            )
    return paths


def _match_is_qualified(line: str, start: int, end: int) -> bool:
    """Return True iff the risky phrase at ``line[start:end]`` is qualified.

    Scoped to a window around the match, not to the whole line -- see the
    comment on :data:`QUALIFIER_WINDOW` for the negative control that forced
    this.
    """
    before = line[max(0, start - QUALIFIER_WINDOW) : start]
    after = line[end : end + QUALIFIER_WINDOW]
    # The matched SPAN is part of the status window: several risk patterns are
    # themselves clause-wide (``\bDOI\b.*\b(published|released)\b``), so a
    # qualifier such as "placeholder" can sit INSIDE the match rather than
    # beside it. Excluding the span made "The DOI is a placeholder until the
    # archive gate is released." a false positive.
    window = before + " " + line[start:end] + " " + after
    if any(pattern.search(window) for pattern in STATUS_QUALIFIERS):
        return True
    return any(pattern.search(before) for pattern in LEADING_NEGATIONS)


def main() -> int:
    errors: list[str] = []
    docs = _iter_docs()

    for path in docs:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in RISK_PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue
                if _match_is_qualified(line, match.start(), match.end()):
                    continue
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

    print(f"Claim-safety check passed for {len(docs)} public-facing files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
