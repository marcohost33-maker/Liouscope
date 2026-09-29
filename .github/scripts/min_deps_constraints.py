#!/usr/bin/env python3
"""Emit pip constraints that pin every runtime dependency to its declared floor.

``pyproject.toml`` is the single source of the floors (``numpy>=X.Y``); the
``min-deps`` CI job installs exactly ``numpy==X.Y.*`` etc. from this script's
output, so a floor can never be raised or lowered without the job testing the
new value (issue #177). Every runtime dependency must declare a ``>=`` floor;
anything else fails loudly instead of silently testing the latest release.

Stdlib only and regex-based on purpose: the job runs on the declared minimum
Python (3.10), which has no ``tomllib``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_DEPS_BLOCK = re.compile(r"^dependencies\s*=\s*\[(?P<body>.*?)^\]", re.S | re.M)
_ENTRY = re.compile(r'"([^"]+)"')
_FLOOR = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)\s*>=\s*(?P<version>[0-9]+(?:\.[0-9]+)*)$")


def constraints(pyproject: Path = ROOT / "pyproject.toml") -> list[str]:
    match = _DEPS_BLOCK.search(pyproject.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("min-deps: no [project] dependencies list found in pyproject.toml")
    lines: list[str] = []
    for entry in _ENTRY.findall(match.group("body")):
        floor = _FLOOR.match(entry.strip())
        if floor is None:
            raise SystemExit(
                f"min-deps: dependency {entry!r} has no plain '>=' floor; "
                "the minimum-version job cannot test it"
            )
        lines.append(f"{floor.group('name')}=={floor.group('version')}.*")
    if not lines:
        raise SystemExit("min-deps: dependency list is empty")
    return lines


if __name__ == "__main__":
    sys.stdout.write("\n".join(constraints()) + "\n")
