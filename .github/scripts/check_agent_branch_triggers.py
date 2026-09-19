#!/usr/bin/env python3
"""Keep documented agent branch prefixes and CI push triggers in sync.

AGENTS.md is the repository single source of truth for agent branch prefixes.
The required scientific/quality workflows must run on every one of those
prefixes so stacked agent PRs can obtain exact-head evidence even when their
base is not ``main``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
REQUIRED_WORKFLOWS = (
    ROOT / ".github" / "workflows" / "ci.yml",
    ROOT / ".github" / "workflows" / "ci-qutip.yml",
    ROOT / ".github" / "workflows" / "quality-contract.yml",
)
TASK_PREFIX_RE = re.compile(r"`([A-Za-z0-9_-]+)/<task>`")


def _documented_agent_prefixes() -> set[str]:
    text = AGENTS.read_text(encoding="utf-8")
    prefixes = {m.group(1) for m in TASK_PREFIX_RE.finditer(text)}
    # The same section also documents human-led ``feat|fix|docs/<task>``.
    # That expression is intentionally excluded: it is not one agent prefix.
    return {prefix for prefix in prefixes if "|" not in prefix}


def _push_branches(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    push_index = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "  push:"),
        None,
    )
    if push_index is None:
        raise ValueError(f"{path}: missing top-level push trigger")

    for line in lines[push_index + 1 :]:
        if line and not line.startswith("    "):
            break
        stripped = line.strip()
        if not stripped.startswith("branches:"):
            continue
        raw = stripped.split(":", 1)[1].strip()
        if not (raw.startswith("[") and raw.endswith("]")):
            raise ValueError(
                f"{path}: agent-trigger contract expects inline branches list, got {raw!r}"
            )
        items = []
        for item in raw[1:-1].split(","):
            value = item.strip().strip('"').strip("'")
            if value:
                items.append(value)
        return set(items)
    raise ValueError(f"{path}: push trigger has no branches list")


def main() -> int:
    errors: list[str] = []
    if not AGENTS.exists():
        errors.append("AGENTS.md is missing")
        prefixes: set[str] = set()
    else:
        prefixes = _documented_agent_prefixes()
        if not prefixes:
            errors.append("no agent branch prefixes found in AGENTS.md")

    expected = {"main", *(f"{prefix}/**" for prefix in prefixes)}
    for path in REQUIRED_WORKFLOWS:
        if not path.exists():
            errors.append(f"required workflow missing: {path.relative_to(ROOT)}")
            continue
        try:
            actual = _push_branches(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        missing = sorted(expected - actual)
        if missing:
            errors.append(
                f"{path.relative_to(ROOT)}: push trigger misses documented branches: "
                + ", ".join(missing)
            )

    if errors:
        print("Agent branch trigger contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    patterns = ", ".join(sorted(expected))
    print(
        f"Agent branch trigger contract passed for {len(REQUIRED_WORKFLOWS)} "
        f"workflows: {patterns}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
