#!/usr/bin/env python3
"""Keep documented agent branches and stacked-PR CI coverage in sync.

AGENTS.md is the repository single source of truth for agent branch prefixes.
Required scientific/quality workflows must cover those prefixes on ordinary
pushes AND must accept pull requests to arbitrary base branches. The latter is
the reliable path for stacked PR verification: automation-authenticated pushes
may legitimately suppress recursive GitHub Actions events.
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
MERGE_SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "ci-reusable-pilot.yml"
PR_HEAD_REF = "${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}"
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


def _pull_request_is_unfiltered(path: Path) -> bool:
    """Return True iff pull_request exists and has no base-branch filter."""
    lines = path.read_text(encoding="utf-8").splitlines()
    pr_index = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "  pull_request:"),
        None,
    )
    if pr_index is None:
        return False
    for line in lines[pr_index + 1 :]:
        if line and not line.startswith("    "):
            break
        stripped = line.strip()
        if stripped.startswith("branches:") or stripped.startswith("branches-ignore:"):
            return False
    return True


def _checks_out_exact_pr_head(path: Path) -> bool:
    """Require PR jobs to checkout the submitted head SHA, not only merge ref."""
    text = path.read_text(encoding="utf-8")
    return f"ref: {PR_HEAD_REF}" in text

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
        if not _pull_request_is_unfiltered(path):
            errors.append(
                f"{path.relative_to(ROOT)}: pull_request must cover arbitrary base "
                "branches (no branches/branches-ignore filter), so stacked PRs "
                "receive exact-head CI"
            )
        if not _checks_out_exact_pr_head(path):
            errors.append(
                f"{path.relative_to(ROOT)}: checkout must use the exact PR head ref "
                f"{PR_HEAD_REF!r}; GitHub's default pull_request checkout is the "
                "synthetic merge commit"
            )

    if not MERGE_SMOKE_WORKFLOW.exists():
        errors.append(
            f"merge-smoke workflow missing: {MERGE_SMOKE_WORKFLOW.relative_to(ROOT)}"
        )
    elif not _pull_request_is_unfiltered(MERGE_SMOKE_WORKFLOW):
        errors.append(
            f"{MERGE_SMOKE_WORKFLOW.relative_to(ROOT)}: merge-smoke pull_request "
            "must cover arbitrary base branches"
        )

    if errors:
        print("Agent branch trigger contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    patterns = ", ".join(sorted(expected))
    print(
        f"Agent branch trigger contract passed for {len(REQUIRED_WORKFLOWS)} "
        f"exact-head workflows plus merge-smoke coverage: {patterns}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
