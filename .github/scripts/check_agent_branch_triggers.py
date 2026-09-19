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
MERGE_SMOKE_CALLEE = ROOT / ".github" / "workflows" / "ci-python-local.yml"
DUAL_EVIDENCE_WORKFLOWS = {
    ROOT / ".github" / "workflows" / "ci-qutip.yml",
    ROOT / ".github" / "workflows" / "quality-contract.yml",
}
PR_HEAD_REF = "${{ github.event.pull_request.head.sha }}"
PR_ONLY_IF = "${{ github.event_name == 'pull_request' }}"
TASK_PREFIX_RE = re.compile(r"`([A-Za-z0-9_-]+)/<task>`")
BLOCK_SCALAR_RE = re.compile(r"[:=-]\\s*[|>][+-]?\\s*$")


def _documented_agent_prefixes() -> set[str]:
    text = AGENTS.read_text(encoding="utf-8")
    prefixes = {m.group(1) for m in TASK_PREFIX_RE.finditer(text)}
    # The same section also documents human-led ``feat|fix|docs/<task>``.
    # That expression is intentionally excluded: it is not one agent prefix.
    return {prefix for prefix in prefixes if "|" not in prefix}


def _structural_lines(path: Path) -> list[tuple[int, int, str]]:
    """Return structural YAML lines, excluding block-scalar payload."""
    out: list[tuple[int, int, str]] = []
    scalar_indent: int | None = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if scalar_indent is not None:
            if indent > scalar_indent:
                continue
            scalar_indent = None
        code = _yaml_code(raw)
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip(" "))
        stripped = code.strip()
        out.append((lineno, indent, stripped))
        if BLOCK_SCALAR_RE.search(stripped):
            scalar_indent = indent
    return out

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


def _yaml_code(line: str) -> str:
    """Return the structural part of a simple repository workflow line."""
    return line.split("#", 1)[0].rstrip()


def _pull_request_is_unfiltered(path: Path) -> bool:
    """Require default PR activity coverage with no base/path suppression."""
    lines = path.read_text(encoding="utf-8").splitlines()
    pr_index = next(
        (
            i
            for i, line in enumerate(lines)
            if _yaml_code(line).rstrip() == "  pull_request:"
        ),
        None,
    )
    if pr_index is None:
        return False

    forbidden = ("branches:", "branches-ignore:", "paths:", "paths-ignore:", "types:")
    for line in lines[pr_index + 1 :]:
        code = _yaml_code(line)
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip(" "))
        if indent < 4:
            break
        stripped = code.strip()
        if stripped.startswith(forbidden):
            return False
    return True


def _checkout_ref_values(path: Path) -> list[str | None]:
    """Return with.ref for every actions/checkout step, None if absent."""
    lines = path.read_text(encoding="utf-8").splitlines()
    refs: list[str | None] = []
    for i, line in enumerate(lines):
        code = _yaml_code(line)
        stripped = code.strip()
        short_form = stripped.startswith("- uses: actions/checkout@")
        named_form = stripped.startswith("uses: actions/checkout@")
        if not (short_form or named_form):
            continue
        uses_indent = len(code) - len(code.lstrip(" "))
        # ``- uses:`` is itself the list item; ``uses:`` under ``- name:`` is
        # a peer property. In both cases this is the indentation where ``with:``
        # must appear.
        property_indent = uses_indent + 2 if short_form else uses_indent
        in_with = False
        ref_value: str | None = None
        for next_line in lines[i + 1 :]:
            next_code = _yaml_code(next_line)
            if not next_code.strip():
                continue
            indent = len(next_code) - len(next_code.lstrip(" "))
            next_stripped = next_code.strip()
            if indent < property_indent:
                break
            if indent == property_indent:
                if next_stripped == "with:":
                    in_with = True
                    continue
                # Another peer property or the next list item ends ``with``.
                in_with = False
                if next_stripped.startswith("- "):
                    break
                continue
            if in_with and indent > property_indent and next_stripped.startswith("ref:"):
                ref_value = next_stripped.split(":", 1)[1].strip().strip(chr(34)).strip(chr(39))
        refs.append(ref_value)
    return refs


def _checks_out_exact_pr_head(path: Path) -> bool:
    """At least one checkout must bind to the submitted PR head SHA."""
    return PR_HEAD_REF in _checkout_ref_values(path)


def _also_checks_out_proposed_merge(path: Path) -> bool:
    """At least one checkout must deliberately retain GitHub PR merge semantics."""
    return None in _checkout_ref_values(path)


def _job_level_uses_values(path: Path) -> list[str]:
    """Return workflow-level job ``uses`` targets, excluding steps/comments."""
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = _yaml_code(line)
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip(" "))
        stripped = code.strip()
        # In this repository a reusable-workflow job property is indented four
        # spaces under its job id. Step-level uses entries are deeper/list items.
        if indent == 4 and stripped.startswith("uses:"):
            values.append(stripped.split(":", 1)[1].strip().strip(chr(34)).strip(chr(39)))
    return values


def _merge_smoke_keeps_default_merge_ref() -> bool:
    """Bind merge-smoke verification to the local workflow actually invoked."""
    if not MERGE_SMOKE_WORKFLOW.exists():
        return False
    targets = _job_level_uses_values(MERGE_SMOKE_WORKFLOW)
    expected = "./.github/workflows/ci-python-local.yml"
    if targets.count(expected) != 1 or len(targets) != 1:
        return False
    callee = ROOT / expected.removeprefix("./")
    if callee != MERGE_SMOKE_CALLEE or not callee.exists():
        return False
    refs = _checkout_ref_values(callee)
    return bool(refs) and all(ref is None for ref in refs)


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
                f"{path.relative_to(ROOT)}: at least one checkout must use the exact "
                f"PR head ref {PR_HEAD_REF!r}"
            )
        if path in DUAL_EVIDENCE_WORKFLOWS and not _also_checks_out_proposed_merge(path):
            errors.append(
                f"{path.relative_to(ROOT)}: must also contain a checkout with no "
                "with.ref so the proposed pull_request merge is exercised"
            )

    if not MERGE_SMOKE_WORKFLOW.exists():
        errors.append(
            f"merge-smoke workflow missing: {MERGE_SMOKE_WORKFLOW.relative_to(ROOT)}"
        )
    elif not _pull_request_is_unfiltered(MERGE_SMOKE_WORKFLOW):
        errors.append(
            f"{MERGE_SMOKE_WORKFLOW.relative_to(ROOT)}: merge-smoke pull_request "
            "must cover arbitrary base branches and default PR activity types"
        )
    if not _merge_smoke_keeps_default_merge_ref():
        errors.append(
            "merge-smoke contract drifted: ci-reusable-pilot.yml must call "
            "ci-python-local.yml and that callee actions/checkout step must "
            "omit with.ref so the synthetic PR merge ref is exercised"
        )

    if errors:
        print("Agent branch trigger contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    patterns = ", ".join(sorted(expected))
    print(
        f"Agent branch trigger contract passed for {len(REQUIRED_WORKFLOWS)} "
        f"exact-head plus required merge-integration coverage: {patterns}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
