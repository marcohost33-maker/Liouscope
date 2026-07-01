#!/usr/bin/env python3
"""Static workflow hardening checks for LiouScope.

This intentionally avoids third-party dependencies so the gate can run in GitHub
Actions with only the standard library.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
USES_RE = re.compile(r"^\s*uses:\s*([^\s#]+)")


def _iter_workflow_files() -> list[Path]:
    if not WORKFLOWS.exists():
        return []
    return sorted(
        path
        for path in WORKFLOWS.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )


def _is_third_party_uses(ref: str) -> bool:
    return not (
        ref.startswith("./")
        or ref.startswith("docker://")
        or ref.startswith("github.com/")
    )


def _check_uses_pin(path: Path, line_number: int, line: str, errors: list[str]) -> None:
    match = USES_RE.match(line)
    if not match:
        return

    ref = match.group(1).strip().strip('"').strip("'")
    if "@" not in ref:
        if _is_third_party_uses(ref):
            errors.append(f"{path}:{line_number}: third-party action is missing @ref: {ref}")
        return

    action, version = ref.rsplit("@", 1)
    if not _is_third_party_uses(action):
        return
    if not FULL_SHA_RE.fullmatch(version):
        errors.append(
            f"{path}:{line_number}: action must be pinned to a full 40-char SHA, got {ref}"
        )


def _check_privileged_trigger(path: Path, text: str, errors: list[str]) -> None:
    if "pull_request_target" in text and "ALLOW_PULL_REQUEST_TARGET:" not in text:
        errors.append(
            f"{path}: uses pull_request_target without ALLOW_PULL_REQUEST_TARGET rationale"
        )


def _check_permissions_declared(path: Path, text: str, errors: list[str]) -> None:
    # Minimal parser: require an explicit top-level permissions key before jobs.
    before_jobs = text.split("\njobs:", 1)[0]
    if "\npermissions:" not in f"\n{before_jobs}":
        errors.append(f"{path}: missing explicit top-level permissions block")


def main() -> int:
    workflow_files = _iter_workflow_files()
    errors: list[str] = []

    if not workflow_files:
        errors.append("no GitHub workflow files found under .github/workflows")

    for path in workflow_files:
        text = path.read_text(encoding="utf-8")
        _check_permissions_declared(path, text, errors)
        _check_privileged_trigger(path, text, errors)
        for line_number, line in enumerate(text.splitlines(), start=1):
            _check_uses_pin(path, line_number, line, errors)

    if errors:
        print("Workflow hardening check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Workflow hardening check passed for {len(workflow_files)} workflow files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
