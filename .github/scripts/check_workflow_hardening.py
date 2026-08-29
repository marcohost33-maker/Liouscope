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
# Container images pin by digest, which is a different shape from a git SHA.
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
# ``- uses:`` (list form) is the commonest way to write a step, and the
# previous pattern did not match it: 7 of 27 refs in this tree were
# invisible to the gate, including every ``actions/checkout``. They were
# pinned by discipline, not by this check. Proven as a pair -- the same
# unpinned ref passes in list form and is caught in block form.
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")


def _iter_workflow_files() -> list[Path]:
    if not WORKFLOWS.exists():
        return []
    return sorted(
        path
        for path in WORKFLOWS.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )


def _is_third_party_uses(ref: str) -> bool:
    """Which refs must carry a pin.

    ``docker://`` was exempt outright, so ``docker://org/img:latest`` -- a
    mutable tag from a third party -- passed the gate. It is exempt only
    when it carries an immutable digest. The former ``github.com/``
    exemption was dead code: ``uses:`` does not accept that prefix, and a
    GitHub owner name cannot contain a dot, so no such org can exist. It
    also tripped a CodeQL incomplete-substring alert, which is how it was
    found.
    """
    if ref.startswith("./"):
        return False
    if ref.startswith("docker://"):
        # Third party either way. Whether the pin is adequate is decided
        # in _check_uses_pin, which alone sees the ref undivided.
        return True
    return True


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
    if action.startswith("docker://"):
        # A container pins by image digest, not by a git SHA. Requiring the
        # git form here would reject the very thing the rule asks for.
        if not DIGEST_RE.fullmatch(version):
            errors.append(
                f"{path}:{line_number}: container must be pinned to an "
                f"immutable @sha256: digest, got {ref}"
            )
        return
    if not FULL_SHA_RE.fullmatch(version):
        errors.append(
            f"{path}:{line_number}: action must be pinned to a full 40-char SHA, got {ref}"
        )


def _check_privileged_trigger(path: Path, text: str, errors: list[str]) -> None:
    """The rationale must be a real key, not the words in a comment.

    ``"ALLOW_PULL_REQUEST_TARGET:" in text`` was satisfied by any
    occurrence -- including one inside a ``#`` comment, i.e. by writing the
    name of the waiver rather than declaring it.
    """
    uncommented = chr(10).join(
        line.split("#", 1)[0] for line in text.splitlines()
    )
    if "pull_request_target" in uncommented and (
        "ALLOW_PULL_REQUEST_TARGET:" not in uncommented
    ):
        errors.append(
            f"{path}: uses pull_request_target without ALLOW_PULL_REQUEST_TARGET rationale"
        )


def _check_permissions_declared(path: Path, text: str, errors: list[str]) -> None:
    """Require a top-level permissions block AND check what it grants.

    Presence alone was the whole test, so ``permissions: write-all``
    satisfied it -- a declaration of total access counted as evidence of
    least privilege.
    """
    before_jobs = text.split("\njobs:", 1)[0]
    if "\npermissions:" not in f"\n{before_jobs}":
        errors.append(f"{path}: missing explicit top-level permissions block")
        return
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.split("#", 1)[0].strip()
        if stripped in {"permissions: write-all", "permissions: read-all"}:
            errors.append(
                f"{path}:{number}: blanket '{stripped}' -- grant the "
                f"individual scopes the workflow needs instead"
            )


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
