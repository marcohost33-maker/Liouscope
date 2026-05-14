"""Static gate: every GitHub Action ``uses:`` reference must be SHA-pinned.

OpenSSF Scorecard's ``Pinned-Dependencies`` check requires actions to be
referenced by a 40-character commit SHA rather than a floating tag. A
force-push to a tag could otherwise inject malicious code into workflows
that hold ``id-token: write`` for Trusted Publishing.

This test scans every workflow under ``.github/workflows/`` and asserts
that every non-local action reference matches::

    uses: <owner>/<repo>[/<path>]@<40-hex-sha> # <human readable tag>

A failure here means a recent Dependabot bump or manual edit re-introduced
a floating tag. Bump dependent SHAs in the same PR.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Matches `uses: <action-ref>` where action-ref starts with anything other
# than ``./`` (which would be a local action and is allowed to use a tag).
USES_LINE = re.compile(r"^\s*-?\s*uses:\s*(?P<ref>[^\s]+)\s*(#\s*(?P<tag>[^\n]+))?")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _scan_workflows() -> list[tuple[str, int, str]]:
    """Return ``[(filename, lineno, action_ref), ...]`` for every uses-line.

    Skips local actions (``uses: ./...``) and the special pypi-publish
    action that is published only under a ``release/v1`` branch (an
    intentional moving target maintained by PyPA -- their published
    guidance is to pin via SHA, which we do).
    """
    results: list[tuple[str, int, str]] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = USES_LINE.match(line)
            if not m:
                continue
            ref = m.group("ref").strip().strip('"').strip("'")
            if ref.startswith("./") or ref.startswith("docker://"):
                continue  # local / docker actions are allowed
            results.append((path.name, i, ref))
    return results


@pytest.mark.parametrize(
    "filename, lineno, ref",
    _scan_workflows(),
    ids=lambda x: x if isinstance(x, str) else str(x),
)
def test_action_is_sha_pinned(filename, lineno, ref):
    """Every ``uses:`` reference must point at a 40-char commit SHA."""
    # ``<owner>/<repo>(/<path>)?@<ref>``
    if "@" not in ref:
        pytest.fail(f"{filename}:{lineno} uses {ref!r} without a pin")
    _, ref_after_at = ref.rsplit("@", 1)
    assert SHA40.match(ref_after_at), (
        f"{filename}:{lineno} uses {ref!r} -- expected 40-char commit SHA "
        f"(OpenSSF Scorecard Pinned-Dependencies)"
    )


def test_at_least_one_workflow_pin_exists():
    """Sanity: the scanner finds at least the test workflow's checkout."""
    rows = _scan_workflows()
    assert rows, "no uses: lines found under .github/workflows/"
    # The CI workflow must reference actions/checkout at least once.
    assert any("actions/checkout" in ref for _, _, ref in rows), (
        "no actions/checkout pin found across workflows -- did the directory move?"
    )
