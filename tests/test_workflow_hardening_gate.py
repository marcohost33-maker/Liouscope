"""The workflow hardening gate, tested against the holes it used to miss.

``.github/scripts/check_workflow_hardening.py`` enforces AGENTS.md section 4 on
every workflow in this repository, and until now nothing enforced *it*. It
passed green while four classes of unsafe workflow walked straight through, so
its green was evidence of nothing.

Each test below is a **pair**: a workflow that must be rejected, next to a
positive control that must still be accepted. Rejection alone would be
satisfied by a gate that fails everything, which is exactly as useless as one
that passes everything.

Why the gate is exercised as a subprocess on a synthetic tree rather than by
importing its functions: the thing under test is the verdict the CI job acts
on -- its exit code -- not the internals. A test of the helpers would keep
passing if ``main`` stopped calling one of them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

GATE = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "check_workflow_hardening.py"
)

# A real, currently-pinned action reference, so the control workflows differ
# from the unsafe ones in exactly one respect.
PINNED = "actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8"

CLEAN = f"""name: clean
on: [push]
permissions:
  contents: read
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - uses: {PINNED}
"""


def _run_gate(tmp_path: Path, workflow: str) -> int:
    """Run the gate over a tree containing exactly one workflow."""
    workflows = tmp_path / ".github" / "workflows"
    scripts = tmp_path / ".github" / "scripts"
    workflows.mkdir(parents=True)
    scripts.mkdir(parents=True)
    copy = scripts / GATE.name
    copy.write_text(GATE.read_text(encoding="utf-8"), encoding="utf-8")
    (workflows / "probe.yml").write_text(workflow, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(copy)], capture_output=True, text=True
    ).returncode


def test_a_clean_workflow_is_accepted(tmp_path: Path) -> None:
    """Positive control. Without it, every rejection below is meaningless."""
    assert _run_gate(tmp_path, CLEAN) == 0


@pytest.mark.parametrize(
    ("name", "workflow"),
    [
        (
            # The pattern was ``^\\s*uses:``, which does not match the list form
            # -- the commonest way to write a step. Seven refs in this repo were
            # invisible to the gate, every ``actions/checkout`` among them. They
            # were pinned by discipline, not by this check.
            "unpinned action in list form",
            """name: l1
on: [push]
permissions:
  contents: read
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - uses: someorg/some-action@main
""",
        ),
        (
            # ``docker://`` was exempt outright, so a mutable third-party tag
            # was waved through. Only an immutable digest earns the exemption.
            "docker:// with a mutable tag",
            """name: l2
on: [push]
permissions:
  contents: read
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - uses: docker://someorg/image:latest
""",
        ),
        (
            # Presence of a ``permissions`` block was the entire test, so a
            # declaration of total access counted as evidence of least privilege.
            "blanket permissions: write-all",
            f"""name: l3
on: [push]
permissions: write-all
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - uses: {PINNED}
""",
        ),
        (
            # The rationale was checked with a substring test over the raw file,
            # so writing the waiver's NAME in a comment satisfied it.
            "pull_request_target justified only in a comment",
            f"""name: l4
on: [pull_request_target]
# ALLOW_PULL_REQUEST_TARGET: we have our reasons
permissions:
  contents: read
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - uses: {PINNED}
""",
        ),
    ],
)
def test_the_gate_rejects(name: str, workflow: str, tmp_path: Path) -> None:
    """Each of these passed the gate before 2026-08-29."""
    assert _run_gate(tmp_path, workflow) != 0, (
        f"the gate accepted a workflow it must reject: {name}"
    )


def test_a_digest_pinned_container_is_still_allowed(tmp_path: Path) -> None:
    """Over-correction guard for the ``docker://`` rule.

    Tightening that exemption must not outlaw containers as such -- an
    immutable digest is exactly the thing the pin requirement asks for, so
    refusing it too would be the same mistake in the other direction.
    """
    digest = "sha256:" + "0" * 64
    workflow = f"""name: ok
on: [push]
permissions:
  contents: read
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - uses: docker://someorg/image@{digest}
"""
    assert _run_gate(tmp_path, workflow) == 0


def test_the_repository_itself_passes_its_own_gate() -> None:
    """The tightened gate must hold on the real tree, not only on fixtures.

    This is the check that would have caught the tightening if it had made the
    repository's own workflows non-compliant.
    """
    assert (
        subprocess.run(
            [sys.executable, str(GATE)], capture_output=True, text=True
        ).returncode
        == 0
    )
