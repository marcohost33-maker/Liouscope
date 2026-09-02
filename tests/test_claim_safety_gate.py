"""Negative controls for ``.github/scripts/check_claim_safety.py``.

Why this file exists
--------------------
``check_workflow_hardening.py`` got ``tests/test_workflow_hardening_gate.py``
after PR #129 found it blind to the commonest way to write a step. Its sibling
``check_claim_safety.py`` had **no test at all**, and on 2026-09-02 a negative
control found it blind in two ways:

1. The allow-list was matched against the WHOLE LINE as bare substrings, so any
   unrelated negation disabled the check for that line. ``"LiouScope is
   production-ready and there are no open gaps."`` passed, because ``"no "``
   occurs somewhere in it.
2. ``paper/paper.md`` -- the JOSS submission -- was outside the scanned scope
   entirely, so ``"production-ready and externally certified"`` passed there.

A gate that only ever runs green proves nothing about its ability to go red
(007acc2 Evidence Binding Contract v1 section 2, "Discrimination Evidence").
Every case below is therefore a *mutation*: the repository is copied, a defect
is injected, and the gate MUST fail. The positive control asserts the pristine
tree passes, so a gate that failed unconditionally could not satisfy this file
either.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = Path(".github/scripts/check_claim_safety.py")


def _run_gate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / SCRIPT_REL)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A throwaway copy of the tree the gate reads, so mutations never touch it.

    The gate resolves its own paths from ``__file__``, so copying the script
    together with the scanned files is enough; no source or test code is needed.
    """
    root = tmp_path / "repo"
    (root / ".github" / "scripts").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / SCRIPT_REL, root / SCRIPT_REL)
    for rel in ("README.md", "CHANGELOG.md", "AGENTS.md", "CONTRIBUTING.md",
                "SECURITY.md", "CITATION.cff"):
        source = REPO_ROOT / rel
        if source.exists():
            shutil.copy2(source, root / rel)
    for rel in ("docs", "paper"):
        source = REPO_ROOT / rel
        if source.exists():
            shutil.copytree(source, root / rel)
    return root


def test_positive_control_pristine_tree_passes(sandbox: Path) -> None:
    """The unmutated tree must pass, else every red below is uninformative."""
    result = _run_gate(sandbox)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("target", "claim"),
    [
        # The plain case the gate already caught before this file existed.
        ("docs/index.md", "LiouScope is production-ready."),
        # REGRESSION (2026-09-02): an unrelated negation on the same line used
        # to disable the check. This is the load-bearing case.
        (
            "docs/index.md",
            "LiouScope is production-ready and there are no open gaps.",
        ),
        # REGRESSION (2026-09-02): the JOSS paper was out of scope.
        ("paper/paper.md", "LiouScope is production-ready and externally certified."),
        # Scope extensions that had never been guarded.
        ("AGENTS.md", "The pipeline is clinically validated."),
        ("SECURITY.md", "This project is externally certified."),
        # Markdown emphasis and the spaced spelling must not evade the pattern.
        ("README.md", "LiouScope is **production ready**."),
        # A trailing negation about something else is not a qualification of
        # the claim that precedes it.
        (
            "docs/index.md",
            "LiouScope is production-ready, though the logo is not final.",
        ),
    ],
)
def test_negative_control_gate_goes_red(sandbox: Path, target: str, claim: str) -> None:
    path = sandbox / target
    path.write_text(path.read_text(encoding="utf-8") + f"\n\n{claim}\n", encoding="utf-8")
    result = _run_gate(sandbox)
    assert result.returncode == 1, (
        f"gate stayed green on an injected risky claim in {target}: {claim!r}\n"
        f"stdout={result.stdout!r}"
    )
    # The gate reports the path with the platform separator (on Windows
    # `docs\index.md`); the parametrisation spells it portably with `/`.
    # Without normalisation this assertion is always false on Windows -- and
    # precisely for the cases WITH a directory, while the root-level files
    # stay green. A negative control that only fires on one platform guards
    # the guard only there.
    assert target in result.stderr.replace("\\", "/")


@pytest.mark.parametrize(
    "sentence",
    [
        # A negation that genuinely governs the claim must still be accepted,
        # otherwise the gate becomes unusable and gets disabled by hand.
        "LiouScope is not production-ready.",
        "LiouScope is not yet production-ready.",
        "This release is UNVERIFIED and must not be called production-ready.",
        "The DOI is a placeholder until the archive gate is released.",
    ],
)
def test_qualified_claims_are_still_accepted(sandbox: Path, sentence: str) -> None:
    path = sandbox / "docs/index.md"
    path.write_text(path.read_text(encoding="utf-8") + f"\n\n{sentence}\n", encoding="utf-8")
    result = _run_gate(sandbox)
    assert result.returncode == 0, (
        f"gate went red on a properly qualified statement: {sentence!r}\n"
        f"stderr={result.stderr!r}"
    )
