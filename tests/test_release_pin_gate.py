"""The release hash-pin gate, run the way the quality contract runs it.

``.github/scripts/check_release_pins.py`` keeps the release toolchain hash
locked (code-scanning alerts #14/#15). Its adversarial unit suite lives beside
it and runs in the quality contract; this module puts the gate's VERDICT -- its
exit code, which is what the CI job acts on -- into ``pytest -q`` as well, so a
contributor sees a broken release path before pushing rather than after.

As in ``test_workflow_hardening_gate.py``, every rejection is paired with a
positive control on a tree that differs from it in exactly one file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / ".github" / "scripts"
GATE = SCRIPTS / "check_release_pins.py"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="the gate reads pyproject.toml with tomllib (3.11+); CI runs it on 3.12",
)

# The two install lines code scanning flagged, verbatim from main before this
# gate existed.
UNPINNED_INSTALLS = (
    "        run: |\n"
    "          python -m pip install --upgrade pip\n"
    "          pip install build twine check-wheel-contents\n"
)


def _tree(tmp_path: Path) -> Path:
    """A copy of everything the gate reads, with the gate itself inside it."""
    root = tmp_path / "repo"
    (root / ".github" / "scripts").mkdir(parents=True)
    shutil.copytree(REPO / ".github" / "workflows", root / ".github" / "workflows")
    shutil.copytree(REPO / ".github" / "requirements", root / ".github" / "requirements")
    shutil.copy(REPO / "pyproject.toml", root / "pyproject.toml")
    shutil.copy(GATE, root / ".github" / "scripts" / GATE.name)
    return root


def _run(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )


def test_repository_release_path_is_hash_pinned() -> None:
    result = _run(GATE)
    assert result.returncode == 0, result.stderr


def test_copied_tree_is_accepted(tmp_path: Path) -> None:
    """Positive control for the rejections below: the copy itself is clean."""
    root = _tree(tmp_path)
    result = _run(root / ".github" / "scripts" / GATE.name)
    assert result.returncode == 0, result.stderr


def test_previous_unpinned_release_installs_are_rejected(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    workflow = root / ".github" / "workflows" / "pypi.yml"
    text = workflow.read_text(encoding="utf-8")
    start = text.index("        run: |\n          python -m pip install --require-hashes")
    end = text.index("\n", text.index("--require-hashes", start)) + 1
    workflow.write_text(text[:start] + UNPINNED_INSTALLS + text[end:], encoding="utf-8")
    result = _run(root / ".github" / "scripts" / GATE.name)
    assert result.returncode == 1
    assert result.stderr.count("not hash-pinned") == 2, result.stderr


def test_lock_below_the_build_backend_floor_is_rejected(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert '"setuptools>=77.0"' in text
    pyproject.write_text(text.replace('"setuptools>=77.0"', '"setuptools>=999"'), encoding="utf-8")
    result = _run(root / ".github" / "scripts" / GATE.name)
    assert result.returncode == 1
    assert "violates pyproject.toml build-system requirement" in result.stderr


@pytest.mark.parametrize("suite", ["test_release_pins.py", "test_compare_dists.py"])
def test_adversarial_unit_suites_pass(suite: str) -> None:
    result = _run(SCRIPTS / suite)
    assert result.returncode == 0, result.stderr
