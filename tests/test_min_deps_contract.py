"""Issue #177: the declared dependency floors and the CI job that tests them stay one.

The ``min-deps`` job derives its pins from ``pyproject.toml`` through
``.github/scripts/min_deps_constraints.py``. These tests keep both halves of
that contract: the script turns every floor into an exact-series pin and
refuses a dependency without a floor, and the workflow really uses the script
on the lowest supported Python.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "min_deps_constraints.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _load():
    spec = importlib.util.spec_from_file_location("min_deps_constraints", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constraints_pin_every_declared_floor() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert block is not None
    floors = dict(re.findall(r'"([A-Za-z0-9_.-]+)>=([0-9.]+)"', block.group(1)))
    pins = _load().constraints()
    assert pins == [f"{name}=={version}.*" for name, version in floors.items()]
    assert {"numpy", "scipy"} <= {p.split("==")[0] for p in pins}


def test_a_dependency_without_a_floor_is_refused(tmp_path: Path) -> None:
    fake = tmp_path / "pyproject.toml"
    fake.write_text('[project]\ndependencies = [\n    "numpy>=2.0",\n    "scipy",\n]\n')
    with pytest.raises(SystemExit, match="no plain '>=' floor"):
        _load().constraints(fake)


def test_ci_runs_the_min_deps_job_from_the_script_on_the_lowest_python() -> None:
    workflow = CI.read_text(encoding="utf-8")
    job = workflow.split("  test-min-deps:", 1)[1].split("\n  test-head:", 1)[0]
    assert "python .github/scripts/min_deps_constraints.py" in job
    assert "--constraint min-deps-constraints.txt" in job
    requires = re.search(r'requires-python\s*=\s*">=([0-9.]+)"',
                         (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert requires is not None
    assert f'python-version: "{requires.group(1)}"' in job
    assert "pytest -q" in job and "tests/test_anchors.py" in job
