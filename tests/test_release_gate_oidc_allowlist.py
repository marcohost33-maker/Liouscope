"""OIDC allowlist of the release gate (LiouScope #184, Claude cross-check 2026-09-30).

Publisher detection on shell text cannot be complete (curl to the upload API,
wrapper scripts, eval). The gate therefore also pins WHICH jobs may request an
OIDC token; PyPI's Trusted Publisher binding (workflow file + environment
``pypi`` + its deployment rules) is the credential boundary itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))
import check_release_pins as gate

ROOT = Path(__file__).resolve().parents[1]


def _wf(job: str) -> str:
    return "name: x\non: push\npermissions: {}\njobs:\n" + job


def test_repository_workflows_pass() -> None:
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        assert gate.oidc_violations(path.name, path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "job",
    [
        "  sneak:\n    runs-on: u\n    permissions:\n      id-token: write\n    steps:\n      - run: ./release.sh\n",
        "  sneak:\n    runs-on: u\n    permissions: write-all\n    steps:\n      - run: echo\n",
    ],
)
def test_unlisted_job_with_oidc_fails(job: str) -> None:
    errors = gate.oidc_violations("other.yml", _wf(job))
    assert any("not in OIDC_ALLOWLIST" in e for e in errors)


def test_workflow_level_grant_fails() -> None:
    text = "name: x\non: push\npermissions:\n  id-token: write\njobs:\n  a:\n    runs-on: u\n    steps:\n      - run: echo\n"
    assert any("workflow-level" in e for e in gate.oidc_violations("other.yml", text))


def test_unreadable_permissions_fail_closed() -> None:
    job = "  a:\n    runs-on: u\n    permissions: ${{ fromJSON(vars.P) }}\n    steps:\n      - run: echo\n"
    errors = gate.oidc_violations("other.yml", _wf(job))
    assert errors, "an expression-valued permissions node must not pass silently"


def test_publish_job_without_environment_fails() -> None:
    job = "  publish:\n    runs-on: u\n    permissions:\n      id-token: write\n    steps:\n      - run: echo\n"
    errors = gate.oidc_violations("pypi.yml", _wf(job))
    assert any("without environment 'pypi'" in e for e in errors)


def test_publish_job_with_other_environment_fails() -> None:
    job = (
        "  publish:\n    runs-on: u\n    environment: staging\n"
        "    permissions:\n      id-token: write\n    steps:\n      - run: echo\n"
    )
    assert gate.oidc_violations("pypi.yml", _wf(job))


def test_publish_secret_reference_fails() -> None:
    job = "  a:\n    runs-on: u\n    steps:\n      - run: twine upload -p ${{ secrets.PYPI_API_TOKEN }} dist/*\n"
    assert any(
        "publish credential secret" in e for e in gate.oidc_violations("other.yml", _wf(job))
    )


def test_read_and_none_grants_pass() -> None:
    job = "  a:\n    runs-on: u\n    permissions:\n      id-token: none\n      contents: read\n    steps:\n      - run: echo\n"
    assert gate.oidc_violations("other.yml", _wf(job)) == []


def test_check_wires_the_allowlist(tmp_path: Path) -> None:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "pypi.yml").write_text(
        (ROOT / ".github/workflows/pypi.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (wf / "evil.yml").write_text(
        _wf(
            "  x:\n    runs-on: u\n    permissions:\n      id-token: write\n    steps:\n      - run: curl -F content=@d https://upload.pypi.org/legacy/\n"
        ),
        encoding="utf-8",
    )
    errors = gate.check(tmp_path)
    assert any("evil.yml" in e and "OIDC_ALLOWLIST" in e for e in errors)


def _workflow(run: str) -> str:
    return f"jobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: {run}\n"


@pytest.mark.parametrize(
    "run",
    [
        "bash -c 'twine --verbose upload dist/*'",
        "sh -ec 'twine --verbose upload dist/*'",
        "bash -o pipefail -c 'cd dist && twine --verbose upload *'",
        "/bin/bash --noprofile -c 'twine --verbose upload dist/*'",
        "eval 'twine --verbose upload dist/*'",
        "bash -c \"sh -c 'twine --verbose upload dist/*'\"",
        "bash -eo pipefail -c 'twine --verbose upload dist/*'",
        "bash --noprofile --norc -eo pipefail -c 'twine --verbose upload dist/*'",
        "bash -c -- 'twine --verbose upload dist/*'",
        "bash --rcfile /dev/null -c 'twine --verbose upload dist/*'",
        "bash --init-file /dev/null -c 'twine --verbose upload dist/*'",
        "bash -c -o pipefail 'twine --verbose upload dist/*'",
    ],
)
def test_nested_shell_publisher_is_evidence(run: str) -> None:
    """Issue #184: the publisher inside a nested script is still seen."""
    evidence, unreadable = gate.publish_evidence(_workflow(run))
    assert unreadable is None
    assert any("twine" in item and "upload" in item for item in evidence), evidence


def test_nested_shell_without_publisher_is_clean() -> None:
    evidence, unreadable = gate.publish_evidence(_workflow("bash -c 'twine check dist/*'"))
    assert (evidence, unreadable) == ([], None)


def test_untokenisable_nested_script_counts_as_publishing() -> None:
    evidence, _ = gate.publish_evidence(_workflow("bash -c \"twine --verbose upload 'dist\""))
    assert evidence


def test_shell_c_without_script_counts_as_publishing() -> None:
    evidence, _ = gate.publish_evidence(_workflow("bash -c"))
    assert evidence == []  # no tool/verb on the line: pre-filter skips it
    evidence, _ = gate.publish_evidence(_workflow("twine --verbose upload; bash -eo pipefail -c"))
    assert any("cannot tokenise" in item for item in evidence), evidence


def test_shell_script_file_is_not_a_nested_script() -> None:
    evidence, unreadable = gate.publish_evidence(_workflow("bash scripts/upload_docs.sh"))
    assert (evidence, unreadable) == ([], None)


@pytest.mark.parametrize(
    "secret",
    [
        "${{ secrets['PYPI_API_TOKEN'] }}",
        '${{ secrets["TWINE_PASSWORD"] }}',
        "${{ toJSON(secrets) }}",
    ],
)
def test_publish_secret_access_forms_fail(secret: str) -> None:
    text = f"jobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo {secret}\n"
    assert gate.oidc_violations("other.yml", text)


def test_secrets_inherit_fails() -> None:
    text = "jobs:\n  x:\n    uses: ./.github/workflows/ci.yml\n    secrets: inherit\n"
    assert any("secrets: inherit" in e for e in gate.oidc_violations("other.yml", text))


def test_id_token_key_is_case_insensitive() -> None:
    text = "jobs:\n  x:\n    permissions:\n      ID-Token: write\n    runs-on: ubuntu-latest\n"
    assert any("OIDC_ALLOWLIST" in e for e in gate.oidc_violations("other.yml", text))


def test_allowlisted_job_with_write_all_fails() -> None:
    text = "jobs:\n  analysis:\n    permissions: write-all\n    runs-on: ubuntu-latest\n"
    assert any("explicitly" in e for e in gate.oidc_violations("scorecard.yml", text))
