"""Static safety contract for the PyPI trusted-publishing workflow."""

from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pypi.yml"


def test_manual_dispatch_is_build_only_without_oidc_permission():
    text = _WORKFLOW.read_text(encoding="utf-8")
    build_block, publish_block = text.split("\n  publish:\n", maxsplit=1)
    assert "workflow_dispatch:" in build_block
    assert "id-token: write" not in build_block
    assert "Dry-run ($GITHUB_EVENT_NAME) verified" in build_block
    assert "no publish credentials exist in this job" in build_block
    assert "id-token: write" in publish_block


def test_dry_run_triggers_reach_the_build_job_only():
    """Pull requests and pushes to main exercise the release build, never publish.

    The build job carries no OIDC permission and no deployment environment, and
    only the release-gated publish job has either; the concurrency group cancels
    superseded pull-request dry-runs but never a release run.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    build_block, publish_block = text.split("\n  publish:\n", maxsplit=1)
    on_block = text.split("\npermissions:", maxsplit=1)[0]
    assert "\n  pull_request:\n" in on_block
    assert "\n  push:\n    branches: [main]\n" in on_block
    assert "environment:" not in build_block
    assert "environment:" in publish_block
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in text


def test_reproducibility_is_verified_before_qa_and_upload():
    text = _WORKFLOW.read_text(encoding="utf-8")
    build_block, _ = text.split("\n  publish:\n", maxsplit=1)
    assert 'SOURCE_DATE_EPOCH="$(git log -1 --format=%ct HEAD)"' in build_block
    assert "python -m build --no-isolation --outdir dist" in build_block
    assert 'python .github/scripts/compare_dists.py dist "$RUNNER_TEMP/rebuild"' in build_block
    step = build_block.index("Build sdist + wheel twice and verify reproducibility")
    assert step < build_block.index("Release QA gate")
    assert step < build_block.index("Upload verified distributions")


def test_publish_job_is_release_only_and_feature_gated():
    text = _WORKFLOW.read_text(encoding="utf-8")
    _, publish_block = text.split("\n  publish:\n", maxsplit=1)
    assert "github.event_name == 'release'" in publish_block
    assert "github.event.action == 'published'" in publish_block
    assert "PYPI_PUBLISH_ENABLED" in publish_block
    assert "ref: ${{ github.event.release.tag_name }}" in publish_block


def test_identity_is_checked_before_artifact_handoff_and_upload():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert text.count("test \"$SOURCE_VERSION\" = \"$DIST_VERSION\"") == 2
    assert text.count("test \"$HEAD_SHA\" = \"$TAG_SHA\"") == 2
    assert text.count("test \"$EVENT_SHA\" = \"$TAG_SHA\"") == 2
    assert text.index("Verify source, artifact, tag and commit identity") < text.index(
        "Upload verified distributions"
    )
    assert text.index("Re-verify release identity before upload") < text.index(
        "Publish via Trusted Publishing"
    )
