"""Static safety contract for the PyPI trusted-publishing workflow."""

from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pypi.yml"


def test_manual_dispatch_is_build_only_without_oidc_permission():
    text = _WORKFLOW.read_text(encoding="utf-8")
    build_block, publish_block = text.split("\n  publish:\n", maxsplit=1)
    assert "workflow_dispatch:" in build_block
    assert "id-token: write" not in build_block
    assert "Manual dry-run verified" in build_block
    assert "id-token: write" in publish_block


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
