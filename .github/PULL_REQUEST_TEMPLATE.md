<!-- Thank you for the contribution. Please fill in this template. -->

## Summary

<!-- 1-2 sentences: what changes and why. -->

## Scope

- [ ] Bug fix (no new feature)
- [ ] New diagnostic or layer (please link the motivating reference)
- [ ] Performance / sparse-path / numerics improvement
- [ ] Docs / examples / packaging only
- [ ] CI / workflow / security
- [ ] Release / packaging / publish evidence

## Verification

- [ ] `pytest -q` passes locally, or this PR is docs/workflow-only and CI is the authority
- [ ] `tests/test_anchors.py` unchanged, or anchor change is documented in `CHANGELOG.md`
- [ ] If touching `MANIFEST_SCHEMA.json`: bumped `schema_version` and noted migration
- [ ] If new external dependency: added to `pyproject.toml` and reviewed against minimal-deps policy
- [ ] If touching `.github/workflows/`: actions remain full-SHA pinned and token permissions remain minimal
- [ ] If touching claims/docs/release wording: new status claims are backed by release audit, tag, CI run, PyPI/DOI evidence, or marked `UNVERIFIED`/not complete
- [ ] If touching branch protection / required checks: the required workflow runs on every relevant PR and is not path-filtered into a stuck-check trap

## Quality contract

- [ ] This PR does not introduce unsupported production-ready, externally certified, PyPI-published, DOI/Zenodo-archived, or release-complete wording
- [ ] This PR preserves the research / pre-clinical disclaimer unless a separate evidence-locked release audit proves a status change
- [ ] False-pass and rework risk considered, not only green CI

## Reproducibility note

<!-- If the PR affects numerical results, state the seed used and which V1-V5 system was tested. -->

## Linked issues

Closes #
