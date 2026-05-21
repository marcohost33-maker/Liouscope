<!-- Thank you for the contribution. Please fill in this template. -->

## Summary

<!-- 1-2 sentences: what changes and why. -->

## Scope

- [ ] Bug fix (no new feature)
- [ ] New diagnostic or layer (please link the motivating reference)
- [ ] Performance / sparse-path / numerics improvement
- [ ] Docs / examples / packaging only
- [ ] CI / workflow / security

## Verification

- [ ] `pytest -q` passes locally
- [ ] `tests/test_anchors.py` unchanged, or anchor change is documented in `CHANGELOG.md`
- [ ] If touching `MANIFEST_SCHEMA.json`: bumped `schema_version` and noted migration
- [ ] If new external dependency: added to `pyproject.toml` and reviewed against minimal-deps policy
- [ ] If touching `.github/workflows/`: actions remain SHA-pinned (regression-gate)

## Reproducibility note

<!-- If the PR affects numerical results, state the seed used and which V1-V5 system was tested. -->

## Linked issues

Closes #
