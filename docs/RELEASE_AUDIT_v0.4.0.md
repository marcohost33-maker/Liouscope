# LiouScope Release Audit v0.4.0

Date: 2026-06-08
Runtime release: v0.4.0
Release date in `CITATION.cff`: 2026-06-07
Repository: `marcohost33-maker/Liouscope`
Scope: release metadata, reproducibility, audit trail, canon/runtime alignment.

## 1. Purpose

This file records the release-audit status for v0.4.0 after a constructive
cross-check of the previous release notes, citation metadata, repo state, and
open PR state. It is intentionally documentation-only: no physics, numerics,
public API, manifest schema, diagnostic schema, or taxonomy behaviour is changed
by this audit file.

## 2. Validated current state

- Runtime single source: `src/liouscope/_version.py` reports `0.4.0`.
- Main release PR: PR #42 merged v0.4.0.
- Current code frontier: PR #43 is open/draft and targets a resolvent large-matrix
  correctness fix plus regression coverage.
- Scientific canon remains `A1-A12-v3.1` and `D1-D24` unless explicitly bumped.
- `MANIFEST_SCHEMA.json` remains schema version `1.2.0`.

## 3. Falsified or corrected release assumptions

| Finding | Severity | Resolution |
|---|---:|---|
| `CITATION.cff` top-level version was `0.4.0`, but `preferred-citation.title` still embedded `v0.2.0`. | P2 | Fixed on the PR branch by removing the stale embedded version from the title. |
| `CITATION.cff` abstract still said "twenty diagnostics (D1-D20)" while the repo canon is D1-D24. | P2 | Fixed on the PR branch: abstract now says 24 diagnostics, with D1-D20 as original submission set and D21-D24 as post-submission additions. |
| v0.4.0 changelog release-scope wording used a precise commit/PR count that was flagged by review as not fully accurate. | P2 | Do not treat the old sentence as a complete audit inventory; this file is the corrective release-audit pointer. |
| PR #43 is mergeable but still draft and lacks confirmed required CI status in the PR body. | P0/P1 | Not merge-ready until draft is lifted and CI/QuTiP gates are green. |
| Direct pushes to `main` are blocked by branch protection and required checks. | GOOD | This is desired behaviour after the prior branch-delete incident. |

## 4. Current release verdict

| Dimension | Verdict | Notes |
|---|---|---|
| Engineering release | PASS with documentation debt now being closed | v0.4.0 is merged, but public/citable release needs final metadata and archive proof. |
| Scientific claim safety | PASS for existing scope | No new scientific claims introduced by this audit. |
| PR #43 merge readiness | BLOCKED | Draft + required CI status not confirmed. |
| Public/citable release | OPEN | Needs release metadata, DOI/Zenodo/SWHID/PyPI provenance proof as applicable. |
| Canon/runtime alignment | PARTIAL | Repo runtime is v0.4.0; Drive-side 0Liou+ canon must be updated separately to avoid package-canon drift. |

## 5. Required gates before treating v0.4.x as public/citable final

```text
[CODE]
[ ] PR #43 ready-for-review, no longer draft
[ ] Required CI matrix 3.10-3.13 green
[ ] QuTiP cross-checks green and non-vacuous
[ ] pytest -q green
[ ] tests/test_anchors.py green
[ ] ruff clean
[ ] mypy clean
[ ] build + twine check green

[METADATA]
[x] CITATION top-level version is v0.4.0
[x] preferred-citation no longer embeds stale v0.2.0
[x] citation abstract says D1-D24 rather than D1-D20-only
[ ] changelog release-scope wording either corrected or linked to an exact audit inventory
[ ] Canon status updated for repo runtime v0.4.0

[ARCHIVE]
[ ] GitHub release notes final
[ ] PyPI release/provenance path decided
[ ] Zenodo DOI metadata prepared, if public archival release is intended
[ ] SWHID captured after public source availability, if applicable
```

## 6. Non-goals

- No merge of PR #43 from this audit.
- No version bump from v0.4.0.
- No change to `MANIFEST_SCHEMA.json`.
- No change to `TAXONOMY_VERSION`.
- No change to D24 Zhou formula/status beyond already-documented claim wording.

## 7. Recommended next action

Finish PR #43 only after CI and review gates pass. Then create a dedicated
v0.4.1 or v0.4.1rc1 release-candidate PR that includes:

1. the PR #43 resolvent correction,
2. this release metadata cleanup,
3. a final changelog section,
4. build/wheel/sdist checks,
5. archive/provenance records.
