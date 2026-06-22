# LiouScope Release Audit v0.4.1

Date: 2026-06-19
Repository: `marcohost33-maker/Liouscope`
Runtime release: v0.4.1
Scope: public/citable-release readiness, archive/provenance gates, and post-v0.4.1 CI hardening.

## 1. Purpose

This audit updates the older v0.4.0 release audit after the repo advanced to
v0.4.1 and after PRs #43-#48 merged. It is documentation-only: it does not
change physics, numerics, diagnostic outputs, manifest schema, taxonomy, or the
runtime API.

## 2. Current validated state

| Area | Current state |
|---|---|
| Runtime version | `src/liouscope/_version.py` reports `0.4.1`. |
| Citation metadata | `CITATION.cff` reports `version: 0.4.1` and `date-released: 2026-06-16`. |
| Diagnostic schema | D1-D24, with D1-D20 as the original submission set and D21-D24 as post-submission additions. |
| Taxonomy | `A1-A12-v3.1`. |
| Manifest contract | `MANIFEST_SCHEMA_VERSION == 1.2.0`. |
| v0.4.1 scope | PRs #43-#45: resolvent correctness/hardening, classifier taxonomy docs, and coverage lifts. |
| Post-v0.4.1 CI hardening | PR #48: Python 3.14 added to the CI matrix and project classifiers. |
| Public/citable archival release | Still open until GitHub release, package publication/provenance, DOI/SWHID, and evidence locks are completed. |

## 3. Best-practice basis

This file is aligned with the following release-engineering rules:

- Changelogs and GitHub releases should be curated for humans, not treated as raw commit logs.
- GitHub release notes may be generated from merged PRs, but they must be reviewed so they include only the intended scope.
- `CITATION.cff` is the machine-readable citation source for the repository and must stay synchronized with the released software version.
- If PyPI publication is used, Trusted Publishing / OIDC is preferred over long-lived API tokens.
- Source-code archival identifiers such as SWHIDs are public-source provenance artefacts; they are applicable only after the relevant source snapshot is publicly archivable.

## 4. v0.4.1 release verdict

| Dimension | Verdict | Rationale |
|---|---|---|
| Engineering release | GREEN | v0.4.1 has version/citation/changelog alignment, and the merged PR trail records the corrected numerics and test hardening. |
| Scientific claim safety | GREEN for existing scope | No new scientific claim is introduced by this audit; D24 remains a related/coarser Zhou-family surrogate, not an exact Eq.(16) claim. |
| CI / QuTiP status | GREEN for observed PR heads | The `ci.yml` test matrix + QuTiP cross-checks ran and passed on the observed heads of both PR #43 and PR #48. |
| Security-workflow status | GREEN for PR #48 only (scope-limited) | The Workflow Security Audit (`zizmor.yml`) is path-filtered to `.github/workflows/**` + `.github/dependabot.yml` on `pull_request`; PR #43 changed no `.github/` files, so zizmor did **not** run on #43. PR #48 changed `.github/workflows/ci.yml`, so zizmor ran and passed on #48. OpenSSF Scorecard (`scorecard.yml`) has no `pull_request` trigger at all (only `workflow_dispatch` / `branch_protection_rule` / `schedule` / `push:main`), so it ran on neither PR head. Security evidence is therefore valid for #48, not #43. |
| Python version support | GREEN / needs protection promotion | Python 3.14 is in the test matrix and classifiers; it should become a required check only after stable green history. |
| Citation metadata | GREEN | `CITATION.cff` is synchronized to v0.4.1 and D1-D24. |
| Public/citable release | YELLOW / OPEN | Requires GitHub release, PyPI/TestPyPI or explicit no-PyPI decision, Zenodo/DOI decision, SWHID decision, checksums, and evidence lock. |
| Drive-side canon | YELLOW / OPEN | 0Liou+ must be updated from older package canon to repo runtime v0.4.1 while keeping the scientific report v2.0 FINAL as baseline. |

## 5. Required public/citable-release checklist

> **Status update (2026-06-22):** the package/provenance and CI-on-tag gates
> below are now discharged with recorded evidence in
> `docs/EVIDENCE_LOCK_v0.4.1.md` (build hashes, `twine check` PASS, required
> checks GREEN on tag commit `1965f2b`, intrinsic SWHIDs, PyPI/Zenodo/SWHID
> ADRs). The GitHub Release for v0.4.1 already exists (published 2026-06-17);
> the only remaining open item is the external 0Liou+ Drive canon update. The
> checklist below is retained as the original point-in-time plan.

```text
[REPO]
[x] Runtime version single-source is v0.4.1
[x] CITATION.cff version/date match v0.4.1
[x] Changelog has [0.4.1] section
[x] Canon status document states runtime v0.4.1
[x] Git tag v0.4.1 exists (at commit 1965f2b) and points to the intended release commit
[ ] GitHub Release v0.4.1 exists with reviewed release notes
[ ] GitHub Release notes link to CHANGELOG.md and this audit file
    NOTE (tag scope): this audit file is POST-tag accompanying documentation. It
    is NOT part of the tagged v0.4.1 source snapshot (the v0.4.1 tag at 1965f2b
    predates this file) and must NOT be tagged into it — re-tagging the audit
    commit would silently pull Unreleased changes (e.g. #47/#48) into the v0.4.1
    source archive. The GitHub Release object (not the git tag's source tree) is
    what links to CHANGELOG.md and this audit; the linkage lives in the release
    notes, leaving the v0.4.1 source archive unchanged.

[CI / QUALITY]
[x] CI matrix success observed for PR #48 head
[x] QuTiP cross-check success observed for PR #48 head
[x] Encoding Guard success observed for PR #48 head
[x] Workflow Security Audit (zizmor) success observed for PR #48 head (#48 changed .github/, so zizmor ran; it did NOT run on #43)
[ ] Required checks confirmed GREEN on the INTENDED v0.4.1 tag commit (1965f2b),
    not merely on the #48 PR head — OPEN until re-verified against the tag commit
[ ] Python 3.14 check promoted to required after stable green history
[ ] Required-check list documented in branch protection / ruleset notes

[PACKAGE / PROVENANCE]
[ ] Build wheel and sdist from clean tag
[ ] `twine check dist/*` passes
[ ] SHA256 checksums recorded for wheel and sdist
[ ] PyPI Trusted Publishing configured, or explicit no-PyPI decision recorded
[ ] If PyPI is used: publish via OIDC/Trusted Publisher, not long-lived token
[ ] If PyPI attestations are available: record attestation reference

[ARCHIVE / CITATION]
[ ] Zenodo DOI prepared or explicit no-Zenodo decision recorded
[ ] SWHID captured after public source availability, or explicit private-repo limitation recorded
[ ] `CITATION.cff` includes DOI once DOI exists
[ ] 0Liou+ Drive canon updated to repo runtime v0.4.1
[ ] Evidence lock register updated with release commit, tag, checksums, DOI/SWHID status
```

## 6. Public release notes draft

Use this as the reviewed GitHub release body for v0.4.1, adapting only the final
commit/tag/checksum values at publication time.

```markdown
# LiouScope v0.4.1

Engineering release: numerics correctness + production hardening.

## Highlights

- Fixed the large-matrix `resolvent_norm` adjoint solve in the SuperLU power-iteration path.
- Hardened resolvent convergence handling and large-Liouvillian scaling.
- Corrected classifier taxonomy documentation to match the authoritative A1-A12/F-family canon.
- Lifted coverage across fitting, Lindblad, sparse, classification, and numerics layers.
- Synchronized `CITATION.cff` to v0.4.1 and D1-D24.

## Compatibility

- PATCH release.
- No `diagnose()` / `DiagnosticReport` API break.
- No manifest schema change.
- No taxonomy version bump.
- No new scientific claim beyond the documented v0.4.1 scope.

## Validation

- See `CHANGELOG.md` `[0.4.1]`.
- See `docs/RELEASE_AUDIT_v0.4.1.md`.
- See PRs #43-#48 for CI and review history.
```

## 7. Risk register

| Risk | Severity | Mitigation |
|---|---:|---|
| Public release without DOI/SWHID/evidence locks | P1 | Do not label the release as citable-final until archive/provenance checklist is complete. |
| Python 3.14 job green but not required | P2 | Promote to required only after stable green history. |
| Drive canon lags repo runtime | P1 | Update 0Liou+ canon to v0.4.1 while preserving scientific report v2.0 FINAL as baseline. |
| PyPI token leakage | P1 | Use Trusted Publishing/OIDC; avoid long-lived API tokens. |
| SWHID requested while repo remains private | P2 | Record private-repo limitation; capture SWHID only after public source availability. |

## 8. Recommended next work unit

Create `LIOUSCOPE_PUBLIC_ARCHIVE_v0.4.1` with:

1. reviewed GitHub Release v0.4.1,
2. wheel/sdist build + hashes,
3. PyPI Trusted Publishing or explicit no-PyPI ADR,
4. Zenodo DOI or explicit no-Zenodo ADR,
5. SWHID capture or explicit private-source limitation,
6. 0Liou+ Drive canon update,
7. evidence lock register update.
