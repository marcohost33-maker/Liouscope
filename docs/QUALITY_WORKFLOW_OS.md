# LiouScope Quality Workflow OS v1

**Status:** production-readiness contract for CI, release evidence, and claim safety.  
**Scope:** repository governance and GitHub workflows only; this document does not certify scientific correctness, PyPI availability, DOI archival status, or production readiness.  
**Source of truth:** code, workflows, release artifacts, tags, and release audit files beat README prose.

## 1. Why this exists

LiouScope already has strong engineering posture: pinned GitHub Actions, a Python CI matrix, workflow security audit, Scorecard posture scanning, guarded PyPI publish, and explicit research-use disclaimers. The remaining quality risk is **drift**:

- README/status text can overstate what the repository has actually released.
- A workflow can stay green while no longer protecting the right failure mode.
- A security scanner score can improve while false-pass or rework risk remains hidden.
- A publish workflow can exist before PyPI Trusted Publishing is actually enabled.

This OS turns those risks into small, repeatable gates.

## 2. Non-negotiable gates

### G1 — Minimal token permissions

Every workflow must set top-level `permissions: {}` or `permissions: contents: read`, then raise permissions only at job level when needed.

Allowed examples:

```yaml
permissions: {}
```

```yaml
permissions:
  contents: read
```

Job-level exceptions must explain the reason in a comment.

### G2 — Full-SHA action pinning

Every third-party `uses:` reference must pin to a full 40-character commit SHA.

Allowed:

```yaml
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
```

Not allowed:

```yaml
uses: actions/checkout@v4
```

Local reusable workflows such as `./.github/workflows/ci-python-local.yml` are allowed.

### G3 — No unsafe privileged PR trigger

`pull_request_target` is disallowed unless a workflow contains a specific reviewed exception marker:

```yaml
# ALLOW_PULL_REQUEST_TARGET: <why privileged context is required and how untrusted checkout is prevented>
```

The default solution is to use `pull_request`, `workflow_dispatch`, `release`, or `workflow_run` with clear artifact trust boundaries.

### G4 — Claim-safety

Documentation must not claim any of the following unless a release/evidence file explicitly supports it:

- not evidence-backed production-ready status
- not evidence-backed clinical or operational validation
- not evidence-backed PyPI-published status
- not evidence-backed DOI/Zenodo archival status
- not evidence-backed external certification
- not evidence-backed release-complete status

Negative disclaimers are allowed and encouraged, for example: “not for diagnostic or operational use”.

### G5 — Release evidence lock

A release is not considered complete until the release audit contains the actual evidence:

- tag and commit SHA
- sdist/wheel hashes
- workflow run URL
- PyPI status if applicable
- artifact attestation/SBOM status if applicable
- changelog/version reconciliation
- explicit unresolved gaps

### G6 — Required-check safety

A workflow may be branch-protection-required only if it runs on every relevant pull request. Path-filtered workflows must not be required unless the branch protection rule is also path-scoped, which standard GitHub branch protection is not.

### G7 — Paired metrics

Quality claims must include at least one success metric and one counter-metric:

| Success metric | Counter-metric |
|---|---|
| CI pass rate | false-pass rate |
| lead time | rework rate |
| Scorecard score | unresolved high-risk finding count |
| release frequency | failed release recovery time |
| coverage | escaped defect rate |

## 3. Repository Quality Delta Score (RQDS v0.2)

`RQDS = 0.20*CI_Reliability + 0.20*Security_Posture + 0.15*Evidence_Coverage + 0.15*Maintainability + 0.15*Delivery_Stability + 0.10*Observability + 0.05*Cost_Discipline - Penalty`

### Component definitions

- **CI_Reliability:** required checks run deterministically on relevant PRs.
- **Security_Posture:** pinned actions, minimal permissions, safe triggers, dependency updates, scanner signal.
- **Evidence_Coverage:** hard claims have linked source-of-truth evidence.
- **Maintainability:** workflow logic is simple, reusable, documented, and locally understandable.
- **Delivery_Stability:** release/publish path is guarded and recoverable.
- **Observability:** failures leave enough logs/artifacts to diagnose cause without guessing.
- **Cost_Discipline:** fast PR gates and deep nightly/release gates are separated.
- **Penalty:** unmarked uncertainty, false release claim, unpinned action, privileged trigger, missing release evidence, or branch-protection trap.

RQDS is advisory, not a certification. A high RQDS never overrides a concrete blocker.

## 4. RED-GREEN-EVIDENCE-LOCK

1. **RED:** State what could be false.
2. **GREEN:** Run or inspect the relevant repo/workflow evidence.
3. **EVIDENCE:** Link the source of truth.
4. **LOCK:** Only then update release/status wording.
5. **LEARN:** Add a negative result if the failure class is reusable.

## 5. PR acceptance checklist

A PR touching workflows, docs, release, packaging, or claims must answer:

- [ ] Did all third-party actions remain full-SHA pinned?
- [ ] Did token permissions remain minimal?
- [ ] Did this avoid unsafe `pull_request_target` use?
- [ ] Are new claims backed by code, tag, release audit, CI run, PyPI, Zenodo/DOI, or explicit `UNVERIFIED` status?
- [ ] If release-related: are hashes, tag, changelog, and unresolved gaps captured?
- [ ] If branch-protection-related: can the required checks run on non-workflow PRs?
- [ ] If a scanner score improved: was false-pass/rework risk also checked?

## 6. What this document does not do

- This document does **not** assert that LiouScope is production-ready.
- This document does **not** assert that LiouScope is externally certified.
- This document does **not** assert that PyPI or DOI publication is complete.
- This document does **not** replace scientific validation.
- This document does **not** replace human release approval.

It defines the repo’s quality contract so those statuses can be reached without drifting into unsupported claims.
