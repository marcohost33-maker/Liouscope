# LiouScope governance audit — 2026-09-01

Repository: `marcohost33-maker/Liouscope`  
Observed `main`: `386041b4072a776a985beb34af5815fb344c74f9`

## Scope

This is a point-in-time repository-governance evidence lock. It does not amend the historical scientific or packaging claims of `RELEASE_AUDIT_v0.5.0.md` and does not certify settings that the available integration cannot read.

## Required checks observed on `main`

GitHub branch metadata reports `main` as protected, enforcement level `everyone`, with these required status checks:

- `test (ubuntu-latest, 3.10)`
- `test (ubuntu-latest, 3.11)`
- `test (ubuntu-latest, 3.12)`
- `test (ubuntu-latest, 3.13)`
- `test (ubuntu-latest, 3.14)`
- `qutip-cross-check (3.11)`
- `qutip-cross-check (3.12)`
- `quality contract`

## Negative-control proof

Draft PR #141 intentionally adds one inert `workflow_dispatch`-only workflow containing a mutable `actions/checkout@main` reference. The `Quality Contract` workflow completed with `failure`; its `quality contract` job failed specifically at `Check workflow hardening`.

This demonstrates that the aggregate required check receives and propagates a repository-policy failure. The control workflow is deliberately non-mergeable test material and must be removed after the evidence is captured.

## Positive / non-workflow proof

This audit and the accompanying `QUALITY_WORKFLOW_OS.md` update are documentation-only. Their PR is used to verify that the required `quality contract` is emitted for a non-workflow change and reaches a terminal result rather than remaining indefinitely `Expected`.

## Explicitly unverified protection settings

The integration can read the branch summary but receives HTTP 403 for the detailed branch-protection endpoint. Therefore this audit does not claim independently verified values for:

- required approving-review count;
- force-push allowance;
- branch deletion allowance;
- administrator bypass;
- linear-history or signed-commit requirements;
- conversation-resolution requirements.

These settings remain a repository-settings verification item if a release or governance claim depends on them.

## Verdict

`quality contract` is observed as a required status check and its fail-closed path is load-bearing. Issue #133 can be closed once the documentation-only positive-control PR is green and the negative-control PR has been closed/reset without merging its deliberate violation.
