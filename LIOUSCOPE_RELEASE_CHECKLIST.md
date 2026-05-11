# LiouScope Release Checklist v1.3 (Drive-verified)

Aligned with `LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml` v1.2.0 and the upstream
Drive Evidence Pack v1.2 (Drive ID `19fS59kyGIA_RdiVC8cv6MFfKR30sJmD0`).

## P0 -- Evidence Lock

- [x] Canonical v2.0-FINAL Drive ID selected: `1N9ldvg8_taUL07xSnQbgdLpTbMLVHkL_`.
- [x] Canonical PDF SHA-256 computed locally:
      `bb72e449d11bb94874655ac3253a34dec4b90d4a83c5e869aa1a61a29d19a847`.
- [x] HANDOFF T11 Drive ID recorded.
- [x] `liouscope_integration.py` Drive ID recorded.
- [x] arXiv v5 bundle Drive ID recorded.
- [x] `1_META_CORE.md` Drive ID recorded.
- [x] `2_PROJECT_INDEX.md` Drive ID recorded.
- [x] `5_RESULTS_SURVIVAL_ITEMS.md` Drive ID recorded; SHA-256
      `2b939926c6cd95d02de424ba99c7cb91c5afaac4cde4412322b9f8e4abb6f6c1`.
- [ ] Duplicate v2.0-FINAL file `1QbIpRWd5pNTEXrzFiG_yPCyLCIN_U9uj`
      renamed or moved to archive **(action lives on Drive, not in repo)**.
- [ ] Raw SHA-256 computed for **all** canonical Drive artefacts; partial
      attestation in `LIOUSCOPE_DRIVE_ATTESTATION.csv`.

## P1 -- Repository

- [x] `LICENSE` (Apache-2.0).
- [x] `README.md`.
- [x] `CHANGELOG.md`.
- [x] `REPRODUCIBILITY.md`.
- [x] `CITATION.cff` with Zenodo-DOI placeholder.
- [x] `codemeta.json` with `runtimePlatform` + `issueTracker`.
- [x] `SECURITY.md` with disclosure policy.
- [x] `CONTRIBUTING.md`.
- [x] `CODE_OF_CONDUCT.md`.
- [x] `MANIFEST_SCHEMA.json`.
- [x] `LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml`.
- [x] `LIOUSCOPE_EVIDENCE_LOCK_REGISTER.csv`.
- [x] `LIOUSCOPE_DRIVE_ATTESTATION.csv`.
- [x] CI matrix 3.10-3.13.
- [x] `pytest` green (145/145).
- [x] coverage >= 80% (current: 88.01%).
- [x] `ruff check` clean.

## P2 -- Package

- [x] `pyproject.toml` validated by `validate-pyproject` in CI.
- [x] `python -m build` step in publish workflow.
- [x] `twine check dist/*` step in publish workflow.
- [x] TestPyPI smoke gate via `workflow_dispatch`.
- [x] PyPI Trusted Publisher (OIDC) configured.
- [ ] Final PyPI URL recorded (after first release).

## P3 -- Archival

- [ ] GitHub release tag `v0.2.0` pushed.
- [ ] Zenodo DOI claimed (auto-mint via GitHub--Zenodo integration).
- [ ] DOI inserted into `CITATION.cff` `identifiers[0].value`.
- [ ] SWHID submitted to Software Heritage (`swh save origin`).
- [ ] Archive checksums recorded in `CHANGELOG.md`.

## P4 -- Security

- [x] Dependency Review Action (`.github/workflows/dependency-review.yml`).
- [x] Dependabot config (`.github/dependabot.yml`).
- [x] OpenSSF Scorecard workflow (`.github/workflows/scorecard.yml`).
- [x] CodeQL workflow (`.github/workflows/codeql.yml`).
- [x] CycloneDX SBOM workflow (`.github/workflows/sbom.yml`).
- [x] Pre-commit hooks (`.pre-commit-config.yaml`).
- [x] Minimal workflow permissions (`contents: read` default).
- [ ] Pinned actions by full commit SHA (currently pinned by major-version tag; SHA pinning recommended before public release).

## P5 -- Drive hygiene (operational, off-repo)

- [ ] 0Liou+ folder restructure: `00_CANON`, `10_REPORTS`,
      `20_RELEASE_TEMPLATES`, `70_ARCHIVE`, `90_SUPERSEDED_DO_NOT_USE`.
- [ ] `README_CANON_0LIOU_PLUS_v1_0.md` placed at top of 0Liou+.
- [ ] Duplicate PDF renamed `DUPLICATE_DO_NOT_USE_*` or moved to 70_ARCHIVE.

## Final acceptance criteria for "public release"

All P0..P4 boxes ticked, plus:

- [ ] `git tag v0.2.0` pushed.
- [ ] GitHub release created (auto-triggers `pypi.yml`).
- [ ] PyPI page live; `pip install liouscope` works.
- [ ] Zenodo DOI live; updated in `CITATION.cff`.
- [ ] PR #1 merged to `main`.

Until all boxes are ticked, the status string in
`LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml` stays
`PUBLIC_RELEASE_FINAL: OPEN`.
