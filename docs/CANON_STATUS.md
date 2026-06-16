# LiouScope Canon Status

Last updated: 2026-06-16
Repository: `marcohost33-maker/Liouscope`

## 1. Canon layers

LiouScope currently has two canon layers that must not be collapsed into one
sentence without qualification.

| Layer | Current status | Authority |
|---|---|---|
| Scientific/Drive canon | 0Liou+ v1.0 / consolidated report v2.0 FINAL / A1-A12-v3.1 / D1-D24 | Drive-side canon and project archive |
| Repo runtime canon | `liouscope.__version__ == 0.4.1` | `src/liouscope/_version.py` |
| Manifest contract | `MANIFEST_SCHEMA_VERSION == 1.2.0` | `src/liouscope/MANIFEST_SCHEMA.json` and `_consts.py` |
| Current code frontier | PRs #43–#45 merged to `main`; v0.4.1 release being cut | GitHub PR state |

## 2. Operational rule

Use this rule in future audits and release work:

```text
Drive canon describes the scientific/report baseline.
Repo runtime canon describes the currently importable software version.
Release canon is final only when metadata, CI, archive, and provenance gates all pass.
```

## 3. Current working verdict

```text
Science baseline: GREEN for existing scope
Runtime repo version: 0.4.1
Release metadata: GREEN for engineering release (version/citation/changelog aligned)
Public/citable release: OPEN (archive/provenance gates pending — see RELEASE_AUDIT)
PRs #43-#45: merged to main; v0.4.1 changelog cut from the Unreleased section
```

## 4. No-drift invariants

The following identifiers must remain synchronized or be explicitly version-bumped:

- `TAXONOMY_VERSION = "A1-A12-v3.1"`
- `DIAGNOSTIC_SCHEMA_VERSION = "D1-D24-Übersicht-v3-2026-04-24"`
- `MANIFEST_SCHEMA_VERSION = "1.2.0"`
- `liouscope.__version__` from `src/liouscope/_version.py`
- `CITATION.cff` top-level `version`
- release notes / changelog version section

## 5. Pending canon work

- Update the Drive-side 0Liou+ canon to reflect the repo runtime moving from the
  older package canon to v0.4.1.
- Keep the scientific report v2.0 FINAL as the baseline unless a new reviewed
  scientific report supersedes it.
- Do not relabel D24 as an exact implementation of Zhou Eq.(16); keep the
  documented status as a related, generally coarser surrogate / reference-
  verified-bound-coarser claim.
- A public/citable v0.4.1 still depends on the archive/provenance gates in
  `docs/RELEASE_AUDIT_v0.4.0.md` §5 (GitHub release notes, PyPI/Zenodo/SWHID as
  applicable); the engineering release (code + metadata) is green.

## 6. Production rule for future changes

Any future PR that changes one of the following must include a canon-impact note:

- diagnostic IDs or schema,
- A-class taxonomy semantics,
- manifest fields or hashes,
- release version metadata,
- citation metadata,
- D24 claim wording,
- branch protection / release workflow policy.
