# LiouScope Canon Status

Last updated: 2026-07-07
Repository: `marcohost33-maker/Liouscope`

## 1. Canon layers

LiouScope currently has three canon layers that must not be collapsed into one
sentence without qualification.

| Layer | Current status | Authority |
|---|---|---|
| Scientific/Drive canon | 0Liou+ v1.0 / consolidated report v2.0 FINAL / `LiouScope_Formelbuechlein_v1_0_2026-06-25` / A1-A12-v3.1 / D1-D24 | Drive-side canon and project archive |
| Repo runtime canon | released `v0.5.0`; default branch `0.6.0.dev0` | tag/release plus `src/liouscope/_version.py` |
| Manifest/report contracts | `MANIFEST_SCHEMA_VERSION == 1.5.0`; schema 1.4 added the canonical structured-ensemble-evidence digest to the hash domain when such evidence is supplied; schema 1.5 makes `compute_input_hash` injective via length-framed, type-tagged fields; hashes remain comparable only within one schema version. `StabilityReport v2.1` is a separate additive projection | `src/liouscope/MANIFEST_SCHEMA.json`, `_consts.py`, `STABILITY_REPORT_SCHEMA.json` |
| Current code frontier | v0.5.0 remains immutable and citable; default branch is the next development line; archive/provenance finalization remains open in issue #50 | GitHub release state + `docs/RELEASE_AUDIT_v0.5.0.md` |

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
Released repo version: 0.5.0
Default-branch development version: 0.6.0.dev0
Release metadata: GREEN for the immutable engineering release; DOI still unverified
Public/citable release: OPEN (archive/provenance gates pending — see RELEASE_AUDIT_v0.5.0)
```

## 4. No-drift invariants

The following identifiers must remain synchronized or be explicitly version-bumped:

- `TAXONOMY_VERSION = "A1-A12-v3.1"`
- `DIAGNOSTIC_SCHEMA_VERSION = "D1-D24-Übersicht-v3-2026-04-24"`
- `MANIFEST_SCHEMA_VERSION = "1.5.0"`
- `liouscope.__version__` from `src/liouscope/_version.py`
- released version metadata in `CITATION.cff`
- release notes / development migration note
- StabilityReport schema when the StabilityReport projection changes

## 5. Pending canon work

- Update the Drive-side 0Liou+ canon to distinguish released v0.5.0 from the
  default-branch 0.6.0 development line and manifest schema 1.4.
- Keep the scientific report v2.0 FINAL as the baseline unless a new reviewed
  scientific report supersedes it.
- Do not relabel D24 as an exact implementation of Zhou Eq.(16); keep the
  documented status as a related, generally coarser surrogate / reference-
  verified-bound-coarser claim.
- A public/citable v0.5.0 still depends on the archive/provenance gates in
  `docs/RELEASE_AUDIT_v0.5.0.md`: PyPI Trusted Publisher activation, direct
  Zenodo DOI verification/backfill, SWHID/public-source status, and evidence-lock
  update.
- Keep issue #50 open until those external gates are directly verified.

## 6. Production rule for future changes

Any future PR that changes one of the following must include a canon-impact note:

- diagnostic IDs or schema,
- A-class taxonomy semantics,
- manifest fields or hashes,
- StabilityReport fields or schema,
- release version metadata,
- citation metadata,
- D24 claim wording,
- branch protection / release workflow policy,
- public/private visibility or publication channel.
