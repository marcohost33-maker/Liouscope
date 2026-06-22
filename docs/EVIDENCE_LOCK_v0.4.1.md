# LiouScope Evidence Lock — v0.4.1

Date: 2026-06-22
Repository: `marcohost33-maker/Liouscope` (visibility: PRIVATE)
Runtime release: v0.4.1
Scope: provenance evidence lock for the v0.4.1 package/archive gates tracked in
issue #50 and `docs/RELEASE_AUDIT_v0.4.1.md` §5. Documentation-only — no
physics, numerics, diagnostic outputs, manifest schema, taxonomy, or runtime API
change. This file is the living register; the audit (`RELEASE_AUDIT_v0.4.1.md`)
is the point-in-time analysis it discharges.

## 1. Release target (verified)

| Field | Value | Verification |
|---|---|---|
| Tag | `v0.4.1` | `git rev-parse v0.4.1` |
| Release commit | `1965f2ba664ab6497236fed27c480e9edc04e102` | matches GitHub tag SHA and the local fetched tag |
| Commit subject | `release: cut v0.4.1 from Unreleased (PRs #43–#45) + sync canon/citation (#46)` | `git show` |
| GitHub Release | EXISTS — id `340608097`, published 2026-06-17, not draft/prerelease | GitHub Releases API |

The v0.4.1 git tag predates `docs/RELEASE_AUDIT_v0.4.1.md` and this file; neither
is part of the tagged v0.4.1 source snapshot. The GitHub Release **object** (not
the tag's source tree) carries the links to `CHANGELOG.md` and the audit, so the
source archive is unchanged.

## 2. Build provenance (from clean tag worktree)

Built from a detached worktree checked out at `v0.4.1` with
`SOURCE_DATE_EPOCH=1781680820` (the release commit's committer timestamp),
`python -m build` (PEP 517 isolated build).

| Artifact | SHA-256 | Reproducible |
|---|---|---|
| `liouscope-0.4.1-py3-none-any.whl` | `3b5c833aabe631ddd1ce176b6da42fbfd88286ecbf4bf002838b2a7d0a7e9257` | **Yes** — byte-identical across three independent builds |
| `liouscope-0.4.1.tar.gz` (sdist) | `79324efab2425792af554f09c2d03891627037c350be5477af76971026c4cb9b` | **No** — three builds produced three different digests |

Reproducibility note (root cause diagnosed, not hand-waved): the **wheel** is
byte-reproducible with `SOURCE_DATE_EPOCH` set (the same digest `3b5c833a…`
recurred on three independent builds). The **sdist** is *not* byte-stable; the
precise cause was isolated by comparing two sdists member-by-member:

- the gzip layer **and** the uncompressed tar both differ, but the tar **member
  order is identical** and **every file's content is byte-identical** (no
  content diff on any of the 88 members);
- the only differences are tar **member mtimes**, and they carry the build
  wall-clock time (e.g. `1782156192` vs `1782156197`, ~6 s apart), **not**
  `SOURCE_DATE_EPOCH` (`1781680820`). setuptools does not apply
  `SOURCE_DATE_EPOCH` to the sdist tar entries (it does for the wheel zip);
- proof of content identity: normalising every member's `mtime` to 0 makes the
  two sdists hash **identically** (`7fb29e2c…`).

So the sdist's *content* is fully deterministic; only timestamps vary. The
recorded sdist digest identifies *this specific artifact*. For a byte-stable
sdist later, either post-process the tar to clamp member mtimes to
`SOURCE_DATE_EPOCH`, or pin the published sdist as a release asset and record
that asset's hash as authoritative.

## 3. Distribution metadata check

State-of-the-art validator sweep on the clean-tag artifacts, all green:

| Validator | Result |
|---|---|
| `twine check --strict dist/*` | **PASSED** (wheel + sdist; strict = warnings are errors) |
| `validate-pyproject pyproject.toml` | **Valid file** |
| `check-wheel-contents dist/*.whl` | **OK** |

Root-cause finding (validated and falsified): an initial `twine check` in the
container reported `InvalidDistribution: unrecognized or malformed field
'license-expression' / 'license-file'`. This is **not** a package defect. The
wheel correctly declares `Metadata-Version: 2.4` with PEP 639
`License-Expression: Apache-2.0` + `License-File: LICENSE` (set by
`pyproject.toml` `license = "Apache-2.0"` / `license-files = ["LICENSE"]`). The
failure came from the container's distro-managed `packaging==24.0`, which
predates PEP 639 / Metadata 2.4 support. **Falsification test:** re-running
`twine check` in a clean venv with `packaging>=24.2` (resolved to 26.2) →
**PASSED** on the identical artifacts. Conclusion: the package is correct;
release tooling must use `packaging>=24.2` (already noted in the #46 commit
message). The repo's CI installs current tooling, so this affects only stale
local environments.

## 4. Required status checks on the tag commit (verified GREEN)

Required checks were confirmed on the **intended v0.4.1 tag commit**
(`1965f2b`), not merely on a later PR head:

| Check | Workflow run | Conclusion |
|---|---|---|
| `test (ubuntu-latest, 3.10)` | CI run `27672586182` | success |
| `test (ubuntu-latest, 3.11)` | CI run `27672586182` | success |
| `test (ubuntu-latest, 3.12)` | CI run `27672586182` | success |
| `test (ubuntu-latest, 3.13)` | CI run `27672586182` | success |
| QuTiP cross-check (3.11 / 3.12) | `ci-qutip.yml` run `27672586173` | success |

`ci.yml` on the tag predates PR #48, so its matrix is 3.10–3.13 (no 3.14 job) —
consistent with the matrix at that commit. The authoritative required-check list
remains branch protection (`gh api repos/.../branches/main/protection`), not this
file; the six checks above match the AGENTS.md §1 statement (test 3.10–3.13 +
QuTiP 3.11/3.12).

Python 3.14: in the matrix and classifiers since PR #48 but **not** promoted to a
required check — promote only after stable green history (audit risk P2).

## 5. Source-archive identifiers (SWHID)

The audit framed SWHID as "capture after public source availability". Web
research (Software Heritage docs/tutorial, 2025) **refines** this: SWHIDs are
*intrinsic* — the core identifier is computed from the artifact itself and can be
produced for private code. Only **resolution in the public Software Heritage
archive** (via "Save Code Now") requires a public repository.

Core SWHIDs captured now (intrinsic git object hashes, equal to the SWHID core):

```text
swh:1:rev:1965f2ba664ab6497236fed27c480e9edc04e102   # revision (release commit)
swh:1:dir:a47213a006649527f8f6265b61254deb5d75ec4f   # directory (root tree of the tag)
```

Validation (official tool, not assertion): the directory SWHID was confirmed
with Software Heritage's own `swh identify` (`swh.model`) run on a **clean,
`.git`-free** export of the tag (`git archive v0.4.1 | tar -x`):

```text
swh:1:dir:a47213a006649527f8f6265b61254deb5d75ec4f   ← swh identify on the export
a47213a006649527f8f6265b61254deb5d75ec4f             ← git rev-parse v0.4.1^{tree}
```

The two match exactly, confirming the SWHID's git-compatible design
(`swh:1:dir` core == git tree hash; `swh:1:rev` core == git commit hash).

Pitfall recorded: running `swh identify --type directory` directly on a *working
tree* yields a **different** value (here `swh:1:dir:96d5a10f…`) because the
filesystem walk includes the `.git` worktree pointer that is not part of the git
tree. Always identify a `.git`-free export (or the git object) for a source
SWHID that matches the repository tree.

Archive resolution is **deferred** until the source is public; the limitation is
the repo's PRIVATE visibility, not a SWHID gap. When/if the repo goes public,
"Save Code Now" the v0.4.1 commit and append the qualified, archive-resolvable
SWHID here.

## 6. Decisions (ADRs)

All three publication decisions are governed by the PRIVATE repo visibility and
are recorded as the explicit, reversible choices the checklist calls for. Going
public / publishing are outward-facing, irreversible steps and are deliberately
left to an explicit owner decision.

- **ADR-1 PyPI: DEFER (no-PyPI while private).** A public index entry would
  publish the source-derived distribution of a research/pre-clinical package
  ("not for diagnostic use"). When publication is desired, it **must** use PyPI
  Trusted Publishing / OIDC, not a long-lived API token (PyPI's own
  recommendation; short-lived per-run tokens, no exfiltratable secret). The repo
  template `.github/workflows/pypi.yml` is already OIDC-shaped and guarded by the
  `PYPI_PUBLISH_ENABLED` repo variable.
- **ADR-2 Zenodo: DEFER (no-Zenodo via GitHub integration while private).**
  Zenodo's automatic GitHub archiving requires a public repo (webhook + release
  ZIP download). Manual upload is possible but also publishes the source. No DOI
  is minted now; `CITATION.cff` gains a DOI only once one exists.
- **ADR-3 SWHID: intrinsic core captured now; archive resolution deferred**
  (see §5). Recorded private-repo limitation; not blocked on tooling.

## 7. Status vs. issue #50 checklist

| Item | Status |
|---|---|
| confirm release commit and tag target | ✅ §1 |
| create or verify tag v0.4.1 | ✅ §1 |
| create GitHub Release v0.4.1 | ✅ already published 2026-06-17 |
| build wheel and sdist from clean tag | ✅ §2 |
| run twine check | ✅ §3 (PASSED with `packaging>=24.2`) |
| record SHA256 hashes | ✅ §2 |
| decide PyPI vs no-PyPI | ✅ ADR-1 (defer; OIDC mandated when used) |
| if PyPI used, Trusted Publishing/OIDC | ✅ documented (ADR-1; `pypi.yml`) |
| decide Zenodo DOI vs no-Zenodo | ✅ ADR-2 (defer) |
| decide SWHID vs private-repo limitation | ✅ ADR-3 / §5 (intrinsic captured) |
| document required status checks | ✅ §4 |
| update evidence lock register | ✅ this file |
| update 0Liou+ Drive canon to v0.4.1 | ⏳ OPEN — external Drive action, owner-driven |

## 8. Remaining owner-driven / external work

1. **0Liou+ Drive canon** → update repo runtime canon to v0.4.1 (keep scientific
   report v2.0 FINAL as baseline). External to this repo.
2. **Promote Python 3.14** to a required check after stable green history.
3. **If going public/citable** (separate explicit decision): reverse ADR-1/2/3 —
   publish via OIDC, enable Zenodo for the DOI, "Save Code Now" for the
   archive-resolvable SWHID, then add the DOI to `CITATION.cff`.
