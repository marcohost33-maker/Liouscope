# LiouScope Release Audit v0.5.0

Date: 2026-06-25
Repository: `marcohost33-maker/Liouscope`
Runtime release: v0.5.0
Scope: release-prep readiness for the Canon v0.5 diagnostics wave + cross-family
CPTP-Choi hardening; metadata sync; PyPI Trusted-Publishing release workflow.

## 1. Purpose

This audit accompanies the `v0.4.1` → `v0.5.0` MINOR bump. It is documentation-
and metadata-oriented: it records what shipped on `main` after `v0.4.1`, verifies
version/citation/changelog alignment, and stages the release-publication path
(GitHub Release, PyPI Trusted Publishing, Zenodo/DOI). It introduces no physics,
numerics, manifest-schema, or taxonomy change beyond what already merged on
`main` under `[Unreleased]`.

## 2. What is new in v0.5.0

MINOR bump per SemVer: backward-compatible, additive API surface.

- **Canon v0.5 diagnostics & contracts** — 8 validated formelbuch entries
  (LIOU-A-011/A-012/A-013, F-018/F-019/F-020, RPT-001, NG-003), each pinned to an
  *independent* oracle (closed form, QuTiP, or analytic soll-value):
  - **CPTP Choi-PSD gate** (`liouscope.numerics.cptp`, LIOU-A-011): complete-
    positivity of `exp(dt*L)` via the Choi-matrix minimum eigenvalue + trace-
    preservation residual. Oracles: transpose map `min_eig = -1`, dephasing on the
    CP boundary `min_eig = 0`, Euler-step non-CP counterexample (NR-002).
  - **Trace distance D_tr** (`diagnostics.relaxation.trace_distance`, LIOU-F-018):
    `(1/2)||rho-sigma||_1` + `RelaxationResult.trace_distance_curve`. Oracles:
    orthogonal pure states → 1, diagonal → total variation, Fuchs–van de Graaf
    bounds, CPTP contractivity, `qutip.tracedist`.
  - **Temporal holdout split** (`fitting.holdout`, LIOU-A-012): out-of-sample
    anti-overfit gate for the M0–M3b hierarchy (time-ordered tail, no shuffling).
  - **Residual-whiteness gate** (`fitting.whiteness`, LIOU-A-013): Ljung-Box Q vs
    the `scipy.stats` chi-squared reference (white noise passes, AR(1) rejected).
  - **Metamorphic spectral oracles** (LIOU-F-020): `gap(c*L)=c*gap(L)` and
    `spec(U L U^dag)=spec(L)` — ground-truth-free invariants.
  - **Gap-invariant reproduction** (LIOU-F-019): exact pack parametrisation
    (amplitude damping `gap=gamma/2`; thermal `gap=0.55`).
  - **StabilityReport v2.1 contract** (`io.stability_report` + packaged
    `STABILITY_REPORT_SCHEMA.json`, LIOU-RPT-001): a claim-safe, machine-auditable
    projection of a `DiagnosticReport` (`claim_level`, `direction`,
    `cp_evidence_level`, recomputed invariant residuals, `evidence_bundle`,
    `provenance`). Purely additive — it does **not** modify the run manifest,
    `MANIFEST_SCHEMA.json`, or `DiagnosticReport` (no schema-version bump; older
    artefacts stay valid).
  - **Petermann interpretation caveat** (D9 docstring, LIOU-NG-003): a large
    Petermann factor is necessary-but-not-sufficient for transient amplification.
- **CPTP Choi gate hardening (PR #55, B1/B2)** — cross-family math review:
  trace preservation is now checked at the propagator `Phi = exp(dt*L)` (not the
  dt-/scale-blind generator); Choi Hermiticity is checked *before* the PSD test
  (Hermitising it masked non-HP defects); absolute `1e-9` tolerances replaced by
  scale-invariant relative ones. `ChoiGateResult` gains `choi_herm_residual` and
  `is_hp`. Regression tests for both repros in `tests/test_cptp_choi.py`.
- **Independent-oracle cross-checks** for the non-normality layer D8–D11 and the
  spectral diagnostic layer (test-only, no production-code change).
- **Repo-wide mypy gate fix** — `[tool.mypy] python_version` raised `3.10 → 3.12`
  so mypy can parse NumPy ≥2.5 PEP-695 stubs. Runtime 3.10/3.11 support stays
  guarded by the test matrix and `ruff target-version = py310`; type-checking
  semantics only, no packaged-code change.
- Every new public boundary ships negative/edge-input gates (negative `dt`, NaN,
  non-square dim, bad `holdout_frac`, `m>=N`, non-positive dof, out-of-enum
  verdict fields, tampered schema fields).

## 3. Current validated state

| Area | Current state |
|---|---|
| Runtime version | `src/liouscope/_version.py` reports `0.5.0`; `importlib.metadata.version("liouscope") == 0.5.0` (single source, dynamic in `pyproject.toml`). |
| Citation metadata | `CITATION.cff` reports `version: 0.5.0`, `date-released: 2026-06-25`; DOI staged (commented placeholder, no DOI minted). |
| CodeMeta | `codemeta.json` `version`/`softwareVersion: 0.5.0`, `datePublished: 2026-06-25`. |
| Diagnostic schema | D1-D24 (D1-D20 submission set; D21-D24 post-submission). Unchanged. |
| Taxonomy | `A1-A12-v3.1`. Unchanged. |
| Manifest contract | `MANIFEST_SCHEMA_VERSION == 1.2.0`. Unchanged (StabilityReport v2.1 is a separate additive schema). |
| Changelog | `[0.5.0] - 2026-06-25` section curated; fresh empty `[Unreleased]`; compare-links pinned. |

## 4. Verification (local CI command-chain mirror)

Verified locally on this branch (CPython via `.venv`, `pip install -e .[dev,qutip]`
+ `jsonschema`):

| Gate | Command | Result |
|---|---|---|
| Lint | `ruff check src tests benchmarks` | exit 0 — All checks passed |
| Type | `mypy src/liouscope` | exit 0 — no issues in 49 source files |
| Anchors (sacred) | `pytest tests/test_anchors.py -q` | 21 passed |
| Full suite + coverage | `pytest --cov=liouscope --cov-fail-under=80 -q` | 380 passed, coverage 93.95% (≥ 80% gate) |
| QuTiP cross-checks | `pytest -m qutip -q` | 8 passed, 372 deselected |
| Version single-source | `liouscope.__version__` == `importlib.metadata.version("liouscope")` | both `0.5.0` |

Note: the CI matrix installs only `.[dev,qutip]` (no `jsonschema`), so CI shows
**379 passed, 1 skipped** (`tests/test_manifest.py` jsonschema-extra skip),
coverage 93.73%. Installing `jsonschema` runs that one test → **380 passed,
93.95%**. Both are the same suite; the single skip is the optional-dependency
guard, not a failure.

## 5. PyPI Trusted-Publishing release workflow

`.github/workflows/pypi.yml` is a real OIDC Trusted-Publishing release workflow
(no API token committed):

- Trigger: `on: release: types: [published]` + `workflow_dispatch`. It does
  **not** run on pull requests, so it cannot affect the required PR checks and
  cannot publish from this PR.
- `permissions: id-token: write` + `contents: read`; `environment: pypi`.
- Build: `python -m build`; publish: `pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b  # v1.14.0` (SHA-pinned), with `print-hash: true`.
- **Release evidence:** `print-hash: true` makes the publish step log the
  SHA-256 / MD5 / BLAKE2-256 of every uploaded sdist/wheel, so the evidence lock
  (§7) can record the published digests directly from the run log. Under Trusted
  Publishing the PyPA action **already** generates and uploads a PEP 740 digital
  attestation by default (default-on since `gh-action-pypi-publish` **v1.11.0**;
  we pin v1.14.0), so we deliberately do **not** add a second
  `attest-build-provenance` step: PyPI accepts **at most two attestations per
  file and rejects more than two or any repeated predicate type** (PyPI docs,
  *Publish Attestations v1*), so a redundant attest step would be rejected. The
  single PyPA-managed attestation is the provenance source of record. Sources
  verified 2026-06-29: PyPA `gh-action-pypi-publish` README (`print-hash`,
  `attestations` default), PyPI Docs `docs.pypi.org/attestations/publish/v1/`,
  PyPI blog *"PyPI now supports digital attestations"* (2024-11-14).
- **Safety guard retained:** `if: vars.PYPI_PUBLISH_ENABLED == 'true'`. Until
  Marco sets the repo variable, a published `v0.5.0` release will **skip** the
  publish job instead of failing with a "Trusted publishing exchange failure"
  (the exact failure recorded on 2026-05-21/05-28 before PyPI setup existed).
  This is the fail-closed choice for a release prepared *before* the PyPI
  Trusted Publisher is registered. **Flagged for Marco/Vero** — see §8.
- PyPI project name **`liouscope` is free** (PyPI JSON API
  `https://pypi.org/pypi/liouscope/json` → HTTP 404, 2026-06-25), so the
  pending-publisher claim can be created without a name collision.

## 6. v0.5.0 release verdict

| Dimension | Verdict | Rationale |
|---|---|---|
| Engineering release | GREEN | Version/citation/codemeta/changelog aligned; full suite + anchors + qutip green locally on the CI command chain. |
| Scientific claim safety | GREEN | Additive diagnostics, each pinned to an independent oracle; new diagnostics carry `claim_status="pending"`; no exact-Eq.(16) Zhou claim introduced. |
| CI status | GREEN expected | Required checks (test 3.10–3.14 + qutip-cross-check 3.11/3.12) replicated locally; CI confirmation on PR head pending. |
| Citation metadata | GREEN | `CITATION.cff`/`codemeta.json` synchronized to 0.5.0; DOI staged, not fabricated. |
| PyPI publication | YELLOW / PENDING | Workflow ready + name free; gated off until Marco registers the Trusted Publisher and sets `PYPI_PUBLISH_ENABLED=true`. |
| Public/citable release | YELLOW / OPEN | Requires GitHub Release, DOI/Zenodo decision, SWHID (private-repo limitation), evidence lock. |

## 7. Definition-of-Done checklist (AGENTS.md)

```text
[REPO]
[x] Runtime version single-source is v0.5.0 (verified: __version__ == dist metadata)
[x] CITATION.cff version/date match v0.5.0 (2026-06-25)
[x] codemeta.json version/softwareVersion/datePublished synchronized
[x] Changelog has [0.5.0] section + fresh [Unreleased] + pinned compare-links
[x] DOI staged without fabrication (commented placeholder)

[CI / QUALITY]
[x] ruff check src tests benchmarks (exit 0)
[x] mypy src/liouscope (exit 0)
[x] pytest tests/test_anchors.py (21 passed — sacred gate)
[x] full suite + coverage (380 passed, 93.95% >= 80%)
[x] pytest -m qutip (8 passed)
[ ] ALL required checks GREEN on PR head (test 3.10-3.14 + qutip-cross-check 3.11/3.12) — pending CI

[PACKAGE / PROVENANCE]
[x] pypi.yml is a real OIDC Trusted-Publishing release workflow (no token), SHA-pinned
[x] pypi.yml does NOT publish on this PR (only on release, and gated by PYPI_PUBLISH_ENABLED)
[x] PyPI name 'liouscope' verified free (HTTP 404)
[ ] Marco: register PyPI Trusted Publisher (pending-publisher) + create 'pypi' environment
[ ] Marco: set repo variable PYPI_PUBLISH_ENABLED=true
[ ] Build wheel+sdist from clean tag, twine check, record SHA256

[ARCHIVE / CITATION]
[ ] Create + tag GitHub Release v0.5.0 (Vero, at finalization)
[ ] Zenodo DOI prepared or explicit no-Zenodo decision; backfill CITATION.cff doi
[ ] SWHID after public source availability, or record private-repo limitation
[ ] Evidence lock register updated with release commit, tag, checksums, DOI/SWHID
```

## 8. Open risks / decisions flagged for Marco/Vero

| Item | Severity | Note |
|---|---:|---|
| `PYPI_PUBLISH_ENABLED` guard retained on pypi.yml | DECISION | Kept fail-closed so a `v0.5.0` release tag does not fail-publish before the PyPI Trusted Publisher exists. Removing the guard activates publishing on the next release — only do so *after* the PyPI setup in §5. |
| Release tag + GitHub Release | P1 | Not created in this PR (per envelope: Vero finalizes). The `[0.5.0]` compare-link resolves once `v0.5.0` is tagged. |
| DOI placeholder | P2 | `CITATION.cff` `doi:` is a commented placeholder; backfill the real DOI at Zenodo release. No DOI fabricated. |
| SWHID while repo private | P2 | Capture only after public source availability; else record the private-repo limitation. |

## 9. Marco — zero-effort PyPI/Zenodo step list (at finalization)

1. PyPI → account → Publishing → **Add a pending publisher**: project `liouscope`,
   owner `marcohost33-maker`, repo `Liouscope`, workflow `pypi.yml`,
   environment `pypi`.
2. GitHub → repo Settings → Environments → create environment **`pypi`** (optional:
   required reviewers / tag-protection).
3. GitHub → repo Settings → Variables → set **`PYPI_PUBLISH_ENABLED = true`**.
4. Vero: create the **`v0.5.0` tag + GitHub Release** (reviewed notes from
   `CHANGELOG.md` `[0.5.0]` + this audit) → the publish job runs via OIDC, no token.
5. (Optional) Zenodo: enable the GitHub-Zenodo integration → mint DOI → backfill
   `CITATION.cff` `doi:`.
