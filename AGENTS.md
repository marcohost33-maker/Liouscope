---
name: liouscope-agents-md
description: AI coding agent instructions for LiouScope (open quantum lattice diagnostics)
version: "1.4"
last_updated: 2026-06-26
priority_when_in_conflict: 1
---

# AGENTS.md — LiouScope

> AI Coding Agent Instructions. Tool-agnostic format per
> [agents.md](https://agents.md/) (Linux Foundation AAIF standard, Dec 2025).
> Read by Codex, Cursor, Goose and others. Claude Code does not read AGENTS.md
> natively (issue anthropics/claude-code#6235); see `CLAUDE.md` which imports
> this file via `@AGENTS.md`. Coworkerz convention: AGENTS.md is the
> single-source-of-truth; `CLAUDE.md` is the thin import layer.

> **Priority when working agreements conflict:** lower number wins.
> §1 (working agreements) > §2 (conventions) > §3 (don't/do) > §4 (when stuck).

## Project context

- **Stack:** Python >=3.10 (CI matrix 3.10/3.11/3.12/3.13/3.14), pytest, ruff, mypy,
  NumPy / SciPy, optional QuTiP cross-checks.
- **Purpose:** Multi-diagnostic relaxation analysis for open quantum lattice
  systems (GKSL / Lindblad). 24 diagnostics D1-D24 in six layers (D1-D20 =
  peer-review submission set; D21-D24 post-submission, D24 = opt-in Zhou
  mixing-time predictor; schema `D1-D24-Übersicht-v3`) + twelve mechanism
  classes A1-A12. Replaces single-number "decay rate" with a layered,
  auditable `DiagnosticReport`.
- **Version:** see `src/liouscope/_version.py` (single source; `pyproject.toml`
  reads it dynamically — numbers drift, pointers don't).
- **Taxonomy version:** `A1-A12-v3.1`
- **Manifest schema:** `src/liouscope/MANIFEST_SCHEMA.json` (version: see its
  `schema_version` const; SHA-256-stable run manifests)
- **License:** Apache-2.0
- **Visibility:** PUBLIC (`marcohost33-maker/Liouscope`). Do not assume that
  internal Drive/canon context is public unless it is explicitly committed or
  cited in release notes.
- **KANON anchor:** `RESEARCH-LIOUSCOPE` in `<internal-ref-redacted>`

## Repository layout

```
src/liouscope/            # main package
  └── MANIFEST_SCHEMA.json # contract for run manifests (packaged, NOT repo root)
tests/                    # pytest suite: anchors, qutip, numerics, fitting, classification
  └── test_anchors.py      # anchor regressions — must pass on every CI run
examples/                 # quickstart + tutorial scripts
benchmarks/               # performance / reproducibility scripts
figures/                  # generated diagnostic plots
CITATION.cff              # DOI / academic citation
codemeta.json             # CodeMeta 3.0 metadata
.github/workflows/
  ├── ci.yml              # test + lint + mypy + anchor regressions (required)
  ├── ci-qutip.yml        # optional-dependency QuTiP cross-checks (required)
  ├── scorecard.yml       # OpenSSF Scorecard (public-repo active)
  ├── encoding-guard.yml  # UTF-8 / line-ending guard
  ├── zizmor.yml          # workflow security audit (SHA-pinned actions)
  └── pypi.yml            # release publication template
```

## Build / test commands

```bash
python -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install -e .[dev,qutip]
pytest -q                       # full suite
pytest tests/test_anchors.py -v # anchor regressions only (CI gate)
ruff check src tests benchmarks
mypy src/liouscope              # enforcing CI gate (must exit 0)
python examples/quickstart.py   # smoke run
```

## Working agreements

1. **Branch protection: Tier-2 active.** `main` requires ALL required status
   checks green (strict=true): Python-matrix `test 3.10-3.14` + QuTiP
   cross-checks `3.11/3.12` (7 checks as of 2026-06-26 — authoritative list:
   branch protection via `gh api`, not this file). PRs only.
2. **Backup-First on destructive ops.** History-handling incident 2026-05-16
   wiped ~20 files via unverified branch-delete + GH-GC. Backup-Triple is the
   recovery anchor in `<internal-ref-redacted>`. Before any
   `git push --force`, branch delete, or ref-PATCH: capture pre-SHA via
   `gh api repos/.../git/refs/heads/...`, run the op, verify post-SHA.
3. **Anchor regressions are sacred.** `tests/test_anchors.py` pins the
   reference behaviour of D1-D20 on canonical fixtures. If anchors must
   change, do it in a dedicated PR with the physics rationale in the body,
   not as a side-effect of an unrelated change.
4. **SHA-pin all GitHub Actions.** Welle G established the gold-standard:
   every action reference is `@<full-sha>  # vX.Y.Z`. Dependabot is on a
   cooldown to avoid PR-spam.
5. **Reproducibility.** Manifest seeds, library versions, lattice geometry
   are recorded automatically; do not bypass the manifest writer.
6. **Reality-Anchor.** Prefer "anchors pass (verified <date>)" over
   "diagnostics implemented". No claims without code-belege.
7. **Plain paths in code-blocks** for file references (no markdown links —
   `code-block` paths are clickable in CLI; markdown links are not).

## Conventions

- **Imports:** absolute from `liouscope`, no `..` traversal.
- **Numerical libraries:** prefer `numpy`/`scipy`; `qutip` is an optional
  extra used for cross-checks, not a core runtime dependency.
- **Plots:** matplotlib only, no interactive backends in CI; save to
  `figures/` if persisted.
- **Citations:** changes touching results or methodology must update
  `CITATION.cff` and the relevant `MANIFEST_SCHEMA` version if the run
  manifest contract changes.
- **Docs:** README is the public surface; deep methodology in module
  docstrings.

## Branch & PR conventions (agents)

- **One agent = one branch prefix:** `claude/<task>` (Claude Code), `codex/<task>`
  (OpenAI Codex), `bot/<task>` (CI/automation). Human-led work: `feat|fix|docs/<task>`.
- **Agent output opens as a Draft PR** and stays draft until Definition-of-Done is
  verified; then mark ready.
- **Label agent PRs:** `agent:claude` / `agent:codex` / `agent:bot`.
- **Auto-merge over manual merge:** enable `gh pr merge --auto --squash` once required
  checks exist; a second concurrent PR must rebase on the updated main.
- **No concurrent agent pushes** to the same repo: serialize, or split work by branch
  namespace and let auto-merge order the merges.

## Don't

- Don't merge without ALL required status checks green (matrix + QuTiP).
- Don't use `--no-verify`, `--no-gpg-sign`, `--force` without explicit User1 OK.
- Don't bump `MANIFEST_SCHEMA` version without simultaneously updating
  consuming code paths and adding a backward-compat note in `CHANGELOG.md`.
- Don't introduce a "single decay rate" API surface — it contradicts the
  library's core thesis ("no single number").
- Don't commit or leak secrets, API keys, `.env`/credential files, `GITHUB_TOKEN`, or signing keys (logs included).

## Do

- Do run `pytest -q` + `ruff check` before pushing.
- Do update `CHANGELOG.md` on feature merge.
- Do reference the KANON anchor (`RESEARCH-LIOUSCOPE`) and
  `claim_status:pending` for new diagnostics until anchors confirm them.
- For any branch / history-touching operation, reference the 2026-05-16
  incident in the PR body as a reminder of why Backup-First
  exists (the backup-triple recovered ~20 files after an unverified
  branch-delete + GH-GC).

## When stuck

- See `README.md` "Why LiouScope" for the design philosophy
  (no-single-number, explicit uncertainty, auditable manifests).
- See `CHANGELOG.md` for what shipped in each version.
- See `src/liouscope/MANIFEST_SCHEMA.json` for the run-manifest contract.
- See `tests/test_anchors.py` for the canonical reference behaviour.
- See `docs/RELEASE_AUDIT_v0.5.0.md` for the current public/citable release gates.
- **Escalate after 3 failed attempts at the same step** — stop and ask in a PR
  draft or issue instead of looping.

## Definition of Done

A change is "done" only when **all** of the following hold:

| # | Check | Exit-Code / Evidence |
|---|---|---|
| 1 | `pytest -q` runs cleanly | exit 0 |
| 2 | `pytest tests/test_anchors.py -v` (anchor regressions) green | exit 0 |
| 3 | `ruff check src tests benchmarks` passes | exit 0 |
| 4 | ALL required status checks green on PR (`test 3.10-3.14` + `qutip-cross-check 3.11/3.12`, 7 as of 2026-06-26) | required-status checks |
| 5 | If methodology/results touched: `CITATION.cff` updated | PR diff |
| 6 | If run-manifest contract touched: `MANIFEST_SCHEMA.json` version bumped + `CHANGELOG.md` migration note | PR diff |
| 7 | `CHANGELOG.md` updated or an explicit no-changelog rationale is in the PR body | PR diff / PR body |
| 8 | PR body contains Summary + Test plan checklist | manual review |

A PR that misses any of 1-8 is not "ready". Anchor regressions (item 2) are
the sacred gate — never merge with them red, even if the change is unrelated.

---

*Tier-1 rollout 2026-05-28 (v1.0 → v1.1 hardening). Format spec: <https://agents.md/>.*
