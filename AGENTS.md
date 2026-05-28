# AGENTS.md — LiouScope

> AI Coding Agent Instructions. Tool-agnostic format per
> [agents.md](https://agents.md/) (Linux Foundation AAIF standard, Dec 2025).
> Read by Codex, Claude Code, Cursor, Goose and others. Coworkerz convention:
> AGENTS.md is primary across all repos.

## Project context

- **Stack:** Python >=3.10 (CI matrix 3.10/3.11/3.12/3.13), pytest, ruff, mypy,
  NumPy / SciPy, optional QuTiP cross-checks.
- **Purpose:** Multi-diagnostic relaxation analysis for open quantum lattice
  systems (GKSL / Lindblad). Twenty diagnostics D1-D20 in six layers + twelve
  mechanism classes A1-A12. Replaces single-number "decay rate" with a
  layered, auditable `DiagnosticReport`.
- **Version:** 0.2.0 (see `pyproject.toml`)
- **Taxonomy version:** `A1-A12-v3.1`
- **Manifest schema:** `MANIFEST_SCHEMA.json` v1.2.0 (SHA-256-stable run manifests)
- **License:** Apache-2.0
- **Visibility:** PRIVATE (marcohost33-maker/Liouscope)
- **KANON anchor:** `RESEARCH-LIOUSCOPE` in `Vero/Meta/KANON/KANON_APPS.yaml`

## Repository layout

```
src/liouscope/            # main package
tests/                    # pytest suite (~18 test modules)
  └── test_anchors.py      # anchor regressions — must pass on every CI run
examples/                 # quickstart + tutorial scripts
benchmarks/               # performance / reproducibility scripts
figures/                  # generated diagnostic plots
MANIFEST_SCHEMA.json      # contract for run manifests (v1.2.0)
CITATION.cff              # DOI / academic citation
codemeta.json             # CodeMeta 3.0 metadata
.github/workflows/
  ├── ci.yml              # test + lint + mypy + anchor regressions (required)
  ├── scorecard.yml       # OpenSSF Scorecard (private-repo guarded)
  ├── encoding-guard.yml  # UTF-8 / line-ending guard
  ├── zizmor.yml          # workflow security audit (SHA-pinned actions)
  └── pypi.yml            # release publication template
```

## Build / test commands

```bash
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -e .[dev,qutip]
pytest -q                      # full suite
pytest tests/test_anchors.py -v   # anchor regressions only (CI gate)
ruff check src tests
mypy src/liouscope             # currently continue-on-error in CI
python examples/quickstart.py  # smoke run
```

## Working agreements

1. **Branch protection: Tier-2 active.** `main` requires `ci.yml` jobs green
   (4 required status checks across the Python matrix, strict=true). PRs only.
2. **Backup-First on destructive ops.** Loss-of-history incident 2026-05-16
   wiped ~20 files via unverified branch-delete + GH-GC. Backup-Triple is the
   recovery anchor in `Vero/Liouscope_Backup_2026-05-16/`. Before any
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

## Don't

- Don't merge without all 4 Python-matrix `ci.yml` jobs green.
- Don't use `--no-verify`, `--no-gpg-sign`, `--force` without explicit User1 OK.
- Don't bump `MANIFEST_SCHEMA` version without simultaneously updating
  consuming code paths and adding a backward-compat note in `CHANGELOG.md`.
- Don't introduce a "single decay rate" API surface — it contradicts the
  library's core thesis ("no single number").
- Don't leak `GITHUB_TOKEN` or signing keys in logs.

## Do

- Do run `pytest -q` + `ruff check` before pushing.
- Do update `CHANGELOG.md` on feature merge.
- Do reference the KANON anchor (`RESEARCH-LIOUSCOPE`) and
  `claim_status:pending` for new diagnostics until anchors confirm them.
- Do reference `[[memory:liouscope_data_loss_2026_05_16]]` in PR bodies for
  any branch / history-touching operation as a reminder of the disaster.

## When stuck

- See `README.md` "Why LiouScope" for the design philosophy
  (no-single-number, explicit uncertainty, auditable manifests).
- See `CHANGELOG.md` for what shipped in each version.
- See `MANIFEST_SCHEMA.json` for the run-manifest contract (v1.2.0).
- See `tests/test_anchors.py` for the canonical reference behaviour.

## Coworkerz-specific anchors

- KANON: `C:\Users\marco\OneDrive\Desktop\Vero\Meta\KANON\KANON_APPS.yaml`
  (section `RESEARCH-LIOUSCOPE`)
- Memory: `C:\Users\marco\.claude\projects\C--Users-marco\memory\MEMORY.md`
- Recovery backup: `C:\Users\marco\OneDrive\Desktop\Vero\Liouscope_Backup_2026-05-16\`
- Rollout plan: `C:\Users\marco\OneDrive\Desktop\Vero\Vero Pläne\2026-05-23_agents-md-cross-repo-rollout-plan.md`

---

*Tier-1 rollout 2026-05-28. Format spec: <https://agents.md/>.*
