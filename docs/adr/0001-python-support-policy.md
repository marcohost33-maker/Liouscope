# ADR 0001 — Scientific-Python support policy (SPEC 0)

- **Status:** Accepted
- **Date:** 2026-06-26 (consolidated 2026-06-29)
- **Deciders:** LiouScope maintainers
- **Supersedes:** `docs/ADR_SUPPORT_POLICY.md` (folded in here; that path is now a
  pointer stub)
- **Superseded by:** —

> This ADR consolidates two parallel records of the same decision: the original
> `docs/adr/0001-…` (PR #64) and `docs/ADR_SUPPORT_POLICY.md` (PR #61). The
> governance detail from the latter (required-steps checklist, release
> classification, evidence gate) is merged in below; the numbered `docs/adr/`
> structure and the web-verified SPEC 0 dates are kept.

## Context

LiouScope is a scientific Python package with a deliberately broad v0.5.0 runtime
matrix (`requires-python >=3.10`; CI on Python 3.10, 3.11, 3.12, 3.13, 3.14).
Issue #59 asks for an explicit support policy rather than silently changing the
runtime baseline inside an unrelated hardening PR.

Two pressures push toward narrowing the supported runtime window:

1. **Toolchain drift.** NumPy ≥ 2.5 ships type stubs that use the PEP 695 `type`
   statement; mypy only accepts them when its `python_version` target is ≥ 3.12,
   which is why the repo-wide mypy gate already runs at `python_version = 3.12`
   (see `CHANGELOG.md`, v0.5.0 "Fixed"). Keeping 3.10/3.11 as *runtime* targets
   while type-checking at 3.12 is a sustainable but widening split.
2. **Maintenance cost.** Every supported minor multiplies CI minutes, the
   dependency-floor compatibility surface, and the set of behaviours the anchor
   regressions must hold invariant.

The relevant external anchor is [Scientific Python SPEC 0][spec0], which
recommends:

- support each **Python** minor for **3 years** after its initial release, and
- support each **core package** (NumPy, SciPy, …) for **2 years** after its
  release.

SPEC 0 is a Scientific-Python *ecosystem* recommendation, **not** a claim that
upstream Python itself is EOL.

Under SPEC 0's published drop schedule (verified 2026-06-29): Python 3.10
(released 2021-10-04) and 3.11 (released 2022-10-24, SPEC 0 support window ending
**2025-10-23**) are already past the recommended 3-year window, while Python 3.12
(released 2023-10-02) remains inside it until **2026-10-01**.

## Decision

1. **Do not drop Python 3.10 or 3.11 in v0.5.x.** The v0.5.0 public-release lane
   stays focused on archiving, publication, citation metadata, and evidence
   locking; user-visible support changes must not be mixed into a
   release-publication gate.
2. **Adopt SPEC 0 as the forward support-policy reference.** Future baseline PRs
   must compare Python and core-dependency floors against SPEC 0 and document any
   intentional deviation.
3. **First candidate post-v0.5 baseline:** `requires-python >=3.12` with CI on
   3.12, 3.13, and 3.14. This is a *candidate*, not an automatic change; it
   requires a dedicated PR and release note.
4. **No silent dependency-floor bumps.** Raising NumPy/SciPy floors must be done
   together with CI evidence, resolver checks, and a compatibility note. A floor
   may remain older than SPEC 0 temporarily for user compatibility, but the
   deviation must be explicit.
5. **Mypy target is not the runtime floor.** The current `mypy` target of Python
   3.12 is a toolchain workaround for modern NumPy stubs. It does **not** imply
   that Python 3.10/3.11 runtime support has been removed while those interpreters
   remain in the CI test matrix.

This ADR is a **decision record only** — it does not modify `pyproject.toml`, the
CI matrix, or the Trove classifiers.

## Required steps for a future Python 3.12+ baseline PR

A future PR that actually drops Python 3.10/3.11 must update all of the following
in one coherent change:

- `pyproject.toml`:
  - `requires-python = ">=3.12"`
  - remove the Python 3.10 and 3.11 Trove classifiers
- `.github/workflows/ci.yml`:
  - remove the Python 3.10 and 3.11 matrix legs
  - keep at least 3.12, 3.13, and 3.14 while they are active supported baselines
- `.github/workflows/ci-reusable-pilot.yml`:
  - update the generated pilot matrix consistently
  - keep the matrix validation fail-closed
- Documentation:
  - changelog entry under `[Unreleased]`
  - release-audit / release-notes support-policy paragraph
  - README compatibility/installation note if one exists at that point

## Release classification

Dropping interpreter support is user-visible and must not be hidden in a patch
release. Default LiouScope governance classification:

- **PATCH:** bug fixes and fail-closed hardening that keep the same supported
  runtime floor.
- **MINOR:** additive APIs, new diagnostics, and explicitly announced
  baseline-policy updates when no public stable API is broken.
- **MAJOR:** reserved for breaking public API, manifest, or diagnostic-report
  contract changes. If downstream users rely materially on Python 3.10/3.11, the
  project may choose a MAJOR bump for the interpreter drop even though the Python
  packaging ecosystem often handles such drops in MINOR releases.

## Evidence gate

Before a baseline PR is marked ready, it must include:

- CI green on every retained Python version.
- `ruff`, `mypy`, anchor regressions, full suite, and QuTiP cross-checks green.
- A dependency-resolver smoke test from a clean environment.
- A changelog note that distinguishes the SPEC-0 recommendation from upstream
  Python EOL status.

## Consequences

**Positive**

- Closes the runtime-vs-typecheck version split (everything targets ≥ 3.12).
- Fewer CI legs and a smaller dependency-compatibility surface to maintain.
- A clean, auditable rationale for the eventual `requires-python` bump.

**Negative / risks**

- Dropping supported runtimes is **SemVer- and governance-relevant**: it changes
  `requires-python`, the CI matrix, classifiers, dependency floors, and release
  notes. It must land in its **own** PR with a `CHANGELOG.md` migration note.
- Anchor regressions (`tests/test_anchors.py`) must stay green across the new,
  narrower matrix; the drop PR re-verifies them as the sacred gate.

## Non-goals

- **Not** a claim that the upstream CPython project considers 3.10 or 3.11
  unsupported — SPEC 0 is an ecosystem support-window convention.
- No dependency-floor bump by itself, and no change to the `ruff`
  `target-version` (tracked in the enacting release).
- No D24 claim expansion, no manifest-schema bump, no change to the v0.5.0
  archive/citation gates.

## References

- SPEC 0 — Minimum Supported Dependencies: <https://scientific-python.org/specs/spec-0000/>
- `CHANGELOG.md` — v0.5.0 "Fixed": mypy `python_version` raised to 3.12.
- `AGENTS.md` §"Working agreements" — branch-protection / required-check policy.
- Roadmap issue #59 (post-v0.5 architecture & policy items); PRs #61, #64.

[spec0]: https://scientific-python.org/specs/spec-0000/
