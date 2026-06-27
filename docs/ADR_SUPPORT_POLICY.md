# ADR: Support policy after v0.5.0

Date: 2026-06-27
Status: Accepted for policy; runtime-baseline change deferred
Scope: LiouScope packaging, CI matrix, dependency floors, release governance

## Context

LiouScope is a scientific Python package with a deliberately broad v0.5.0 runtime
matrix (`python_requires >=3.10`; CI on Python 3.10, 3.11, 3.12, 3.13, 3.14).
Issue #59 asks for an explicit support policy rather than silently changing the
runtime baseline inside an unrelated hardening PR.

The relevant external policy anchor is SPEC 0, which recommends dropping support
for older Python feature releases three years after initial release and core
package dependencies two years after initial release. SPEC 0 is a Scientific
Python ecosystem recommendation, not a claim that upstream Python itself is EOL.

## Decision

1. **Do not drop Python 3.10 or 3.11 in v0.5.x.** The v0.5.0 public-release lane
   remains focused on archiving, publication, citation metadata, and evidence
   locking. Keeping the existing runtime matrix avoids mixing user-visible
   support changes into a release-publication gate.
2. **Adopt SPEC 0 as the forward support-policy reference.** Future baseline PRs
   must compare Python and core dependency floors against SPEC 0 and document any
   intentional deviation.
3. **First candidate post-v0.5 baseline:** Python 3.12+ with CI on 3.12, 3.13,
   and 3.14. This is a candidate, not an automatic change. It requires a dedicated
   PR and release note.
4. **No silent dependency-floor bumps.** Raising NumPy/SciPy floors must be done
   together with CI evidence, resolver checks, and a compatibility note. A floor
   may remain older than SPEC 0 temporarily if needed for user compatibility, but
   the deviation must be explicit.
5. **Mypy target is not the runtime floor.** The current `mypy` target of Python
   3.12 is a toolchain workaround for modern NumPy stubs. It does not imply that
   Python 3.10/3.11 runtime support has been removed while those interpreters stay
   in the CI test matrix.

## Required steps for a future Python 3.12+ baseline PR

A future PR that actually drops Python 3.10/3.11 must update all of the following
in one coherent change:

- `pyproject.toml`:
  - `requires-python = ">=3.12"`
  - remove Python 3.10 and 3.11 Trove classifiers
- `.github/workflows/ci.yml`:
  - remove Python 3.10 and 3.11 matrix legs
  - keep at least 3.12, 3.13, and 3.14 while they are active supported baselines
- `.github/workflows/ci-reusable-pilot.yml`:
  - update the generated pilot matrix consistently
  - keep the matrix validation fail-closed
- Documentation:
  - changelog entry under `[Unreleased]`
  - release audit / release notes support-policy paragraph
  - README compatibility badge or installation note if one exists at that point

## Release classification

Dropping interpreter support is user-visible. It must not be hidden in a patch
release. For LiouScope governance, the default classification is:

- PATCH: bug fixes and fail-closed hardening that keep the same supported runtime
  floor.
- MINOR: additive APIs, new diagnostics, and explicitly announced baseline-policy
  updates when no public stable API is broken.
- MAJOR: reserved for breaking public API, manifest, or diagnostic-report contract
  changes. If downstream users rely materially on Python 3.10/3.11, the project
  may choose a MAJOR bump for the interpreter drop even if the Python packaging
  ecosystem often handles such drops in MINOR releases.

## Evidence gate

Before a baseline PR is marked ready, it must include:

- CI green on every retained Python version.
- `ruff`, `mypy`, anchor regressions, full suite, and QuTiP cross-checks green.
- A dependency resolver smoke test from a clean environment.
- A changelog note that distinguishes SPEC-0 recommendation from upstream Python
  EOL status.

## Non-goals

- No D24 claim expansion.
- No manifest schema bump.
- No change to v0.5.0 archive/citation gates.
- No claim that Python 3.10 or 3.11 is upstream-EOL merely because it is outside
  the SPEC-0 recommended Scientific Python support window.
