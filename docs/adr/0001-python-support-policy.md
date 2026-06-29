# ADR 0001 — Scientific-Python support policy (SPEC 0)

- **Status:** Accepted
- **Date:** 2026-06-26
- **Deciders:** LiouScope maintainers
- **Supersedes:** —
- **Superseded by:** —

## Context

LiouScope currently declares `requires-python = ">=3.10"` and runs its CI test
matrix on Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14, with Trove classifiers to
match. As the project moves past `v0.5.0`, two pressures push toward narrowing
the supported runtime window:

1. **Toolchain drift.** NumPy ≥ 2.5 ships type stubs that use the PEP 695 `type`
   statement; mypy only accepts them when its `python_version` target is ≥ 3.12,
   which is why the repo-wide mypy gate already runs at `python_version = 3.12`
   (see `CHANGELOG.md`, v0.5.0 "Fixed"). Keeping 3.10/3.11 as *runtime* targets
   while type-checking at 3.12 is a sustainable but widening split.
2. **Maintenance cost.** Every supported minor multiplies CI minutes, the
   dependency-floor compatibility surface, and the set of behaviours the anchor
   regressions must hold invariant.

The [Scientific Python ecosystem][spec0] addresses exactly this with
**SPEC 0 — Minimum Supported Dependencies**, which recommends:

- support each **Python** minor for **3 years** after its initial release, and
- support each **core package** (NumPy, SciPy, …) for **2 years** after its
  release.

Under SPEC 0's published drop schedule (verified 2026-06-29): Python 3.10
(released 2021-10-04) and 3.11 (released 2022-10-24, SPEC 0 support window ending
**2025-10-23**) are already past the recommended 3-year window, while Python 3.12
(released 2023-10-02) remains inside it until **2026-10-01**.

## Decision

LiouScope adopts **SPEC 0** as the reference policy for its Python and core
dependency support windows.

**Post-`v0.5.0` target (to be enacted in a dedicated support-policy release, not
as a side-effect of an unrelated change):**

- `requires-python = ">=3.12"`
- CI test matrix: **3.12 / 3.13 / 3.14**
- Drop the 3.10 and 3.11 legs, their Trove classifiers, and any 3.10/3.11-only
  compatibility shims **together** in that one release.
- Re-evaluate NumPy/SciPy floors against the SPEC 0 2-year window at the same
  time (separately reviewed; not mandated by this ADR).

Until that release ships, the **status quo is unchanged**: `>=3.10` and the
five-version matrix remain in force. This ADR records the decision and its
rationale; it does **not** modify `pyproject.toml`, the CI matrix, or classifiers.

## Consequences

**Positive**

- Closes the runtime-vs-typecheck version split (everything targets ≥ 3.12).
- Fewer CI legs and a smaller dependency-compatibility surface to maintain.
- A clean, auditable rationale for the eventual `requires-python` bump.

**Negative / risks**

- Dropping supported runtimes is **SemVer- and governance-relevant**: it changes
  `requires-python`, the CI matrix, classifiers, dependency floors, and release
  notes. It must therefore land in its **own** PR with a `CHANGELOG.md` migration
  note, not be folded into a feature or fix PR.
- Anchor regressions (`tests/test_anchors.py`) must stay green across the new,
  narrower matrix; the drop PR re-verifies them as the sacred gate.

## Non-goals

- This ADR is **not** a claim that the upstream CPython project considers 3.10 or
  3.11 unsupported. SPEC 0 is a *Scientific-Python ecosystem* support-window
  convention; those interpreters continue to receive upstream security fixes on
  their own schedule.
- It does not bump any dependency floor by itself, and it does not change the
  `ruff` `target-version` (tracked separately in the enacting release).

## References

- SPEC 0 — Minimum Supported Dependencies: <https://scientific-python.org/specs/spec-0000/>
- `CHANGELOG.md` — v0.5.0 "Fixed": mypy `python_version` raised to 3.12.
- `AGENTS.md` §"Working agreements" — branch-protection / required-check policy.
- Roadmap issue #59 (post-v0.5 architecture & policy items).

[spec0]: https://scientific-python.org/specs/spec-0000/
