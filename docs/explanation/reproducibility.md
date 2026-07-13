# Auditable reproducibility

LiouScope treats reproducibility as a *contract with an auditor*, not a
best-effort courtesy. The pieces — seeding, SPEC 7 normalisation, manifests,
fail-closed defaults — are individually small; this page explains how they
compose into an audit chain.

## The chain

A published LiouScope result should be checkable by someone with no access to
the original machine. That requires answering four questions mechanically:

1. **What ran?** — `framework_version`, `taxonomy_version`,
   `diagnostic_schema_version` and the `solver_path` in the manifest.
2. **On what?** — the run-invariant `input_hash`, a SHA-256 over the run
   inputs.
3. **With which randomness?** — one integer `seed`, from which every
   stochastic step (bootstrap, jackknife, Haar draws) derives.
4. **Under which environment?** — Python/NumPy/SciPy versions and platform.

The `run_id` binds (2)+(3)+(1) together. None of these fields depends on the
clock; the `timestamp` is the only run-varying field, and even it can be
pinned via `SOURCE_DATE_EPOCH` (the reproducible-builds standard).

## One seed, one field, any input style

SPEC 7 allows callers to pass rich random-state objects (`SeedSequence`,
`Generator`, …). A manifest, however, wants one stable, serialisable value.
LiouScope reconciles the two with a normalisation bridge: whatever arrives
via `rng` is reduced to a **derived integer seed** (`liouscope.derive_seed`),
and that integer is what runs and what the manifest records. The consequence
is worth stating explicitly:

> A manifest alone reproduces the run, regardless of whether the original
> caller used `seed=`, `rng=SeedSequence(...)` or a live `Generator`.

This is why the manifest schema did not need a new field for SPEC 7 support,
and why `seed`-only calls remained byte-identical across the migration.

## Manifests are projections, not documents

`io.dump_manifest` writes a mechanical projection of the report's governance
block — sorted keys, schema-validated, no free-text fields. Two design rules
follow:

- **Never bypass or post-edit the writer.** A hand-crafted manifest is
  indistinguishable from a fabricated one; the artefact's value is that it
  is produced only by the code path it describes.
- **Claim only what is recorded.** Lattice geometry and dissipator family
  are not manifest fields in the current schema, so a manifest does not pin
  them — the input hash *reflects* them, but they are not independently
  auditable fields. Extending the schema requires a version bump plus a
  changelog migration note, precisely so that hash domains stay comparable
  only within one schema version (schema `1.4.0` added the
  ensemble-evidence digest to the hash domain this way; schema `1.5.0`
  switched to a length-framed, injective input-hash encoding).

## Fail-closed as a reproducibility feature

Several guards that look like input validation are really provenance
protection:

- `solver_path="sparse_arpack"` raises `NotImplementedError` instead of
  silently running the dense path — otherwise manifests would record a
  solver that never ran.
- Non-finite eigenvalues in the LEP layer raise instead of propagating NaN
  into fitted quantities.
- `ensemble_confirmation=True` (a bare caller assertion) raises instead of
  being honoured; only typed, digest-bound evidence participates in the
  hash (see {doc}`../how-to/ensemble-evidence`).

The common principle: **a wrong artefact is worse than no artefact.** An
exception is visible and fixable; a subtly untrue manifest silently poisons
every downstream audit.

## Known boundaries

Stated limits, so they are not discovered the hard way:

- Bit-exactness across BLAS builds is out of scope (threading is not
  controlled); reproducibility targets are seed-stable statistics and
  hash-stable manifests, not cross-platform bit-identity of every float.
- The anchor suite (`tests/test_anchors.py`) pins the reference numerics on
  canonical fixtures in CI — anchor changes require a dedicated PR with the
  physics rationale, never a side effect.
