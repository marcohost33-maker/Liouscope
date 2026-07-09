# How to export and validate a run manifest

**Goal:** persist a schema-valid JSON manifest for a diagnostic run, and
verify a manifest you received from someone else.

## Write a manifest

```python
import liouscope as lp
from liouscope import io

report = lp.diagnose(L, seed=42)          # any DiagnosticReport
io.dump_manifest(report, "artefacts/run_manifest.json")
```

`dump_manifest` creates missing parent directories and writes deterministic,
sorted-key JSON. The payload is the schema-compliant projection of the report
— a deliberately small governance subset. If you want the *full* structured
result (all diagnostic values), use `io.dump_report(report, path)` instead;
the manifest is for provenance, the report dump is for data.

## Validate a manifest

```python
import json
from liouscope import io

payload = json.loads(open("artefacts/run_manifest.json").read())
io.validate_manifest(payload)   # raises ValueError on any violation
```

Validation runs against `src/liouscope/MANIFEST_SCHEMA.json`. With the
optional [`jsonschema`](https://python-jsonschema.readthedocs.io/) package
installed (part of the `dev` extra), a cached Draft 2020-12 validator is used;
without it a built-in subset check still enforces required fields and types.

## Check run identity

The two fields that make manifests comparable are run-invariant:

```python
g = report.governance
assert payload["input_hash"] == g.input_hash   # same inputs
assert payload["run_id"] == g.run_id           # same inputs + seed + version
```

- `input_hash` is a SHA-256 over the run inputs (and, since schema `1.4.0`,
  the canonical digest of any structured ensemble evidence that participated).
- `run_id` is derived from `input_hash`, the resolved seed and the framework
  version — never from the clock.

Consequently two runs with identical inputs and seed differ **only** in the
`timestamp` field. For byte-identical files including the timestamp, set
`SOURCE_DATE_EPOCH` (see the {doc}`reproducibility tutorial
<../tutorials/reproducible-runs>`).

## Rules that keep manifests trustworthy

- Do not bypass the manifest writer or post-edit manifest files; the value of
  the artefact is that it is produced mechanically from the report.
- Compare `input_hash` values only within one `schema_version`.
- The manifest records what the code actually used — e.g. `solver_path` is
  `"dense"` today, and `"sparse_arpack"` cannot appear in an honest manifest
  because that path fail-closes with `NotImplementedError`.
- Lattice geometry and dissipator family are **not** manifest fields in the
  current schema; do not claim they are pinned by the manifest.
