# Reproducible runs

LiouScope is built for paper-grade reproducibility: every stochastic step is
seeded, and every run can emit a SHA-256-stable JSON manifest that pins what
ran, on what, with which versions. This tutorial covers the three pieces:
seeding, the SPEC 7 `rng` keyword, and the manifest.

## 1. Seeding a run

The stochastic steps inside `diagnose()` (parametric bootstrap, jackknife,
Haar draws in D18) all derive from one integer seed:

```python
import numpy as np
import liouscope as lp

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
L = lp.build_liouvillian(0.5 * sx, jump_ops=[sz], rates=[0.3])

report_a = lp.diagnose(L, seed=123)
report_b = lp.diagnose(L, seed=123)
assert report_a.relaxation.bca_ci_beta == report_b.relaxation.bca_ci_beta
```

If you pass nothing, the default seed is 42 — a no-argument call is still
deterministic. For script-level determinism beyond `diagnose()` there is
`lp.seed_everything()`, which pins Python's `random`, NumPy's global state and
`PYTHONHASHSEED`, and hands you a dedicated `np.random.Generator` for your own
draws. One honest caveat: BLAS threading is *not* controlled, so bit-exactness
across different BLAS builds is out of scope.

## 2. The SPEC 7 `rng` keyword

[SPEC 7](https://scientific-python.org/specs/spec-0007/) is the
Scientific-Python standard for random-state APIs: accept a single `rng`
argument and normalise it via `np.random.default_rng`. LiouScope implements
the additive first phase: `diagnose()`, `seed_everything()` and the D18
surface accept `rng` alongside the legacy `seed`, and passing **both raises**
`ValueError`.

```python
from numpy.random import SeedSequence, default_rng

report = lp.diagnose(L, rng=123)                  # int
report = lp.diagnose(L, rng=SeedSequence(2026))   # SeedSequence
report = lp.diagnose(L, rng=default_rng(7))       # Generator (consumes one draw)
```

Whatever you pass is normalised to a **derived integer seed** by the public
bridge `lp.derive_seed`, and that derived integer is what the manifest
records. This keeps the manifest contract unchanged: a manifest alone still
reproduces the run, whether it was launched with `seed` or `rng`.

Two semantics worth knowing:

- Passing a `Generator` consumes one draw from *your* generator (SPEC 7
  consumption semantics), so successive calls with the same generator object
  give independent runs.
- `seed`-only and no-argument calls are byte-identical to pre-SPEC-7
  behaviour; `seed` remains fully supported in this phase.

## 3. Export the run manifest

Every `DiagnosticReport` can be projected onto the manifest schema
(`src/liouscope/MANIFEST_SCHEMA.json`) and written to disk:

```python
from liouscope import io

report = lp.diagnose(L, rho_initial=None, seed=42)
io.dump_manifest(report, "artefacts/run_manifest.json")

payload = io.manifest_payload(report)   # the same dict, in memory
io.validate_manifest(payload)           # raises on any schema violation
```

The manifest records the seed, framework/schema/taxonomy versions,
Python/NumPy/SciPy versions, platform, solver path, quality label, and two
run-invariant identifiers:

- `input_hash` — SHA-256 over the run inputs,
- `run_id` — derived from `input_hash`, seed and framework version.

Neither depends on the clock, so **two runs with the same inputs and seed
produce manifests that are byte-identical except for the recorded
`timestamp`**. If you need fully byte-identical files (timestamp included),
set the reproducible-builds standard variable `SOURCE_DATE_EPOCH`:

```bash
SOURCE_DATE_EPOCH=1750000000 python my_analysis.py
```

Both properties are pinned by tests in `tests/test_manifest.py`.

## 4. Reproduce from a manifest

To reproduce a published run you need the manifest plus the inputs it hashes
(the Liouvillian construction and initial state — typically your analysis
script at the recorded `framework_version`). The workflow:

1. Install the recorded `framework_version` of LiouScope.
2. Rebuild the inputs and call `diagnose(..., seed=<manifest seed>)`.
3. Check that `report.governance.input_hash` and `run_id` match the manifest —
   if they do, you are provably running the same computation on the same
   inputs; if they differ, the inputs (not just the environment) changed.

One boundary to be aware of: manifest `input_hash` values are comparable only
within one `schema_version`. Schema `1.4.0` added the structured
ensemble-evidence digest to the hash domain (when evidence participates) and
schema `1.5.0` made the input-hash encoding injective via length framing, so
hashes must not be compared across schema versions.

For the design rationale behind the manifest contract, see
{doc}`../explanation/reproducibility`.
