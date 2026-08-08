# LiouScope

**Multi-Diagnostic Relaxation Analysis for Open Quantum Lattice Systems**

[![CI](https://github.com/marcohost33-maker/Liouscope/actions/workflows/ci.yml/badge.svg)](https://github.com/marcohost33-maker/Liouscope/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/liouscope)](https://pypi.org/project/liouscope/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21246109.svg)](https://doi.org/10.5281/zenodo.21246109)
[![Citation metadata](https://img.shields.io/badge/citation-CITATION.cff-green)](CITATION.cff)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

LiouScope is a research framework for **time-homogeneous Markovian open quantum systems** described by
Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) generators. It quantifies *when and why the Liouvillian gap
fails as a relaxation-time predictor* and replaces single-number "decay rate" reporting with a layered
diagnostic that surfaces the underlying mechanism.

The library implements the diagnostic schema `D1-D24-Übersicht-v3` and a
**twelve-class mechanism taxonomy A1-A12** (`TAXONOMY_VERSION = "A1-A12-v3.1"`).
**Code-backed today: D1-D20** (the peer-review submission set, plus sub-diagnostics
D2b/D7b/D11b) **and D24** (the opt-in Zhou mixing-time predictor). D21-D23 are
schema-defined post-submission slots that are not yet implemented in this
repository — the layer table below reflects what the code actually computes.

> **Status:** Research / pre-clinical. Not for diagnostic or operational use on production
> hardware. Released `v0.5.0` remains immutable and citable; the default branch reports
> `0.6.0.dev0` from `src/liouscope/_version.py`. See [`CHANGELOG.md`](CHANGELOG.md),
> [`docs/DEVELOPMENT_MIGRATION_0.6.0.dev0.md`](docs/DEVELOPMENT_MIGRATION_0.6.0.dev0.md), and
> [`src/liouscope/MANIFEST_SCHEMA.json`](src/liouscope/MANIFEST_SCHEMA.json).

---

## Why LiouScope

For an open quantum lattice the Liouvillian gap (the smallest non-zero real part of the spectrum of the
GKSL generator) is the textbook proxy for relaxation. In practice the gap is often a poor predictor —
non-normality, anomalous Mpemba effects, transient blow-up, classical sub-radiance, and other mechanisms
distort the relationship between gap and observed relaxation by orders of magnitude.

LiouScope addresses this with three deliberate choices:

1. **No single number.** Every analysis returns a structured `DiagnosticReport` with the code-backed
   diagnostic set grouped by physical concept, not a scalar.
2. **Explicit uncertainty.** Bootstrap CIs (BCa), GLS with AR(1) residuals, AICc model selection, and a
   parametric-bootstrap pipeline are first-class — not optional add-ons.
3. **Auditable manifests.** Each run produces a SHA-256-stable JSON manifest (`MANIFEST_SCHEMA v1.5.0`)
   that captures the seed, framework/schema/taxonomy versions, Python/NumPy/SciPy versions,
   platform, solver path, quality label and a run-invariant `input_hash`. Structured ensemble
   evidence, when supplied, is bound into that hash by its canonical digest.

---

## Quick start

```bash
pip install -e .                # editable install from this repository
pytest -q                       # full test suite, target: all green
python examples/quickstart.py   # dephased-qubit demo with QuTiP cross-check
```

```python
import numpy as np
import liouscope as lp

# Dephased qubit: H = (1/2) sigma_x, jump = sigma_z, gamma = 0.3
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
L  = lp.build_liouvillian(0.5 * sx, jump_ops=[sz], rates=[0.3])

plus  = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho_0 = np.outer(plus, plus.conj())

report = lp.diagnose(L, rho_initial=rho_0, bootstrap_B=100, seed=42)

# A-class mechanism + 95% BCa CI on the fitted relaxation rate
print(report.classification.a_class)         # one of "A1".."A12"
print(report.relaxation.beta_D, "in",
      report.relaxation.bca_ci_beta)         # (lo, hi)
```

The same example with a 1D lattice geometry:

```python
from liouscope.core import (
    boundary_dephasing_jumps,
    heisenberg_xxz_hamiltonian,
    one_d_chain,
)

lattice = one_d_chain(n=3)                                # 3 qubits, open boundary
H       = heisenberg_xxz_hamiltonian(lattice, J=1.0, Delta=0.5)
jumps   = boundary_dephasing_jumps(lattice.n_sites)
L       = lp.build_liouvillian(H, jumps, rates=[0.25] * len(jumps))
report  = lp.diagnose(L, bootstrap_B=50, seed=42)
```

---

## Diagnostic layers (D1-D24)

The table below is the **code-backed** numbering (module docstrings and
`StabilityReport` keys are the source of truth). D21-D23 are defined in the
Drive-side canon schema (`D1-D24-Übersicht-v3`) but are **not yet implemented
in this repository**; D24 ships as an opt-in module.

| Layer | IDs | What it measures | Key modules |
|---|---|---|---|
| **S — Spectral** | D1-D4 (+D2b) | gap, GNS-symmetrised gap, KMS gap, oscillating-mode gap, spectral spread | `diagnostics/spectral.py` |
| **R — Relaxation** | D5-D7 (+D7b) | von-Neumann entropy, relative entropy, Uhlmann fidelity, entanglement asymmetry | `diagnostics/relaxation.py` |
| **N — Non-normality** | D8-D13 (+D8b, D10b, D11b) | Henrici departure (+ dimensionless `henrici_relative`), Petermann factors, Kreiss grid lower bound (+ scale-relative `kreiss_scaled`), Bohr-AP, resolvent peak/FWHM, pseudospectral radius (+ scale-relative radius & abscissa) | `diagnostics/nonnormality.py`, `diagnostics/resolvent.py` |
| **T — Transient** | D14-D15 | sup-norm transient amplification, numerical-abscissa ratio | `diagnostics/transient.py` |
| **C — Classification** | D16-D20 | LEP proximity, gap-rate consistency, initial-state sensitivity, Mpemba overlap/scaling; A1-A12 mechanism classifier on top | `diagnostics/lep.py`, `diagnostics/mpemba.py`, `diagnostics/classification.py` |
| **U/G — Uncertainty & Governance** | U0-U2, D24 | BCa CIs, AICc model selection M0..M3b, GLS+AR(1); manifest export; Zhou mixing-time predictor (opt-in, frozen) | `fitting/`, `diagnostics/uncertainty.py`, `io/manifest.py`, `_zhou.py` |

The Zhou predictor (D24) is opt-in and lives in `liouscope._zhou`; see CHANGELOG.
Its `claim_status` is **reference-verified-bound-coarser** — the cited reference
(Yi-Neng Zhou, "Universal Predictors for Mixing Time more than Liouvillian Gap",
arXiv:2601.06256, v3 2026-05-20) was independently verified (re-audit
2026-06-04). Our implemented upper bound is in the same family as Zhou's Eq.(16)
and exact in the normal-mode limit, but uses the Petermann (Schatten-2) factor
with a global gap/K_max rather than Zhou's per-mode trace-norm factor `C_j`, so
it is a related, generally coarser surrogate (not a verbatim Eq.(16)). See the
`liouscope._zhou` module docstring and `CLAIM_STATUS` for the exact differences.

---

## Geometries, models, dissipator families

| Concept | Implementations |
|---|---|
| Lattices | 1D chain, 2D square, honeycomb, triangular |
| Hamiltonians | Ising, XY, Heisenberg-XXZ, Bose-Hubbard |
| Dissipator families | bulk, boundary, engineered (5 validation systems V1-V5) |
| Sparse utilities | Low-level ARPACK shift-invert helpers via `liouscope.sparse`; not yet wired into `diagnose()` |

V1-V5 validation systems are exposed as library functions in `liouscope.examples`; see
[`examples/`](examples) for runnable demonstrations.

---

## Project structure

```
src/liouscope/
  core/             Hamiltonians, jump operators, lattices, Lindblad superoperators
  diagnostics/      Six layers (spectral, nonnormality, relaxation, transient,
                    uncertainty, classification, mpemba, lep, resolvent)
  fitting/          GLS+AR(1), BCa bootstrap, AICc model selection
  io/               Run-manifest export, seed control
  sparse/           Low-level ARPACK shift-invert helpers (not yet wired into diagnose())
  _zhou.py          Zhou universal mixing-time predictor (opt-in, D24)
benchmarks/         Heisenberg scaling, paper-reproduction harness
examples/           quickstart.py, jc_ep_sweep.py, qutrit_v1.py
figures/            Fig1..Fig3 paper plot pipeline
tests/              anchors, numerics, sparse, classification, fitting, V1-V5 reference
```

The repo follows the **src/-layout** convention; install editable with `pip install -e .` and
imports resolve from `src/liouscope/...`.

---

## Reproducibility and provenance

LiouScope is built for paper-grade reproducibility:

- **Seeded everywhere.** `liouscope.io.seed.seed_everything()` pins Python `random`, NumPy's
  global state, `PYTHONHASHSEED`, and returns a dedicated `np.random.Generator` for all
  subsequent draws (see [`tests/conftest.py`](tests/conftest.py)). SciPy draws through NumPy;
  BLAS threading is *not* controlled — bit-exactness across BLAS builds is out of scope.
- **Run manifests are SHA-256 stable.** Every diagnostic run can emit a JSON manifest
  (`io.dump_manifest(report, path)`) validated against
  [`MANIFEST_SCHEMA.json`](src/liouscope/MANIFEST_SCHEMA.json) (schema v1.5.0) via
  `io.validate_manifest(payload)`. The validator uses a cached
  `jsonschema.Draft202012Validator` when [`jsonschema`](https://python-jsonschema.readthedocs.io/)
  is installed, and falls back to a built-in subset check otherwise. **Two runs with the same
  seed and inputs produce manifests that are byte-identical except for the recorded wall-clock
  `timestamp`** — the `run_id` and `input_hash` fields are run-invariant (derived purely from
  inputs, seed and framework version, never from the clock). For *fully* byte-identical
  manifests (timestamp included), set the reproducible-builds standard env var
  `SOURCE_DATE_EPOCH` to a fixed Unix timestamp; the manifest writer then uses it instead of the
  wall clock. Both properties are gated by tests in `tests/test_manifest.py`
  (`test_manifests_byte_identical_modulo_timestamp`,
  `test_manifests_byte_identical_with_source_date_epoch`).
- **Anchor tests.** `tests/test_anchors.py` locks the numerical anchors that paper figures depend on;
  changes to physics code that move these values are caught in CI.
- **Paper-figure pipeline.** `figures/generate_all.py` regenerates Fig 1-3 deterministically.

---

## How to cite

Please cite via the [`CITATION.cff`](CITATION.cff) file. GitHub's "Cite this repository" widget reads
it automatically. The preferred citation is the methods paper:

> Coworker Research (2026). *Beyond the Liouvillian Gap: Multi-Diagnostic Relaxation Analysis for
> Open Quantum Lattice Systems.* Preprint.

If you cite the software directly, please include the version (`liouscope.__version__`) and the
TAXONOMY_VERSION (`A1-A12-v3.1`) so the mechanism labels are unambiguous.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow. In short: open an issue
first for non-trivial changes, run `pytest -q` locally, keep diagnostic IDs and taxonomy version
constants stable. Security-relevant findings → [`SECURITY.md`](SECURITY.md).

This project follows the SemVer convention and uses the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

**Getting help.** Questions, bug reports and feature requests all go through
[GitHub issues](https://github.com/marcohost33-maker/Liouscope/issues) — use the
bug-report / feature-request templates where they fit, or a blank issue for
usage questions. There is no separate mailing list or chat channel. Maintainer
decisions (releases, taxonomy/schema versioning, anchor changes) are made by the
repository owner; substantial methodology changes are documented in
`CHANGELOG.md` and, where architectural, as ADRs under `docs/adr/`.

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE). © Coworker Research / Coworkerz.

---

## Status disclaimer

LiouScope is a **research framework**. It does not constitute a medical device, a diagnostic tool,
a clinical decision aid, or an instrument fit for safety-of-life applications. Numerical results
require physics-domain interpretation; no claim of universality is made beyond the regimes
covered by the V1-V5 validation systems.

Two honesty notes on the classifier surface (tracked in issues #101/#102):
`classification.confidence` is a deterministic **heuristic support score**,
not a calibrated probability; and the A10/F5 (phantom-relaxation) verdict
path is **not yet invariant under a change of rate units** — the
scale-relative successor diagnostics (`henrici_relative`, `kreiss_scaled`,
`pseudospectral_radius_rel`, `pseudospectral_abscissa_rel`) are computed on
every run with `claim_status: pending` but are advisory-only until the
preregistered calibration study and independent physics review complete.
See `docs/explanation/layers-and-taxonomy.md` for details.
