# LiouScope

**Multi-Diagnostic Relaxation Analysis for Open Quantum Lattice Systems**

[![CI](https://github.com/marcohost33-maker/Liouscope/actions/workflows/ci.yml/badge.svg)](https://github.com/marcohost33-maker/Liouscope/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![DOI / citation](https://img.shields.io/badge/cite-CITATION.cff-green)](CITATION.cff)

LiouScope is a research framework for **time-homogeneous Markovian open quantum systems** described by
Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) generators. It quantifies *when and why the Liouvillian gap
fails as a relaxation-time predictor* and replaces single-number "decay rate" reporting with a layered
diagnostic that surfaces the underlying mechanism.

The library implements **twenty diagnostics D1-D20** organised in six layers
(Spectral / Non-normality / Relaxation / Uncertainty / Classification / Governance) and a
**twelve-class mechanism taxonomy A1-A12** (`TAXONOMY_VERSION = "A1-A12-v3.1"`).

> **Status:** v0.2.0 released 2026-04-17. Research / pre-clinical. Not for diagnostic or operational use
> on production hardware. See [`CHANGELOG.md`](CHANGELOG.md) for version history and
> [`src/liouscope/MANIFEST_SCHEMA.json`](src/liouscope/MANIFEST_SCHEMA.json) for the run-manifest contract.

---

## Why LiouScope

For an open quantum lattice the Liouvillian gap (the smallest non-zero real part of the spectrum of the
GKSL generator) is the textbook proxy for relaxation. In practice the gap is often a poor predictor —
non-normality, anomalous Mpemba effects, transient blow-up, classical sub-radiance, and other mechanisms
distort the relationship between gap and observed relaxation by orders of magnitude.

LiouScope addresses this with three deliberate choices:

1. **No single number.** Every analysis returns a structured `DiagnosticReport` with twenty diagnostics
   grouped by physical concept, not a scalar.
2. **Explicit uncertainty.** Bootstrap CIs (BCa), GLS with AR(1) residuals, AICc model selection, and a
   parametric-bootstrap pipeline are first-class — not optional add-ons.
3. **Auditable manifests.** Each run produces a SHA-256-stable JSON manifest (`MANIFEST_SCHEMA v1.2.0`)
   that captures seeds, library versions, lattice geometry, dissipator family, and full result graph.

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
print(report.classification.a_class)         # e.g. "A1"
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

## Diagnostic layers (D1-D20)

| Layer | IDs | What it measures | Key modules |
|---|---|---|---|
| **S — Spectral** | D1-D4 | gap, sub-gap density, spectrum shape, dissipative-gap separation | `diagnostics/spectral.py` |
| **N — Non-normality** | D5-D8 | Henrici departure, pseudospectrum, condition number, Schur diagonal | `diagnostics/nonnormality.py` |
| **R — Relaxation** | D9-D13 | tau-effective, transient amplification (D11), early-time slope, GLS fit | `diagnostics/relaxation.py`, `diagnostics/transient.py` |
| **U — Uncertainty** | D14-D17 | BCa CIs, AICc model selection M0..M3b, N_eff via Geyer IPS, bootstrap-pivot | `fitting/aicc.py`, `fitting/bootstrap.py`, `fitting/gls.py` |
| **C — Classification** | D18-D22 | A1-A12 mechanism classifier, Mpemba detector, LEP scan, resolvent peaks | `diagnostics/classification.py`, `diagnostics/mpemba.py`, `diagnostics/lep.py`, `diagnostics/resolvent.py` |
| **G — Governance** | D23-D24 | manifest export, Zhou universal mixing-time predictor (opt-in, frozen) | `_zhou.py`, `MANIFEST_SCHEMA.json` |

The Zhou predictor (D24) is opt-in and lives in `liouscope._zhou`; see CHANGELOG.
Its `claim_status` is **pending/unverified** — the cited reference
(arXiv:2601.06256) could not be independently verified (audit 2026-06-04), so
D24 is an exploratory diagnostic only. See `liouscope._zhou.CLAIM_STATUS`.

---

## Geometries, models, dissipator families

| Concept | Implementations |
|---|---|
| Lattices | 1D chain, 2D square, honeycomb, triangular |
| Hamiltonians | Ising, XY, Heisenberg-XXZ, Bose-Hubbard |
| Dissipator families | bulk, boundary, engineered (5 validation systems V1-V5) |
| Sparse path | ARPACK shift-invert via `liouscope.sparse` for d up to 128 |

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
  sparse/           ARPACK shift-invert path (d up to 128)
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

- **Seeded everywhere.** `liouscope.io.seed.seed_everything()` controls NumPy, SciPy, Python
  random, and the BLAS thread-pool (see [`tests/conftest.py`](tests/conftest.py)).
- **Run manifests are SHA-256 stable.** Every diagnostic run can emit a JSON manifest
  (`io.dump_manifest(report, path)`) validated against
  [`MANIFEST_SCHEMA.json`](src/liouscope/MANIFEST_SCHEMA.json) (schema v1.2.0) via
  `io.validate_manifest(payload)`. The validator uses a cached
  `jsonschema.Draft202012Validator` when [`jsonschema`](https://python-jsonschema.readthedocs.io/)
  is installed, and falls back to a built-in subset check otherwise. Two runs on the same
  hardware with the same seed produce byte-identical manifests.
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

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE). © Coworker Research / Coworkerz.

---

## Status disclaimer

LiouScope is a **research framework**. It does not constitute a medical device, a diagnostic tool,
a clinical decision aid, or an instrument fit for safety-of-life applications. Numerical results
require physics-domain interpretation; no claim of universality is made beyond the regimes
covered by the V1-V5 validation systems.
