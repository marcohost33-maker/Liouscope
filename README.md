# LiouScope

**LiouScope: Beyond the Liouvillian Gap -- Multi-Diagnostic Relaxation Analysis
for Open Quantum Lattice Systems.**

Open-source diagnostic framework for time-homogeneous Markovian open quantum
systems described by Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) generators.
LiouScope quantifies when and why the Liouvillian gap `Delta` fails as a
relaxation-time predictor, providing twenty diagnostics, a twelve-class
mechanism taxonomy, and a reproducibility manifest.

## What LiouScope does (and does not) do

In scope:
- Finite-dimensional Markovian quantum lattice systems in the GKSL framework.
- Relaxation and gap-failure diagnostics (twenty diagnostics, six layers).
- Mechanism classification A1-A12 covering Mori-Shirai overlap, Liouvillian
  skin effect, symmetrised gap, quantum Mpemba effect and phantom relaxation.
- Geometry-resolved benchmarking on four lattices (1D chain, 2D square,
  honeycomb, triangular) and four Hamiltonians (Ising, XY, Heisenberg-XXZ,
  truncated Bose-Hubbard).
- Three dissipator families (bulk, boundary, engineered).

Out of scope:
- Non-Markovian dynamics (no time-convolutionless or memory-kernel methods).
- Floquet open dynamics.
- Thermodynamic-limit theorems (LiouScope is a finite-size diagnostic).
- MLSI / cMLSI verification.

LiouScope is a complementary diagnostic layer, not a Lindblad simulator. A
typical workflow is: simulate with QuTiP or dynamiqs, then run `diagnose()`
on the Liouvillian.

## Installation

```bash
pip install liouscope                 # core
pip install liouscope[qutip]          # add QuTiP for cross-validation
pip install liouscope[figures]        # add matplotlib for plotting
pip install liouscope[dev]            # development extras
```

## Quickstart

```python
import numpy as np
import liouscope as ls

# 1. Build a small open system: dephased qubit.
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
H = 0.5 * sx
jump_ops = [sz]
rates = [0.3]

# 2. Construct the Liouvillian (column-stacking, order='F').
L = ls.build_liouvillian(H, jump_ops, rates)

# 3. Optional QuTiP sanity check (catches column-stacking bugs immediately).
try:
    import qutip
    H_qt = qutip.Qobj(H)
    c_ops = [np.sqrt(rates[0]) * qutip.Qobj(jump_ops[0])]
    L_qt = qutip.liouvillian(H_qt, c_ops).full()
    assert np.allclose(L, L_qt, atol=1e-10), "Column-stacking mismatch!"
    print("Column-stacking OK")
except ImportError:
    pass

# 4. Run the full multi-diagnostic pipeline.
report = ls.diagnose(L)

# 5. Inspect results.
print(f"Liouvillian gap Delta        = {report.spectral.gap:.6f}")
print(f"GNS gap Delta_s              = {report.spectral.gns_gap:.6f}")
print(f"KMS gap                      = {report.spectral.kms_gap:.6f}")
print(f"Petermann factor K_max       = {report.nonnorm.petermann_max:.3e}")
print(f"Kreiss constant              = {report.nonnorm.kreiss:.3e}")
print(f"Classification verdict       = {report.classification.verdict}")
print(f"Mechanism class              = {report.classification.a_class}")
print(f"Gap-failure family           = {report.classification.f_family}")
print(f"Run ID (SHA-256 prefix)      = {report.governance.run_id[:16]}")
```

## Six-layer architecture

| Layer | Letter | Diagnostics | Content |
|---|---|---|---|
| Spectral | S | D1, D2, D2b, D3, D4 | Delta, Delta_s (GNS), KMS-gap, oscillating-mode gap, spread |
| Non-normality | N | D8, D9, D10, D11 | Henrici eta_N, Petermann K, Kreiss, Bohr arithmetic-progression |
| Relaxation | R | D5-D7b, M0-M3b | Von Neumann entropy, relative entropy, fidelity, entanglement asymmetry, fit hierarchy |
| Uncertainty | U | U0, U1, U2 | Fit, solver, system-size uncertainty |
| Classification | C | A1-A12, F1-F5 | Twelve mechanism classes, five gap-failure families |
| Governance | G | SHA-256, version pinning | Run-id, quality tier, reproducibility |

## Mechanism taxonomy

`TAXONOMY_VERSION = "A1-A12-v3.1"` and `DIAGNOSTIC_SCHEMA_VERSION =
"D1-D24-Übersicht-v3-2026-04-24"` are stamped on every result.

| Class | Name | F-family | Reference |
|---|---|---|---|
| A1 | Asymptotic-gap-controlled | F1 (asymptotic) | primitive QMS |
| A2 | Sym-gap-corrected transient | F3 | Mori-Shirai PRL 130, 230404 (2023) |
| A3 | Overlap/eigenvector-amplified | F1 | Mori-Shirai PRL 125, 230604 (2020) |
| A4 | Skin-affected | F2 | Haga PRL 127, 070402 (2021) |
| A5 | Metastable plateau | -- | Macieszczak PRL 116, 240404 (2016) |
| A6 | Accelerated decay / operator spreading | -- | -- |
| A7 | Weak-dissipation singular | -- | Mori PRB 2024 |
| A8 | Oscillatory transient | -- | this work |
| A9 | Prethermalization-affected | -- | -- |
| A10 | Phantom relaxation | F5 | Znidaric arXiv:2306.07876 (2023) |
| A11 | Non-normal Mpemba | F4 | MDPI Entropy 27, 581 (2025) |
| A12 | Mixed / unresolved | -- | -- |

## Reproducibility

Every `diagnose()` call emits a manifest including SHA-256 run-id, framework
version, dependency versions, seed, and quality tier. Use
`liouscope.io.export.dump_report()` to persist a report to JSON, and re-run with
`seed_everything(42)` for bit-identical results on the same platform.

## Citation

```bibtex
@software{liouscope_2026,
  title  = {LiouScope: Multi-Diagnostic Relaxation Analysis for Open Quantum Lattice Systems},
  author = {{Coworker Research}},
  year   = {2026},
  url    = {https://github.com/coworker-research/liouscope},
  version = {0.2.0},
  license = {Apache-2.0}
}
```

`CITATION.cff` and `codemeta.json` provide machine-readable metadata for
GitHub citation and FAIR4RS-compliant indexers.

## Evidence and release gates

`LIOUSCOPE_EVIDENCE_LOCK_REGISTER.csv` and
`LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml` enumerate the FAIR4RS, security
and reproducibility gates. The current status is:

- **P0 evidence lock**: PASS
- **Engineering release-ready**: PASS
- **Public release final**: OPEN (pending GitHub release tag + PyPI Trusted
  Publishing run)
- **Citable archival release**: OPEN (pending Zenodo DOI + Software
  Heritage SWHID per ISO/IEC 18670:2025)

See `SECURITY.md` for the disclosure policy, `REPRODUCIBILITY.md` for the
SHA-256 run-id contract and pinned versions, and `CONTRIBUTING.md` for the
development workflow.

## License

Apache-2.0. See `LICENSE`.
