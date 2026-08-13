# Your first diagnostic run

This tutorial walks through one complete LiouScope analysis: build a GKSL
(Lindblad) generator for a driven, dephased qubit, run the full diagnostic
pipeline, and read the report layer by layer. At the end you will know what
`diagnose()` computes and — just as importantly — what it refuses to claim.

## 1. Build the Liouvillian

LiouScope works on the **column-stacking superoperator** representation of the
GKSL generator

$$
\mathcal{L}[\rho] = -i[H, \rho]
  + \sum_k \gamma_k \left( L_k \rho L_k^\dagger
  - \tfrac{1}{2}\{L_k^\dagger L_k, \rho\} \right).
$$

`build_liouvillian` takes the Hamiltonian, the jump operators and their rates
and returns the dense $d^2 \times d^2$ matrix:

```python
import numpy as np
import liouscope as lp

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

H = 0.5 * sx                     # coherent drive
L = lp.build_liouvillian(H, jump_ops=[sz], rates=[0.3])   # dephasing at gamma=0.3
print(L.shape)                   # (4, 4) for a qubit: d=2, d^2=4
```

The convention matches `qutip.liouvillian` (column stacking), which is what
makes the optional QuTiP cross-checks a genuine differential test — see
{doc}`../how-to/qutip-cross-check`.

## 2. Choose an initial state and run `diagnose()`

The initial state matters: several diagnostics (relaxation fits, D18
initial-state sensitivity, D19 Mpemba overlap) are functions of *which*
trajectory you launch. We use the $|+\rangle$ state, which is maximally
sensitive to dephasing:

```python
plus  = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho_0 = np.outer(plus, plus.conj())

report = lp.diagnose(L, rho_initial=rho_0, bootstrap_B=200, seed=42)
```

Notes on the arguments:

- `rho_initial` defaults to the maximally mixed state $I/d$ if omitted.
- `bootstrap_B` controls the parametric bootstrap used for the BCa confidence
  intervals (larger is slower and tighter; 200 is the default).
- `seed` pins every stochastic step. `rng` is the SPEC 7 alternative —
  see {doc}`reproducible-runs`. Passing both raises `ValueError`.
- `solver_path="dense"` is the only implemented path today;
  `"sparse_arpack"` is a reserved manifest value and raises
  `NotImplementedError` instead of silently falling back (fail-closed).

`diagnose()` validates its inputs at the boundary: non-finite or non-square
operators are rejected with a named error before any LAPACK call, and a
dimension mismatch between `L_super` and `rho_initial` fails immediately.

## 3. Read the report, layer by layer

The return value is a frozen `DiagnosticReport`. There is deliberately no
`report.decay_rate` — the library's core thesis is that a single number is not
a faithful summary. Instead the report groups the code-backed diagnostics
D1–D20 (plus sub-diagnostics) into physical layers:

```python
r = report

# S — Spectral layer (D1-D4): where the eigenvalues are
print(r.spectral.gap)          # D1: Liouvillian gap Delta
print(r.spectral.gns_gap)      # D2: GNS-symmetrised gap
print(r.spectral.kms_gap)      # D2b: KMS gap

# N — Non-normality layer (D8-D13): why the spectrum can mislead
print(r.nonnorm.petermann_max) # D9: worst-mode Petermann factor
print(r.nonnorm.kreiss)        # D10: Kreiss constant (transient lower bound)

# R — Relaxation layer (D5-D7) + fits: what the trajectory actually does
print(r.relaxation.aicc_model) # AICc-selected model of the M-hierarchy
print(r.relaxation.beta_D)     # fitted relaxation rate
print(r.relaxation.bca_ci_beta)# 95% BCa confidence interval (lo, hi)

# C — Classification layer (D16-D20): the mechanism verdict
print(r.classification.a_class)    # one of "A1".."A12"
print(r.classification.f_family)   # gap-failure family "F1".."F5" or "none"
print(r.classification.verdict)    # CONFIRMED / CANDIDATE / NOT_EXCLUDED / UNDEFINED
print(r.classification.tier)       # e.g. EXPLORATION vs PUBLICATION_GRADE
print(r.classification.confidence) # heuristic support score 0..1 (NOT calibrated)
```

How to read the classification:

- `a_class` names the *mechanism* (e.g. `A1` gap-controlled, `A8` oscillatory
  transient, `A11` non-normal Mpemba). `A1` means "the gap is a good
  predictor here" and maps to `f_family == "none"`.
- `verdict` is evidence-graded. `CONFIRMED` requires positive certificates;
  `CANDIDATE` and `NOT_EXCLUDED` are weaker; `UNDEFINED` means the run does
  not carry enough evidence to decide — which is a *correct* answer, not a
  failure. A single-state run on a maximally mixed steady state, for example,
  reports A11 as `UNDEFINED`/`EXPLORATION` by design (see
  {doc}`../how-to/ensemble-evidence`).
- The full mechanism catalogue with literature anchors is in
  {doc}`../explanation/layers-and-taxonomy`.

## 4. Check the governance block

Every report carries the metadata needed to reproduce it:

```python
g = report.governance
print(g.run_id[:16])       # run-invariant ID (inputs + seed + version)
print(g.seed)              # the resolved integer seed (42 here)
print(g.input_hash)        # SHA-256 over the run inputs
print(g.framework_version) # liouscope.__version__ at run time
```

Exporting and validating this block as a JSON manifest is covered in
{doc}`../how-to/export-validate-manifest`.

## 5. Where to go next

- Scale up to a lattice: `liouscope.core` provides `one_d_chain`,
  `heisenberg_xxz_hamiltonian` and dissipator families; the README shows a
  3-qubit XXZ chain in four lines.
- The five validation systems V1–V5 used by the test suite are available as
  library functions in `liouscope.examples` (`v1_qutrit()` …
  `v5_jaynes_cummings()`), handy as known-good inputs for experiments.
- {doc}`reproducible-runs` continues this tutorial with seeding and manifests.
