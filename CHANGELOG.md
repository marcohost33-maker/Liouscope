# Changelog

All notable changes to LiouScope are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] -- v0.2.1 (post-submission tooling)

### Added
- `liouscope._zhou`: Zhou universal mixing-time predictor (D24) as an opt-in
  diagnostic. Frozen `ZhouPredictorResult` dataclass.
- Evidence Pack v1.2 materialised in-repo: `SECURITY.md`,
  `REPRODUCIBILITY.md`, `LIOUSCOPE_EVIDENCE_LOCK_REGISTER.csv`,
  `LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml`.
- Drive-template integrations (2026-05-13): `LIOUSCOPE_BENCHMARK_MANIFEST.yaml`
  (BM-001..003 + BM-003b), `ROADMAP_FLOQUET.md`,
  `LIOUSCOPE_NEGATIVE_RESULTS_REGISTER.md` (NR-001..202), expanded
  `CITATION.cff` references (Mori-Shirai 2023, Zhou 2026, Dinc-Eckardt-
  Schnell 2025).
- `examples.v1b_thermal_qutrit`: detailed-balance qutrit matching BM-003.
- `benchmarks/run.py`: deterministic per-entry runner with SHA-256 fingerprint
  and JSON export.
- `make benchmarks` target running every BM-* manifest entry.
- FAIR4RS gates: persistent identifier policy, reproducibility policy,
  archival deferral notes (SWHID ISO/IEC 18670:2025, Zenodo DOI).
- Security gates: OpenSSF Scorecard workflow, GitHub Dependency Review,
  CodeQL, CycloneDX SBOM.
- Supply-chain hardening in `pypi.yml`: pyproject validation,
  `twine check`, TestPyPI dispatch gate, environment-scoped Trusted
  Publishing.

### Changed
- `numerics.adjoint`: new `metric={"gns", "kms"}` keyword on
  `alicki_adjoint` and `symmetrised_liouvillian`; new
  `gram_matrix(rho, metric)` single source of truth. KMS branch now uses
  the proper `G_KMS = rho^{1/2} (x) rho^{1/2}.conj()` Gram matrix instead
  of the previous GNS-symmetrised generator viewed through the KMS metric.
- `diagnostics.spectral.kms_gap`: rebuilt to use the new metric branch.
  On a non-detailed-balance off-diagonal qutrit (BM-003b) the KMS / GNS
  ratio is now 1.147 (was 1.000 before).
- `diagnostics.spectral._real_gap_from_symmetric`: filters for **negative**
  eigenvalues only. Non-detailed-balance systems can carry isolated
  positive eigenvalues of the symmetrised generator (transient GNS-norm
  amplification); they are not relaxation gaps and were previously
  reported as negative numbers.
- `diagnostics.classification`: A11 verdict now gets demoted to A12 when
  `initial_state_sensitivity > MPEMBA_SENSITIVITY_THRESHOLD` (Mackinnon-
  Paternostro NJP 28, 2026; NR-159). Initial-state sensitivity is added
  to the evidence dict.
- `pyproject.toml`: PEP 639 license expression (`license = "Apache-2.0"`,
  `license-files = ["LICENSE"]`); old License classifier removed.
  Wheel + sdist now pass `twine check`.

### Fixed
- `io/export.py`: narrowed `is_dataclass(value)` check to instances only
  (mypy clean).
- mypy now runs without errors (configuration in `pyproject.toml`).

### Tests
- 158/158 GREEN, coverage 84%.
- New anchor tests for the KMS Gram matrix and the KMS pi-adjoint branch.
- New regression tests in `tests/test_validation_systems/test_v1_qutrit.py`:
  - `test_v1b_thermal_qutrit_kms_equals_gns` (DB: KMS = GNS exactly)
  - `test_offdiagonal_qutrit_kms_above_gns` (non-DB: KMS / GNS in [1.10, 1.20])
- New end-to-end test in `tests/test_classification.py`:
  - `test_classifier_demotes_a11_when_sensitivity_too_high` validates
    both the demotion path and the robust path.

## [0.2.0] -- 2026-04-17

### Added
- Twenty diagnostics D1-D20 organised in six layers S/N/R/U/C/G.
- Twelve-class mechanism taxonomy A1-A12 (`TAXONOMY_VERSION = "A1-A12-v3.1"`).
- Fit hierarchy M0/M1/M2/M3a/M3b with Prony-seed initialisation for M3b.
- Statistical pipeline: GLS with AR(1) residuals, N_eff via Geyer 1992 IPS
  estimator, AICc with N_eff correction, parametric bootstrap with BCa
  confidence intervals.
- Sparse path (`liouscope.sparse`) with ARPACK shift-invert for d up to 128.
- Run manifest with SHA-256 run-id and JSON export. Schema version 1.2.0.
- Four lattice geometries (1D chain, 2D square, honeycomb, triangular) and
  four benchmark Hamiltonians (Ising, XY, Heisenberg-XXZ, Bose-Hubbard).
- Three dissipator families (bulk, boundary, engineered).
- Five validation systems V1-V5 as library functions in `liouscope.examples`.
- Paper figure pipeline.

### Fixed
- FIX-1: GNS Gram matrix is `rho_ss^T (x) I` (not the KMS form).
- FIX-2: Column-stacking via `flatten(order='F')` is enforced everywhere.
- FIX-3: LEP detection includes complex-conjugate eigenvalue pairs.
- FIX-4: Pauli-sector rate is labelled distinctly from `Delta_s`.
- FIX-5: M0 fit uses log-linear regression on D(rho||pi) as baseline.
- FIX-6: Henrici eta_N via Schur decomposition.
- E1-E10: All ten normative patches from the v3 audit applied.

### Correctness anchors enforced (regression tests in `tests/test_anchors.py`)
- A. Column-stacking `order='F'`.
- B. GNS Gram `rho_ss^T (x) I`.
- C. Alicki adjoint direction `L_tilde* = (rho^{-1} (x) I) L (rho (x) I)`.
- D. zgeev (`scipy.linalg.eig`) for non-Hermitian Liouvillian.
- E. SuperLU for resolvent computations.
- F. AICc-only model comparison (M1/M2 are not nested).
- G. Parametric bootstrap on GLS-AR(1) residuals with BCa-CI.
- H. N_eff via Geyer 1992 IPS in AICc small-sample correction.
- I. Conjugate-pair inclusion in LEP proximity.
- J. supp(rho_0) subset supp(rho_ss) check with eps = 1e-12 regularisation.
- K. HS adjoint distinct from pi-weighted adjoint.
- L. `TAXONOMY_VERSION` stamped on every `ClassificationResult`.
- M. D11 = Bohr-AP (Basso 2025), D11b = resolvent peak.
- N. D24 = Zhou (universal mixing-time predictor), not Lee-Bound.
