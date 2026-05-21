# Changelog

All notable changes to LiouScope are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] -- v0.2.1 (post-submission tooling)

### Added
- `liouscope._zhou`: Zhou universal mixing-time predictor (D24) as an opt-in
  diagnostic. Frozen `ZhouPredictorResult` dataclass.
- `liouscope.io.manifest_payload`: schema-compliant projection of a
  `DiagnosticReport` that includes `schema_version`, `taxonomy_version`,
  and `diagnostic_schema_version` as documented in `MANIFEST_SCHEMA.json`.
- `liouscope.io.dump_manifest`: writes the manifest payload to a JSON file.
- `liouscope.io.validate_manifest`: validates a manifest dict against
  `MANIFEST_SCHEMA.json`. Uses `jsonschema` when available, falls back to a
  built-in subset check otherwise.

### Fixed
- `liouscope.io.build_manifest` now uses timezone-aware
  `datetime.datetime.now(UTC)`. The previous `datetime.utcnow()` call was
  deprecated in Python 3.12 and slated for removal in 3.14.
- `liouscope._zhou.mixing_time_upper_bound` rescaling formula. The previous
  version contained a no-op (`epsilon / epsilon`) and dropped the `1/gap`
  factor, returning incorrect mixing-time estimates for any `eps` other
  than the original one. `ZhouPredictorResult` now carries the spectral
  gap and Petermann factor used to build it so rescaling is well-defined.
- README quickstart code now uses the actual public API
  (`one_d_chain`, `heisenberg_xxz_hamiltonian`, `boundary_dephasing_jumps`,
  `diagnose(L, rho_initial=...)`, `report.relaxation.beta_D`,
  `report.relaxation.bca_ci_beta`) — the previous snippet referenced
  symbols that did not exist (`Chain1D`, `XXZ`, `tau_eff`, `ci95`).

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
