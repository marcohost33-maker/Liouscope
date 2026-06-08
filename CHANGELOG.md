# Changelog

All notable changes to LiouScope are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `numerics.resolvent.resolvent_norm` large-matrix branch (n > 128): the
  SuperLU power-iteration computed the wrong conjugate-transpose for the
  resolvent. `lu.solve(y.conj()).conj()` evaluates `conj(A)^{-1} y`, which
  equals the required `(A^H)^{-1} y` only for symmetric `A`; on the non-normal
  Liouvillians this kernel targets the returned `||(zI - L)^{-1}||_2` was wrong
  by ~50% (e.g. 0.550 vs the dense reference 1.181 on a random non-symmetric
  matrix). Fixed to solve the LU's conjugate-transpose system directly via
  `lu.solve(y, trans="H")`; the power-iteration estimate now matches a dense
  SVD reference to < 1e-6 relative across seeds. The function is a public
  `numerics` utility and is not on the `diagnose()` report path, so no anchor
  or `DiagnosticReport` output changes (the small-matrix dense branch, used for
  Hilbert dimensions up to 128, was already correct). The power-iteration
  convergence guard was also tightened to 200 iterations / 1e-12 (from 80 / 1e-9,
  adopting PR#39's stricter bound; Trefethen & Embree) for more robust convergence
  near degenerate singular values — still far below the 1e-6 test tolerance.

### Added
- Regression tests pinning the A1–A12 mechanism classifier decision tree
  (`tests/test_classification.py`): synthetic-input coverage of every
  `_pick_a_class` branch, the `_pick_verdict_tier` thresholds (including the
  `EXCLUDED` and `UNDEFINED` paths unreachable through natural confidence
  values), and the `_confidence` scoring rules. Raises `classification.py`
  coverage from 71% to 100%.
- Tests for the previously-untested large-matrix branches of
  `numerics.resolvent` (`tests/test_numerics.py`): the SuperLU sparse solve
  (n > 256) and the power-iteration resolvent norm (n > 128, the regression
  guard for the fix above).

## [0.4.0] — 2026-06-07

Release cut: everything below shipped on `main` between `v0.3.0` (2026-05-28)
and this tag (18 commits, PRs #34–#41 — audit waves 2026-06-04/06-06 + D14
physics-scaling). MINOR bump per SemVer: backward-compatible API additions
(`gap=` forwarding, `TransientGridWarning`, `SOURCE_DATE_EPOCH`).

### Fixed
- D14 transient time-grid physics-scaling (audit 2026-06-06, P2-1):
  `diagnostics.transient.trans_amplitude_ratio` (`sup_t ||e^{tL}||_2`)
  previously used a fixed coarse grid `linspace(0.01, 5.0, 30)`. For systems
  whose relaxation timescale `1/Delta` falls outside `[0.01, 5]` — i.e. the
  small-gap, strongly non-normal regime the diagnostic is meant to detect — the
  supremum was silently *underestimated* (measured 27.8% too low on an
  amplitude-damping channel with gap 0.01: fixed grid 1.0209 vs the true
  sqrt(2) = 1.4142, recovered by a dense oracle). The grid is now physics-scaled
  to the spectral gap (D1) and the numerical abscissa: a two-scale window
  `[0, ~8/Delta]` with a log-spaced early segment (to resolve sharp growth
  peaks) plus a linear late segment. Across damping rates spanning three orders
  of magnitude the auto-scaled D14 now matches a dense reference to < 1e-3
  relative. An explicit `t_grid=` still overrides the scaling (backward
  compatible); when no gap and no grid are given, the legacy coarse grid is
  kept as a fallback. A `TransientGridWarning` is emitted when the propagator
  norm is still rising non-negligibly at the right grid edge (the returned sup
  is then a lower bound). `compute_transient_layer` forwards the gap so
  `diagnose()` benefits automatically. New tests in `tests/test_transient.py`
  pin the behaviour against a dense brute-force oracle; the QuTiP physics-kernel
  parity cross-checks remain green (the change does not touch the Lindblad
  builder, steady state, or spectrum).
- Version single-source (audit 2026-06-06, P0): `pyproject.toml` no longer
  carries a second hard-coded version literal. `[project]` now declares
  `dynamic = ["version"]` and `[tool.setuptools.dynamic]` reads the version from
  `src/liouscope/_version.py` (the documented single source). Previously
  `pyproject.toml` said `0.3.0` while `_version.py` (= runtime
  `liouscope.__version__` and every manifest's `framework_version`) said
  `0.2.0`, so a built wheel reported the wrong version and every run manifest
  recorded a wrong `framework_version` — a provenance bug for a tool that claims
  paper-grade reproducibility. `_version.py` is bumped to `0.3.0`; a new test
  (`test_version_single_source`) asserts
  `importlib.metadata.version("liouscope") == liouscope.__version__`.
- Reproducibility claim precision (audit 2026-06-06, P1): the README claim that
  two runs produce "byte-identical manifests" was empirically false — the
  embedded wall-clock `timestamp` varies, so the manifest SHA differed each run.
  README now states the true property (byte-identical *except for* the recorded
  `timestamp`; `run_id`/`input_hash` are run-invariant) and both properties are
  gated by tests.
- Diagnostic-count consistency (audit 2026-06-06, P1): README headline said
  "twenty diagnostics D1-D20" while its own table and the code constant
  `DIAGNOSTIC_SCHEMA_VERSION = "D1-D24-Übersicht-v3"` run to D24. README and
  AGENTS.md now consistently say "24 diagnostics D1-D24 (D1-D20 submission set;
  D21-D24 post-submission)".

### Added
- `SOURCE_DATE_EPOCH` support in the manifest writer (audit 2026-06-06, P1):
  when this reproducible-builds standard env var is set to a fixed Unix
  timestamp, the manifest's `timestamp` field uses it instead of the wall
  clock, making the manifest *fully* byte-identical across runs. An
  unparseable/negative value is rejected fail-closed (no silent fallback to the
  wall clock). Gated by `test_manifests_byte_identical_with_source_date_epoch`.
- Boundary-guard hardening wave (audit 2026-06-06): fail-closed input
  validation and structured IO errors at the public surface. No API breaks, no
  new mandatory dependencies, anchor regressions unchanged.
  - `numerics.linalg.require_finite_square_2d`: a reusable boundary validator
    that rejects non-finite (NaN/inf), non-square, or empty operators with a
    structured, argument-named `ValueError` (e.g. `"L_super contains
    non-finite entries (1 NaN, 0 inf)"`). Previously such inputs flowed into
    `scipy.linalg.expm`/`svd` and surfaced as opaque, location-blind LAPACK
    messages (`"array must not contain infs or NaNs"` / `"SVD did not
    converge"`) that named neither the offending argument nor the real defect.
  - `diagnose()` now validates `L_super` and any caller-supplied `rho_initial`
    / `rho_steady_state` (finiteness + shape match against `d`) at the entry
    boundary, before any numerics run. The previously uncaught path was
    supplying `rho_steady_state` (which bypasses the steady-state SVD), where an
    inf/NaN in `L_super` was only caught deep inside `expm`.
  - `io.export.load_report` fails closed with structured, path-bearing errors:
    a missing path raises `FileNotFoundError("report file not found: ...")`;
    malformed JSON or a non-object top level raises `ValueError` naming the
    file. Replaces the raw `json.loads(Path(path).read_text(...))` that gave
    callers no context (the "exists()-then-read-raw" fail-open class).
  - `io.export.dump_report` and `io.manifest.dump_manifest` now create missing
    parent directories (`parents=True, exist_ok=True`) so nested artefact paths
    do not fail with a bare `FileNotFoundError` on first write.
  - `tests/test_input_guards.py` (12 tests) and `tests/test_export.py` (8
    tests): negative/edge inputs (NaN, inf, empty, wrong shape, missing file,
    malformed/non-object JSON) fail before the happy path; valid input is not
    rejected (no false positives). Closes the prior coverage gap for
    `io.export`.
- Statistics-hardening wave (audit 2026-06-04, findings S1-S6):
  - `core.lindblad.DegenerateSteadyStateError` + `steady_state(allow_degenerate=)`
    (S1): a multi-dimensional Liouvillian null space (non-unique NESS /
    decoherence-free subspace) now fails closed instead of silently returning
    an arbitrary representative; opt in via `allow_degenerate=True` for one
    representative with a `RuntimeWarning`.
  - `fitting.neff.ar1_correlation_corrected` (S2): small-sample bias-corrected
    lag-1 autocorrelation `(rho*(n-1)+1)/(n-3)` (Marriott-Pope/Kendall;
    arXiv:2010.05870) + `RuntimeWarning` at `n <= 40`. Wired into `fit_gls_ar1`
    so AR(1)-whitened CIs are no longer over-confident at small `n`.
  - Two exact analytic anchors in `tests/test_anchors.py` (anti-circularity):
    pure-dephasing Liouvillian gap = `2*gamma`, amplitude-damping coherence
    decay rate = `gamma/2`, both asserted to `atol=1e-12`.
  - `.github/workflows/ci-qutip.yml`: dedicated job that installs the QuTiP
    extra and runs the QuTiP cross-checks **without skipping** (`pytest -m
    qutip`), plus a guard against a vacuous 0-collected run. Closes the gap
    where cross-family validation only ever skipped in CI. (Required-check
    registration is done separately by a maintainer.)
  - `_zhou.CLAIM_STATUS` / `_zhou.CLAIM_REFERENCE` (S6): the D24 Zhou predictor
    is marked `pending`/unverified because its cited reference
    (arXiv:2601.06256) could not be independently verified at audit time.
    README + module docstring annotated accordingly.
- `_consts.EPS_DIV`: canonical division-by-zero floor (1.0e-300) shared by the
  Petermann inner-product guard in `diagnostics.nonnormality.petermann_factors`
  and `_zhou`. Replaces the previously hard-coded `1.0e-300` magic numbers.
  Deliberately distinct from the physics-scale `EPS_GAP`/`EPS_SUPP`.
- `tests/test_zhou.py`: a closed-form anchor for the D24 Zhou predictor
  (single-qubit pure dephasing, gap = 1, K = 1 -> both bounds = log(1/eps))
  plus a defective-mode guard test (a near-defective mode must not poison the
  finite upper bound).
- `fitting.prony._default_seed` and guarded `prony_seed`: the Prony seed now
  catches `LinAlgError`/`ValueError` from `lstsq`/`np.roots` on near-singular
  Hankel data (e.g. all-NaN/inf signals), emits a `RuntimeWarning`, and falls
  back to a safe default seed with a strictly positive amplitude. New
  regression tests in `tests/test_fitting.py` (fails-before on the pre-guard
  code, which raised `LinAlgError` on non-finite input).

### Changed
- `fitting.bootstrap.bca_ci` (S3, S4): the BCa bias-correction `z0` now uses
  the Efron-1987 half-correction for ties (`mean(<) + 0.5*mean(==)`) instead of
  a strict `<` (which drove `z0 -> -inf` when bootstrap replicates equalled
  `theta_hat`); CI endpoints now use linear quantile interpolation
  (`np.quantile`) instead of granular nearest-rank.
- CI `mypy src/liouscope` is now an enforcing gate (`continue-on-error` removed);
  the previously type-blind 18 `mypy` findings were fixed (annotated numpy
  return locals, `is_dataclass` instance narrowing). No behaviour change.
- `_zhou`: zero-eigenvalue and division-by-zero thresholds now use the canonical
  `EPS_GAP`/`EPS_DIV` constants instead of inline `1.0e-10`/`1.0e-300`.
- `_zhou.CLAIM_STATUS` (S6 re-audit 2026-06-04): `pending`/unverified ->
  `reference-verified-bound-coarser`. The cited reference was independently
  verified against the arXiv PDF: Yi-Neng Zhou, "Universal Predictors for
  Mixing Time more than Liouvillian Gap", arXiv:2601.06256 (v3 2026-05-20,
  University of Geneva). The placeholder title and `UNVERIFIED` marker are
  replaced with the real title/author/version. The implemented upper bound is
  in the same family as Zhou's central result Eq.(16) and exact in the
  normal-mode limit (pure-dephasing anchor), but is a *related, generally
  coarser* surrogate: it uses the Petermann (Schatten-2) factor `sqrt(K)`
  rather than Zhou's per-mode trace-norm factor `C_j = ||rho_j||_1 *
  ||sigma_j||_op`, a single global `gap`/`K_max` instead of a per-mode maximum
  of `(1/lambda_j) log(N_mode C_j)`, and omits the `N_mode` factor. Differences
  documented exactly in the `_zhou` module docstring; `CLAIM_REFERENCE`,
  README, and the status-lock test updated accordingly. (No formula change.)
- `fitting.neff.ar1_correlation_corrected` docstring hardened (no formula
  change): added a validity note (first-order Kendall / Marriott-Pope
  correction; reliable up to `rho ~ 0.85`, residual bias beyond that set by the
  truncation order) and full references (Marriott & Pope 1954; Kendall 1954;
  arXiv:2010.05870; Dou et al. 2026, Br. J. Math. Stat. Psychol.). Documents
  that a higher-order Kendall variant tested marginally better at high
  `rho`/`n` but was deliberately not adopted to keep the audit formula
  bit-stable.

## [0.3.0] - 2026-05-28

### Added
- `liouscope._zhou`: Zhou universal mixing-time predictor (D24) as an opt-in
  diagnostic. Frozen `ZhouPredictorResult` dataclass.

### Changed
- `MANIFEST_SCHEMA.json` moved from the repo root into
  `src/liouscope/MANIFEST_SCHEMA.json`. `pyproject.toml` already listed it
  under `[tool.setuptools.package-data]`, but the file was not actually at
  that path, so wheels built from PyPI shipped without it. The schema is
  now correctly bundled (verified by inspecting the wheel) and loaded via
  `importlib.resources` so the lookup works under editable, wheel and
  zipfile installs.
- `validate_manifest` now uses a cached
  `jsonschema.Draft202012Validator` (per the python-jsonschema performance
  guidance) instead of the autodetecting `jsonschema.validate` convenience
  wrapper. The bundled schema declares `draft/2020-12`, so this is the
  matching validator class.
- `_utc_now_iso()` uses the idiomatic `isoformat(timespec="microseconds").replace("+00:00", "Z")`
  pattern. Pinning `timespec` guarantees a fixed-width 27-character
  timestamp string on every call, eliminating the case where
  zero-microsecond timestamps would lose the fractional component.
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


### Security & CI Hardening (Welle 4)
- `zizmor` workflow security audit added (SHA-pinned `zizmorcore/zizmor-action@v0.5.6` via `5f14fd08...`).
- All Actions SHA-pinned (`actions/checkout@v4.2.2`, `setup-python@v6.2.0`, etc) — Welle G Gold-Standard pattern.
- Dependabot configured (weekly grouped pip + github-actions, 7-day cooldown against Shai-Hulud-style supply-chain attacks, May 2026).
- OpenSSF Scorecard workflow (private-repo SARIF guard + workflow_dispatch).
- Tier-2.5 Branch Protection: `enforce_admins=true`, `required_conversation_resolution=true`, `dismiss_stale_reviews=true` (solo-dev pattern, `required_approving_review_count=0`).
- `delete_branch_on_merge=true`.
- `.gitattributes` (eol=lf) for cross-platform consistency.

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
