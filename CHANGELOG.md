# Changelog

All notable changes to LiouScope are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Classifier: A1 now maps to family `"none"`, not `"F1"`.** A1 is the
  asymptotic-gap-controlled control case and therefore should not be stamped as
  the Mori-Shirai overlap gap-failure family. Synthetic branch-regression tests
  now pin both strong and moderate A1 paths to `("A1", "none")`.
- **CI: pinned `ruff==0.15.20` and reformatted the `_diagnostics` import
  block.** Ruff's isort `I001` heuristic for `lines-after-imports` before a
  module-level constant changed between 0.15.14 and 0.15.20, flagging a
  byte-identical, previously-green import block and turning the whole test
  matrix red on a tool release rather than a code defect. Ruff is now pinned
  exactly in both `pyproject` (`dev` extra) and the reusable
  `ci-python-local.yml` install step; a bump is now a deliberate,
  reformat-in-one-PR event.
- **`diagnose()` fails closed on the solver path.** An unknown `solver_path`
  raises `ValueError`; the reserved `sparse_arpack` path raises
  `NotImplementedError` instead of silently falling back to the dense solver.
  Now covered by `tests/test_solver_path_contract.py` (2 tests, executed).
- **D2/D2b symmetrised gaps (`numerics.adjoint`, `diagnostics.spectral`) are
  now genuine Gram-adjoint constructions** (2026-07 audit A1). Three coupled
  defects were fixed:
  - `alicki_adjoint` built the pi-weighted adjoint as
    `(rho^-1 (x) I) L (rho (x) I)` while the GNS similarity transform used the
    anchor-B Gram `G = rho.T (x) I`. The two coincide only for real-diagonal
    steady states; off that manifold the "symmetrised" generator was not
    G-Hermitian, its silent Hermitisation masked the defect, and `gns_gap()`
    returned unphysical **negative** values (repro: generic d=3 GKSL with
    steady-state coherences gave `Delta_s = -4.4` at `Delta = 0.63`). The Kron
    factors are now transposed — i.e. exactly `G^{-1} L G`. No-op for the
    real-diagonal anchor fixtures (all 21 anchors unchanged).
  - `kms_gap` mixed pictures: it symmetrised with the *GNS* adjoint but
    conjugated with the *KMS* Gram. It now uses the KMS-Gram adjoint via the
    new `numerics.adjoint.gram_adjoint(L, G)` helper.
  - Gap extraction (`_real_gap_from_symmetric`) now deflates exactly ONE
    steady-state zero mode instead of filtering *all* near-zero eigenvalues.
    A degenerate Hermitian-part kernel means there is **no certified
    exponential GNS contraction** (`Delta_s = 0`); the old filtering
    over-certified the bound (`Delta_s > Delta`, contradicting the
    Kadison-Schwarz contraction property). Consequence: driven pure dephasing
    with `rho_ss = I/2` now honestly reports `gns_gap = 0` (the sigma_z mode
    has zero instantaneous GNS decay rate); detailed-balance systems keep
    `Delta_s = Delta`. New contraction-bound regression tests
    (`0 <= Delta_s <= Delta`, `0 <= Delta_KMS <= Delta`) pin the fix on
    non-diagonal steady states (40/40 random systems verified).
- **`sparse.chi1.chi1_lower_bound` returned `K^(1/4)` instead of the
  documented `K^(1/2)`** (2026-07 audit A4): the ratio
  `||r||*||l|| / |<l,r>|` is already the square root of the Petermann factor;
  the code took a second square root, quadratically weakening the
  ARPACK-reliability certificate. A regression test now pins the certificate
  against the dense `petermann_factors` oracle (the old test only asserted
  `chi > 0`).
- **Silent zero-width confidence interval on bootstrap failure**
  (`diagnostics.relaxation.compute_relaxation_layer`, 2026-07 audit A7): a
  swallowed `parametric_bootstrap`/`bca_ci` exception used to keep the
  degenerate initialisation `(beta_D, beta_D)`, which the uncertainty layer
  read as `fit_uncertainty = 0.0` — *perfect certainty* as the failure mode.
  The CI is now `(nan, nan)` plus a `RuntimeWarning`;
  `compute_uncertainty_layer` already maps that to `fit_uncertainty = nan`.
- Robustness guards (2026-07 audit A10/A11): `core.lindblad.steady_state`
  casts integer/bool input instead of crashing in `np.finfo`;
  `examples.v4_thermal_two_level` rejects `beta*omega <= 0` (divide-by-zero);
  `io.seed.seed_everything` rejects `bool` seeds (silent 0/1 seeding);
  `core.jumps.engineered_target_jumps` rejects a zero target vector and its
  docstring now matches the two-operator return.
- Docstring/comment drift fixed against implementations: D4 spread comment in
  `_types.py`, pseudospectrum grid location, D18 sensitivity ensemble, D7b
  fallback value, `diagnostics/__init__` layer count.
- `liouscope.numerics.cptp`: fail-closed input validation now rejects
  non-finite generator/channel entries (`L_super`, `channel_super`) and invalid
  (negative / non-finite) tolerances *before* the matrix exponential, Choi
  construction, or eigensolvers run. This replaces opaque LAPACK failures with a
  clear `ValueError` and prevents a bad caller tolerance from silently inverting
  the CP/TP verdict.

### Added
- **Property-based test suite** (`tests/test_property_based.py`, Hypothesis):
  ground-truth-free invariants on randomly drawn GKSL systems — gap scaling
  `gap(c*L) = c*gap(L)`, spectrum invariance under unitary similarity,
  trace-distance axioms + CPTP contractivity, the GKSL semigroup law
  `Phi_{t1+t2} = Phi_{t2} Phi_{t1}`, and no-false-alarm coverage for the Choi
  CPTP gate. `derandomize=True` keeps CI deterministic (the `hypothesis` dev
  dependency was previously declared but unused).
- **CodeQL SAST workflow** (`.github/workflows/codeql.yml`): weekly + PR/push
  `security-and-quality` Python analysis, SHA-pinned, minimal permissions
  (OpenSSF Scorecard SAST check; ruff is a linter, not SAST).
- **pre-commit configuration** (`.pre-commit-config.yaml`): ruff-check,
  zizmor (regular persona), CITATION.cff validation (`cffconvert`) and
  standard hygiene hooks. No formatter (history churn) — CI remains the
  authoritative gate.
- `numerics.adjoint.gram_adjoint(L_super, G)`: adjoint of the
  Heisenberg-picture generator w.r.t. an arbitrary Gram matrix (GNS and KMS
  are the two instantiations).
- `jsonschema>=4.18` added to the `dev` extra so CI actually exercises the full
  `Draft202012Validator` structural-conformance path for run manifests. It was
  previously absent from the dev/CI environment, so `_compiled_validator()`
  returned `None` and the manifest schema-contract test was silently skipped on
  every CI run (the built-in required-field fallback still ran). No runtime
  change — `jsonschema` remains an optional runtime extra with graceful fallback.
- `tests/test_cptp_choi.py`: regression coverage for the new non-finite / invalid
  tolerance seams, plus an independent partial-trace oracle that pins the
  `choi_matrix` tensor-leg convention (the identity channel's Choi matrix is the
  unnormalised maximally entangled operator, *not* `eye(d**2)`) and its
  CP-boundary PSD-ness across `d in {2, 4, 8, 16}`.

### CI
- `ci.yml`/`ci-qutip.yml`: `concurrency` groups auto-cancel superseded runs on
  non-main refs (sp-repo-review GH102).
- `pypi.yml`: fail-closed release QA gate (`twine check --strict` +
  `check-wheel-contents`) between build and Trusted-Publishing upload.
- `scorecard.yml`: `publish_results: true` now that the repo is public
  (badge-capable, externally auditable score); stale private-repo comment
  refreshed.
- `pyproject.toml` strictness batch (sp-repo-review): pytest gains
  `--strict-config`, `xfail_strict`, `log_level` and
  `filterwarnings = ["error", ...]` (unexpected warnings — including
  deprecations from dependency drift — now fail CI; the documented
  `ResolventConvergenceWarning` lower-bound path is allowlisted); mypy gains
  `warn_unreachable` + `enable_error_code = [ignore-without-code,
  redundant-expr, truthy-bool]`; `wheel` removed from `build-system.requires`
  (setuptools adds it itself when needed); redundant ruff `target-version`
  dropped (inferred from `requires-python`).
- `pypi.yml`: the Trusted-Publishing step now sets `print-hash: true` so each
  uploaded sdist/wheel's SHA-256 is logged for the release-evidence lock. No
  second attestation step is added — the PyPA action already uploads a PEP 740
  attestation by default under Trusted Publishing, and PyPI rejects duplicate
  predicates / more than two attestations per file. `docs/RELEASE_AUDIT_v0.5.0.md`
  §5 documents the evidence flow; the stale "private repo" comment was refreshed
  to "public".

### Documentation
- README: the diagnostic layer table now uses the **code-backed D-numbering**
  (module docstrings / StabilityReport keys as source of truth) — the previous
  table circulated a second, contradictory numbering scheme. The "24
  diagnostics" claim is qualified honestly: D1-D20 (+D2b/D7b/D11b) and D24 are
  code-backed; D21-D23 are schema-defined but not yet implemented.
- README/AGENTS.md reproducibility claims corrected to what the code does: the
  manifest does not (yet) record lattice geometry / dissipator family / full
  result graph; `seed_everything` does not control SciPy/BLAS threading.
- README quickstart no longer suggests the dephased-qubit example classifies
  as "A1" (see the tracked Mpemba false-positive issue).
- Synchronized `docs/CANON_STATUS.md` and `AGENTS.md` with the v0.5.0 runtime
  canon: public-repo status, Python 3.14 CI coverage, the dedicated QuTiP
  cross-check checks, and the `StabilityReport v2.1` additive projection.
- Added an Architecture Decision Record directory (`docs/adr/`) with **ADR 0001
  — Scientific-Python support policy (SPEC 0)**. ADR 0001 records the decision to
  follow SPEC 0 and the *post-v0.5* target (`requires-python >=3.12`, CI
  3.12/3.13/3.14, drop 3.10/3.11 in a dedicated support-policy release). It is a
  decision record only — `pyproject.toml`, the CI matrix, and classifiers are
  unchanged in this PR.
