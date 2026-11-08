# Development migration: 0.6.0.dev0

Status: Unreleased development line.

The immutable v0.5.0 release remains the citable release. The default branch now reports 0.6.0.dev0 so VCS installations and run manifests do not reuse a released version identifier for materially different code.

## Ensemble evidence API

A caller-controlled Boolean is no longer accepted as evidence. The public API now requires an immutable `liouscope.EnsembleEvidence` value through `ensemble_evidence`.

The evidence contract validates and binds the evidence-manifest SHA-256, initial-state family and ordering parameter, paired run IDs and input hashes, relaxation metric, comparison test, uncertainty method, generating software version, gate status and reason code, and distinct producer and reviewer attestation SHA-256 digests.

Only a PASS gate with reason `ENSEMBLE_MPEMBA_CONFIRMED` can lift the single-state maximally-mixed A11 insufficient-evidence floor. The canonical evidence digest enters the run input hash and the complete payload is retained in `DiagnosticReport.extras`.

The object is a provenance contract, not proof of organizational independence or scientific correctness. Repository review and release promotion remain separate gates.

## Manifest migration

`MANIFEST_SCHEMA_VERSION` moved from 1.3.0 to 1.4.0 because the input-hash derivation domain now includes the canonical ensemble-evidence digest when structured evidence is supplied. Run IDs and input hashes must be compared only within the same manifest schema version. Existing 1.3.0 manifests remain valid historical records but do not re-derive under 1.4.0 when ensemble evidence participates.

`MANIFEST_SCHEMA_VERSION` then moves from 1.4.0 to 1.5.0 because `compute_input_hash` now absorbs each input object as a length-framed, type-tagged field instead of a bare `repr`/byte concatenation. The old encoding was not injective: distinct input tuples could collide when their serialised forms concatenated to the same byte stream (for example `compute_input_hash(12, 3)` and `compute_input_hash(1, 23)`; issue #97 item 4). Within `diagnose()` the fixed arity and types made this practically unreachable, but `compute_input_hash` is public API, so the derivation is hardened. As with every schema step, 1.4.0 run IDs / input hashes do not re-derive under 1.5.0; compare provenance keys only within one schema version.

## Relaxation layer: default time window and model selection

A call to `diagnose()` or `compute_relaxation_layer()` **without `t_grid`** now fits the gap-scaled window `[0, RELAXATION_HORIZON / gap]` (80 points; a two-scale grid when the fastest mode is not resolved) instead of the absolute `linspace(0, 10, 80)` of v0.5.0. The two are identical only at `gap = 1`. Everywhere else the selected model (`aicc_model`), `beta_D` and its interval can change. Measured on one damped transverse-field site (`gap = 0.35`, `tests/test_pr127_default_window_migration.py`): v0.5.0 window M0, `beta_D = 0.75552`; new default M0, `beta_D = 0.753447` with interval `[0.74726, 0.75827]`, on Windows x86-64 and Linux x86-64 alike (see "Reproducibility of the default fit" below).

- To reproduce v0.5.0 numbers, pass `t_grid=np.linspace(0.0, 10.0, 80)` explicitly.
- `RelaxationResult.t_grid_source` (`"caller"`, `"gap_scaled"`, `"gap_scaled_multiscale"`, `"legacy_fixed"`) and `t_grid` record which window produced a result.
- `RelaxationResult.interval_method` names the interval estimator (`"BC"` on the default 80-point grid, `"BCa-delete1"` on grids of at most 60 points, `"unavailable"`). A BC interval is a shifted percentile interval of the parametric bootstrap. It need not contain the point estimate (in the example above it does).
- `aicc_model` and `linear_fit_model` can now be `"INDETERMINATE"` (`liouscope._consts.MODEL_SELECTION_INDETERMINATE`) when the AICc ranking is not resolved above the rounding and solver level of the curve. In that case the corresponding rate is NaN and `interval_method` is `"unavailable"`. Consumers that switch on the model name must handle this value as they handle `"none"`.

### Reproducibility of the default fit

**Resolved (PR #127 option B, prereg `docs/prereg/2026-11-08_m2_identifiability.md`).** M2 is now optimised in its characteristic variables, and an M2 whose rates leave the window the grid resolves, or whose equilibrated Jacobian is singular, is not a model-selection candidate (`FitResult.identifiable`, `identification_reason`). On the example the second rate now converges to 7.67 (`beta2 * dt = 2.77 > 1`), M2 is refused, M0 is selected, and `beta_D = 0.753447` agrees between Windows and Linux to ~2e-10 relative. The history below is kept because results computed between PR #127 and this fix carry it.

Before the fix the default-window `beta_D` reproduced across platforms only to about 0.6 %, although the selected model (M2), the interval method (BC) and the legacy-window result (M0, `0.75552`, agreeing to better than 1e-9 relative) did. Measured at commit 2f3b7e3 with numpy 2.5.4 / scipy 1.18.1: Windows x86-64 reports `0.72143` (estimate above its interval), Linux x86-64 on Python 3.12 and 3.14 and Linux arm64 report `0.71737`-`0.71738` (estimate inside). The cause is the fit, not the BLAS of the fit: the second M2 component runs off to `beta2 ~ 1e6`, decays inside the first grid step and has an exactly zero Jacobian column; `scipy.optimize.least_squares` then stops on its `xtol` test, which is relative to `|x|` and therefore dominated by `beta2`, as soon as a `beta1` step falls below about `1e-2`. Where it stops is decided by the propagated relative-entropy curve, which differs between the platforms by 5e-8 relative: the Windows solver run on the Linux curve returns the Linux value, and 25 one-ulp shifts of the field amplitude on one platform reach both values. With `xtol = 1e-12` both curves converge to `beta1 = 0.71731` at a higher likelihood, so both reported values were early stops.

## PyPI release workflow

Manual workflow dispatch is build and QA only and receives no OIDC token. The publish job runs only for a published GitHub Release and only when the repository publication flag is enabled. Before upload, both jobs verify source version, built-wheel version, release tag, checked-out commit and release event SHA.

The build job also runs as a dry-run on every pull request and every push to `main`, so the release path is exercised before a release depends on it; it never uploads. The build toolchain (`build`, `twine`, `check-wheel-contents` and the `setuptools` backend) is installed from the hash lock `.github/requirements/release.txt` with `--require-hashes --only-binary :all:`, and the distribution is built with `--no-isolation` so that no unpinned backend is downloaded at build time. Each run builds twice from the same commit with `SOURCE_DATE_EPOCH` set to the commit time and refuses to continue unless the two wheels are byte-identical and the two sdists are content-identical (setuptools does not apply `SOURCE_DATE_EPOCH` to sdists, so only their timestamps and owner fields may differ); the SHA-256 of each artifact is written to the job summary. `.github/scripts/check_release_pins.py`, run by the Quality Contract, keeps the lock and these installs consistent.

## Compatibility

Calls that omit the former Boolean, or explicitly pass false, preserve single-state behavior. Passing true now fails closed with migration guidance. This intentional change belongs to the next development minor line and does not alter released v0.5.0.
