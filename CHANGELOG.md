# Changelog

All notable changes to LiouScope are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hypothesis-wise evidence matrix + `support_score` + reachability gate
  (issue #102, additive, `claim_status: pending`, no verdict change).**
  - `ClassificationResult.hypothesis_matrix` reports one entry for **every**
    hypothesis of the A1-A12 taxonomy: each decision rung, the A12 fallback,
    and the schema-reserved A6/A7/A9. Each entry records `status`
    (`SUPPORTED`/`NOT_SUPPORTED`/`UNEVALUABLE`/`RESERVED`), the `supporting`
    conditions with the evidence values they read, the failing conditions as
    `counterevidence`, `missing` required evidence keys, an explicit
    fail-closed `claim_floor` and the per-class ordinal `support_score`.
    Claim-floor rules: `RESERVED`/`UNEVALUABLE` → `UNDEFINED` (no rule / no
    evidence — no claim); `NOT_SUPPORTED` → `NOT_EXCLUDED` (a threshold that
    did not fire is absence of support, not proof of absence, issue #70 A5);
    `SUPPORTED` → the verdict the hypothesis would receive were it the winner
    (via the unchanged `_confidence` → `_pick_verdict_tier` → A11
    maximally-mixed-floor pipeline), so the winner's floor equals the
    reported verdict exactly (pinned by test).
  - To keep the matrix from drifting away from the decision, the priority
    chain is now defined **declaratively**: `_ladder_spec()` returns rungs of
    atomic `_Condition` predicates (each naming the evidence keys it reads),
    and `_hypothesis_ladder` (winner + shadow report) and
    `hypothesis_evidence_matrix` are both derived from that one spec.
    Behaviour-preserving: conditions, order, short-circuit evaluation and all
    (class, family, verdict, tier, confidence) outputs are unchanged
    (669-test baseline green, anchors untouched).
  - **`ClassificationResult.support_score` (issue #102 "rename" option 1):**
    the honestly-named twin of `confidence`, with documented, explicitly
    NON-probabilistic ordinal semantics (`0.20 < 0.50 < 0.70 < 0.85 < 0.95`
    ranks rule strength; no calibration evidence exists). `confidence` stays
    as the legacy alias carrying the identical value (pinned equal by test);
    a *calibrated* replacement remains gated on the preregistered validation
    design in issue #102. Additive + defaulted (`NaN`) so serialised older
    results stay valid.
  - **Reachability/ontology gate:** new `liouscope.REACHABLE_A_CLASSES`
    (taxonomy minus `RESERVED_A_CLASSES`, 9 classes) is the coverage
    denominator for any "n of N classes" statement; reserved classes appear
    in the matrix as `RESERVED` with a permanent `UNDEFINED` floor and are
    excluded from claims. `RESERVED_A_CLASSES` is now exported too.
  - The per-hypothesis numerical-uncertainty and perturbation-robustness
    columns from the issue-#102 wishlist are deliberately **not faked**; they
    remain open in the issue. No `MANIFEST_SCHEMA` bump: the run-manifest
    contract is untouched (report fields are additive with defaults).
  - Docs: `docs/explanation/layers-and-taxonomy.md` gains the matrix
    vocabulary and the reachable-denominator rule; README documents
    `hypothesis_matrix`; the tutorial prints `support_score`.
  - Tests: `tests/test_hypothesis_matrix.py` (57 tests: taxonomy coverage,
    SUPPORTED ⟺ ladder-fires metamorphic equivalence, claim-floor truth
    table incl. non-finite `beta_D` and the A11 ensemble override, RFC-8259
    serialisation with `inf` evidence, report-only non-influence, legacy
    default contract).
- **Branch-shadowing report: `ClassificationResult.triggered_hypotheses`
  (issue #102 slice "assess branch shadowing", `claim_status: pending`).** The
  classifier resolves by PRIORITY and returns exactly one dominant `a_class`,
  so a system that simultaneously shows, say, Mpemba overlap *and*
  pseudospectral phantom evidence reported only the first — the concurrently
  supported mechanism was silently erased. The new additive field reports
  **every** hypothesis that fires, in decision order, each as
  `{rule_id, a_class, f_family, shadowed}`.
  `shadowed` keys on the (class, family) pair, **not** on ladder position:
  several rungs can reach the same conclusion — `gap_rate_consistency < 0.05`
  (with D17) fires the strong A1 rung and necessarily the residual `< 0.20`
  rung too — and marking the second one shadowed would report a mechanism
  conflict where none exists. Every firing rung is still listed, so two rules
  supporting one conclusion stay visible as corroboration; only a genuinely
  different suppressed (class, family) counts as shadowing.
  The tuple is rule-level and deliberately **not** deduplicated — one
  suppressed mechanism can occupy several entries (`A1` via both its rungs,
  `A10/F5` via the pseudospectral and the M3a rung) — so the count of
  suppressed mechanisms is the number of distinct `(a_class, f_family)` pairs
  among the shadowed entries, not the number of entries. Documented with the
  counting recipe on the public README surface.
  To keep the report from drifting away from the decision it describes, the
  priority chain was extracted into a single `_hypothesis_ladder()` that
  evaluates all rungs and returns `(rule_id, a_class, f_family, fires)`;
  **both** `_pick_a_class()` (first firing rung, else `A12`) and
  `triggered_hypotheses()` are derived from that one list, so there is no
  second copy of the conditions to fall out of sync.
  **No verdict behaviour changed**: class, family, verdict, tier and
  confidence are computed exactly as before (the ladder preserves the decision
  order; only the early `return`s became eager evaluation, and every directly
  indexed evidence key is unconditionally populated by `_gather_evidence`).
  The report is deliberately **not** consumed by any decision — letting it
  drive verdicts would be a classifier change and belongs behind the
  preregistered calibration study of issue #102. The `confidence` field is
  re-documented in `_types.py` as a heuristic support score, not a posterior
  probability; the `confidence` → `support_score` rename remains open.
  Serialised older results stay valid (default `()`).
- **D14 transient amplitude: projector baseline and norm geometry separated
  (issue #103, `claim_status: pending`).** The legacy D14
  `trans_amplitude_ratio = sup_t ||e^{tL}||_2` conflated three questions. It is
  preserved byte-identically and re-documented precisely as an *unstructured
  Hilbert-Schmidt semigroup norm estimate over the full complex Liouville
  space* — not a state-amplitude ratio. Three additive, advisory fields split
  the confounds apart:
  - `steady_projector(L)` builds the asymptotic Riesz projector `P_inf` from an
    **ordered Schur decomposition plus a Sylvester solve**, not from a single
    arbitrary null vector, so a degenerate stationary manifold yields the
    correct rank-`k` conditional expectation. Semisimplicity of every
    peripheral mode is verified via rank deficiency of `L - λI`; a defective
    zero mode is **fail-closed** (`semisimple=False`, downstream value `NaN`).
    The peripheral tolerance is relative to `||L||_F`, so the split is
    rate-unit invariant (consistent with issue #101).
  - D14b `centered_transient_amplitude` = `sup_t ||e^{tL} − P_inf||_2`, and
    D14d `decaying_transient_amplitude` = `sup_t ||e^{tL}|_decay||_2` on the
    decaying invariant subspace in an orthonormal basis.
    **These are not equivalent**, contrary to the wording of issue #103: at
    `t = 0` the centred form is `||I − P_inf||`, and for a non-trivial oblique
    projector `||I − P|| = ||P||`, so centring alone still carries exactly the
    baseline it was meant to remove. Only the restricted semigroup starts at
    `1`. Both are reported so the difference stays visible; a regression test
    pins the identity.
  - D14c `operational_trace_amplitude` evaluates trace-norm amplification on
    traceless-Hermitian **differences of density matrices**, recorded as an
    explicit lower bound on the induced 1→1 norm (state family, seed and time
    grid are reported). For CPTP dynamics it must not exceed 1, which makes it
    its own contractivity control.
  No classifier change: the F2 branch keeps consuming the legacy D14 until the
  preregistered calibration study in issue #102.

  Hardened after cross-family review of PR #105: the peripheral cutoff is the
  Schur **backward error** `n·eps·||L||_F`, not `sqrt(eps)` (the latter absorbed
  genuinely resolved slow modes — a generator with rates 1 and 1e-8 reported
  rank 4 instead of 1); only genuinely zero modes count as stationary, and an
  **oscillatory** peripheral mode fails closed because `e^{tL}` has no
  time-independent limit then; the new diagnostics validate their time grids
  (finite, non-negative, strictly increasing — a backward-time grid evaluates
  the non-CPTP inverse and would fake a contractivity violation); their default
  grids include `t = 0`; the run seed is threaded into D14c and recorded in
  `TransientResult.transient_seed`; and `compute_transient_layer` computes the
  propagator sweep **once** and shares it across the variants.
- **Scale-relative non-normality/pseudospectrum diagnostics (issue #101
  slice A, `claim_status: pending`).** One shared operator rate scale
  `liouscope.numerics.scale.rate_scale(L) = ||L||_F` (documented zero-operator
  semantics, fail-closed on non-finite input) now underpins additive,
  dimensionless variants of the rate-dimensioned legacy diagnostics: D8b
  `henrici_relative = η_N/||L||_F` in `[0, 1]` (clip-tolerance fail-closed),
  D10b `kreiss_grid_lower_bound` (dimensionless grid, local refinement, edge-
  maximizer + convergence metadata in the new `KreissGridEstimate`), scale-
  relative D11b/D12 (`resolvent_peak_scaled`, `ridge_fwhm_rel`), D13 with
  `eps_abs = eps_rel · rate_scale` (`pseudospectral_radius_rel`) and the new
  gap-directed intrusion diagnostic `pseudospectral_abscissa(_rel)` via
  `numerics.pseudospec.pseudospectrum_extent` (single-sweep radius+abscissa,
  NaN "under-resolved" marker instead of a fake `0.0`). All are exactly
  invariant under a positive unit rescale `L → cL` for
  `c ∈ {1e-10 … 1e10}` — pinned by the new slice-B conformance suite
  `tests/test_scale_conformance.py` (invariance, rate-valued `~c` scaling,
  D14 `t → t/c` metamorphic agreement, zero/normal/gapless-normal/Jordan
  oracles, unitary-basis invariance, fail-closed guards). The new fields are
  additive with NaN/False defaults on `NonNormalityResult`/`ResolventResult`
  (older callers and serialised results stay valid), surfaced as **advisory**
  evidence keys (`ADVISORY_EVIDENCE_KEYS` extended; the pinned metamorphic
  non-influence test covers them) and as pending-stamped
  `D8b_henrici_relative`/`D10b_kreiss_scaled` entries in the stability
  report. **No classifier/verdict behaviour changed**: per #101, the F5 gate
  switch is deferred to the preregistered calibration study + independent
  physics review. No manifest-contract change (run manifest fields
  unchanged, schema stays 1.5.0).

### Changed
- **Estimator labelling for D10 (issue #101 re-audit).** The docstrings of
  `kreiss_constant` and the non-normality module no longer describe the
  legacy grid search as "Mitchell 2020": the value is a finite-grid **lower
  bound** without globality certificate. Values are byte-identical; docs
  only.
- **Docs honesty (issues #101/#102 release policy).** `confidence` is now
  documented as a deterministic heuristic support score (NOT calibrated; the
  tutorial's "calibrated 0..1" claim is fixed) and the docs state explicitly
  that the A10/F5 verdict path is not yet rate-unit invariant, with the new
  scale-relative diagnostics listed as pending advisory evidence
  (`docs/explanation/layers-and-taxonomy.md`).
- **`MANIFEST_SCHEMA_VERSION` 1.4.0 → 1.5.0 — injective input-hash encoding
  (issue #97 item 4).** `compute_input_hash` now absorbs each input object as a
  *length-framed, type-tagged* field (`tag || len(payload) || payload`) instead
  of a bare `repr`/byte concatenation. The old encoding was not injective:
  distinct input tuples could collide when their serialised forms concatenated
  to the same byte stream — e.g. `compute_input_hash(12, 3)` and
  `compute_input_hash(1, 23)` both hashed `"123"`. Within `diagnose()` (fixed
  arity/types) the collision was practically unreachable, but `compute_input_hash`
  is exported public API, so the derivation is hardened and the schema stepped.
  Migration: input hashes and run IDs are, as always, comparable only within one
  `schema_version`; 1.4.0 manifests remain valid historical records but do not
  re-derive under 1.5.0. Pinned in `tests/test_manifest.py`
  (`test_input_hash_framing_is_injective`). Docs (`README`, `docs/CANON_STATUS`,
  `docs/DEVELOPMENT_MIGRATION_0.6.0.dev0`, reproducibility tutorial/how-to/
  explanation) updated to `1.5.0`.

### Fixed
- **`steady_state` / `sparse_steady_state` tolerance is now scale-relative
  (issue #97 item 5).** The dense null-space tolerance was
  `max(atol, n2·eps·s[0])` with a default **absolute** floor `atol = 1e-9` in
  arbitrary rate units; a Liouvillian has rate dimension, so a pure change of
  units `L → c·L` flipped the uniqueness diagnosis. Symptom: `1e-10 · L` for
  amplitude damping (unique steady state `|0⟩⟨0|`) raised
  `DegenerateSteadyStateError` with a wrong "null space has dimension 4"
  diagnosis, and with `allow_degenerate=True` returned a wrong state plus a
  warning asserting non-uniqueness as fact. The tolerance is now
  `max(atol, max(rtol, n2·eps) · s[0])` with `rtol = 1e-9` (relative, new
  keyword) and `atol = 0.0` (absolute floor now **opt-in**); the near-zero-trace
  check inside the normaliser is decoupled from `atol` (the null-vector
  candidate has unit 2-norm, so that check is dimensionless). The sparse guard
  in `sparse_steady_state` gets the same semantics: guard threshold
  `max(tol·scale, |sigma_shift|)` with `scale = sqrt(‖L‖₁‖L‖∞)` (cheap upper
  bound on `s_max`) and a default `sigma_shift` chosen relative to that scale
  (`None` → `1e-8·scale`) instead of the fixed absolute `1e-8`. Fail-closed
  direction preserved: genuine degeneracy is detected at every scale. Pinned in
  `tests/test_lindblad.py` (`…_invariant_under_rate_rescaling`,
  `…_degeneracy_still_detected_at_small_scale`, `…_atol_is_an_opt_in_absolute_floor`)
  and `tests/test_sparse.py`
  (`test_sparse_steady_state_diagnosis_invariant_under_rate_rescaling`).
  API note: `steady_state(..., atol=…)` keeps its absolute-floor meaning but no
  longer doubles as the trace threshold; `sparse_steady_state(..., sigma_shift=…)`
  keeps its meaning when passed explicitly. Anchors unaffected (verified
  2026-07-13: `pytest tests/test_anchors.py` green, all canonical fixtures are
  O(1)-scaled where relative ≈ old absolute tolerance).
- **D7b `entanglement_asymmetry` now computes the Rylands charge-sector measure
  (issue #97 item 1).** The previous implementation applied a full single-qubit
  Pauli twirl, which maximally mixes the twirled qubit and yields an entropy
  *deficit* `S(rho_1) + ln2 − S(rho)` rather than the published
  Ares–Murciano–Calabrese / Rylands entanglement asymmetry
  `ΔS_A = S(Σ_q Π_q ρ Π_q) − S(ρ)` with U(1) charge-sector projectors. The
  symptom: a Bell state `(|01⟩+|10⟩)/√2`, which lives entirely in the single
  charge sector `q=1` and is therefore exactly symmetric (`ΔS_A = 0`), was
  reported as `2 ln2 ≈ 1.386`. D7b is now the charge-block-dephasing measure
  (total magnetisation `Q = n_1 + n_2` for the supported `d=4` block). D7b is
  advisory-only — it never feeds a classifier verdict — and is not an anchor
  fixture, so this changes no `tests/test_anchors.py` behaviour and no gate
  outcome; it corrects a mislabelled report value. Pinned in
  `tests/test_relaxation.py::test_d7b_entanglement_asymmetry_is_rylands_charge_measure`.
  `CITATION.cff` stays pinned to released v0.5.0 per its own policy (the citable
  diagnostic surface — that D7b exists — is unchanged); the corrected methodology
  is recorded here for the next release's citation.
- **AR(1) bias-correction docstrings corrected to match measured behaviour
  (issue #97 item 3, §6 Reality-Anchor).** `fitting/neff.py` claimed the raw
  lag-1 estimator is biased by `−(1+3ρ)/n` and that the frozen first-order
  correction `(ρ̂(n−1)+1)/(n−3)` "removes the leading O(1/n) bias term". Monte-Carlo
  (40k reps) shows the raw bias is empirically closer to `−(1+4ρ)/n` and the
  correction leaves a residual `O(1/n)` downward bias of order `−2ρ/n`
  (e.g. `−0.013` at `n=80, ρ=0.5`). The **formula is unchanged** — it stays the
  audit-frozen first-order closed form — but the docstrings now state the true
  cancelled/residual terms and tie the residual to the existing small-`n`
  warning, so the documented claim matches the code. No numerical/result change.
- **Fail-closed hardening batch (2026-07-12 full-repo deep review).** Three
  independent review passes (numerical core, classifier gates, IO/manifest/CI)
  found paths where malformed or non-finite input could silently *upgrade* a
  verdict, bypass a gate or fabricate a result. All fixes below are pinned in
  `tests/test_failclosed_hardening.py` (25 tests) plus additions to
  `tests/test_sparse.py` / `tests/test_lindblad.py` / `tests/test_export.py`;
  methodology-gated findings that need physics decisions are tracked in
  issue #97, not blind-fixed here.
  - **A11 floor override now type-gated** (`diagnostics/api.py`). Any
    duck-typed object exposing `permits_claim_floor_override=True` could lift
    the single-state maximally-mixed A11 floor, bypassing all provenance
    validation in `ensemble.py`; non-`EnsembleEvidence` input now raises
    `TypeError` fail-closed.
  - **Non-finite symmetrised gaps no longer grant the issue-#80 A1
    certificate** (`diagnostics/classification.py`). `kms_gap=inf` (ratio
    collapses to 0.0 <= 1.2) or `gns_gap=inf` (passes the `>=` floor test)
    silently granted `sym_gap_corroborated` and upgraded A1 to
    CONFIRMED/PUBLICATION_GRADE 0.95; both certificate legs now require
    finite gaps.
  - **F5 gapless phantom limit fires when the GNS gap is also floored**
    (`diagnostics/classification.py`). With `gap=0` AND `gns_gap=0` (the
    realistic gapless case) the old `inf > inf` comparison was False and the
    documented gapless-phantom contract never fired, falling through to A12.
  - **Malformed steady-state shapes fail closed in the maximally-mixed floor**
    (`diagnostics/classification.py`). A still-vectorised `(d^2,)` steady
    state read as "not maximally mixed" and silently disabled the A11 floor;
    uninterpretable shapes now raise. The documented `d < 2` placeholder
    behaviour is unchanged.
  - **`EnsembleEvidence` rejects duplicate `input_hashes`** (`ensemble.py`).
    Two paired runs with identical input hashes are the same system + initial
    state (no ordering-parameter variation) and are structurally not a
    reference-family comparison.
  - **`build_liouvillian` / `build_sparse_liouvillian` input gates**
    (`core/lindblad.py`, `sparse/build.py`). NaN/inf rates passed the sign
    test (`NaN < 0` is False), an all-real ±inf diagonal H passed the
    Hermiticity gate, and the sparse builder skipped Hermiticity/rate/shape
    validation entirely (accepting non-GKSL generators the dense twin
    rejects). Both builders now share the same fail-closed gates;
    `examples.v4_thermal_two_level` additionally rejects non-finite
    `beta`/`omega`.
  - **`sparse_steady_state` degeneracy guard (sparse/dense parity, S1 audit
    anchor)** (`sparse/arnoldi.py`). Pure-dephasing-like degenerate NESS
    manifolds returned an arbitrary (not even PSD) ARPACK vector silently
    while the dense path raised `DegenerateSteadyStateError`; the sparse path
    now checks the two smallest-magnitude eigenvalues and fails closed, with
    the same `allow_degenerate=True` escape hatch.
  - **`steady_state` no-null-vector fallback is no longer silent**
    (`core/lindblad.py`). An invertible superoperator (no steady state at
    all) returned the smallest-singular-value direction with residual
    `||L rho|| = O(1)` and no signal; the fallback now emits a
    `RuntimeWarning` carrying the residual, the docstring describes the
    actual mechanism (SVD direction, not "smallest-real-part eigenvector"),
    and degenerate `(0, 0)` / non-2-D inputs get structured `ValueError`s
    instead of a bare `IndexError`.
  - **`diagnose()` validates `t_grid` like every other boundary input**
    (`_diagnostics.py`). A NaN-contaminated grid propagated through
    `expm(L t)` and still produced a finite, confident-looking `beta_D`;
    non-finite, negative, unordered or non-1-D grids now raise.
  - **Zhou predictor early return honours caller-supplied arguments**
    (`_zhou.py`). The all-zero-modes return hard-coded `gap=0.0` /
    `petermann_factor=nan`, overwriting explicitly passed values so the
    manifest-grade record contradicted its own call.
  - **`dump_report` writes RFC 8259-valid JSON** (`io/export.py`). The
    default `allow_nan=True` emitted bare `Infinity`/`NaN` tokens (evidence
    ratios are legitimately inf), which strict consumers (JavaScript
    `JSON.parse`, Postgres `jsonb`, serde) reject; Python's lenient parser
    hid it from the round-trip tests. Non-finite floats are now tagged
    `{"__nonfinite__": "inf" | "-inf" | "nan"}` (mirroring the `__complex__`
    tagging) and all three JSON writers (`dump_report`, `dump_manifest`,
    `dump_stability_report`) enforce `allow_nan=False`.
  - **CI/tooling drift.** The pre-commit ruff hook (v0.12.1) linted with a
    materially different rule engine than the CI gate (ruff==0.15.20) — now
    pinned together with a co-bump note. `docs/requirements.txt` was
    lower-bound-only while RTD builds with `fail_on_warning: true`; the RTD
    toolchain is now pinned exactly (sphinx 9.0.4 / furo 2025.12.19 /
    myst-parser 5.1.0, verified clean under `sphinx-build -W`).
- **Contributor-docs drift.** `CONTRIBUTING.md` still pointed the clone URL at
  the defunct `coworker-research` org (the same drift the packaging metadata
  fix already removed from `pyproject.toml`); it now points at
  `marcohost33-maker/Liouscope`. The lint command was aligned with AGENTS.md
  (`ruff check src tests benchmarks` — `benchmarks/` is part of the CI lint
  gate and was missing from the contributor instructions).
- **Support / governance statement in the README** (issue #72 item 3, JOSS
  community gate). The Contributing section now states explicitly where to get
  help (GitHub issues, no separate channel) and who makes maintainer decisions
  (repository owner; methodology changes via `CHANGELOG.md` / ADRs).

### Added
- **Docs slice 2: Diátaxis sections written out** (issue #72 item 2,
  follow-up slice). The Tutorials, How-to and Explanation stubs are replaced
  with hand-written content: two tutorials (first diagnostic run;
  reproducible runs with SPEC 7 `rng` and manifests), four how-to guides
  (manifest export/validation, QuTiP cross-checks, A11 `EnsembleEvidence`,
  D24 Zhou predictor) and three explanation pages (why "no single number",
  D1–D24 layers + A1–A12 taxonomy + verdict vocabulary, auditable
  reproducibility). All code snippets were executed against the current API
  as part of the change; `sphinx-build -W` stays clean (MyST `dollarmath`
  enabled for the math blocks).
- **QuTiP dynamics differential oracle** (`tests/test_qutip_dynamics_oracle.py`,
  007acc portfolio-audit 2026-07-07 ask). The spectral oracle suite never
  exercised the time-domain propagation that the relaxation layer (D5–D7)
  fits against. New coverage: closed-form trajectory oracles for amplitude
  damping and pure dephasing, CPTP invariants (trace/Hermiticity/positivity)
  at every propagated time point, and QuTiP differentials — `_evolve`
  trajectories against `qutip.mesolve` (independent adaptive-ODE integrator)
  and `expm(L t)` against the exponential of QuTiP's independently built
  Liouvillian. 8 tests; the QuTiP half runs under the existing
  `qutip` marker / `ci-qutip.yml` job.
- **FAIR / registry badges in the README** (issue #72 item 5, "Kleinvieh").
  PyPI version badge (the package is live on PyPI), Zenodo version-DOI badge
  (10.5281/zenodo.21246109, verified 2026-07-08) and the fair-software.eu
  badge at its honestly measured 4/5 (●●●●○: public repository, license,
  registry, citation; the checklist dot requires an OpenSSF Best Practices
  badge, which is future work). Badge state follows the `howfairis` checker
  semantics (Zenodo does not count as *registry* there — PyPI does).
  PEP 735 `[dependency-groups]` was evaluated and **deliberately not added**:
  per the issue's own criterion it only pays off once CI moves to uv, and
  mirroring the extras into groups today would create a second place for the
  dependency list to drift.
- **Sphinx + Read the Docs documentation skeleton** (issue #72 item 2).
  A `docs/` Sphinx site (furo theme, MyST Markdown, `autodoc` + `napoleon`
  API reference) organised along the Diátaxis structure
  (tutorials / how-to / reference / explanation), plus a `.readthedocs.yaml`
  (v2) and `docs/requirements.txt` / a `docs` optional-dependency group.
  Builds clean under `sphinx-build -W` (warnings-as-errors); RTD mirrors that
  gate via `fail_on_warning: true`. The Reference section is auto-generated
  from the public `liouscope.__all__` surface; tutorials / how-to /
  explanation are intentional stubs for a follow-up slice.
- **JOSS paper skeleton** (issue #72 item 3). `paper/paper.md` +
  `paper/paper.bib` following the JOSS 2026 section requirements (Summary,
  Statement of need, State of the field, Software design, Research impact, AI
  usage disclosure). All six bibliography DOIs are resolver-verified; author
  identities/ORCIDs and research-impact evidence are flagged TODO (not
  fabricated) and must be completed before submission.
- **SPEC 7 `rng` keyword, additive phase (a)** (issue #72 item 1).
  `diagnose()`, `seed_everything()` and the D18 surface
  (`compute_lep_layer` / `initial_state_sensitivity`) now accept a SPEC 7
  `rng` argument (int, `numpy.random.SeedSequence`, `Generator` or
  `BitGenerator`) alongside the legacy `seed`; supplying both raises
  `ValueError`. `rng` inputs are normalised to a *derived integer seed* by the
  new public bridge `liouscope.derive_seed` (re-exported from
  `liouscope.io.seed` with the `SeedLike`/`RNGLike` aliases), and that derived
  value is what the run manifest records -- the manifest `seed` field,
  `make_run_id` derivation and schema version are unchanged, so a manifest
  alone still reproduces the run. Passing a `Generator` consumes one draw from
  the caller's generator (SPEC 7 consumption semantics). Legacy `seed`-only
  and no-argument calls remain byte-identical (defaults 42 / 7 preserved).
  `seed` stays fully supported in this phase; deprecation is a later,
  separate SPEC 7 phase. Contract pinned in `tests/test_spec7_rng.py`.

### Changed
- **Coverage ratchet 80 → 90** (issue #72 item 4). Measured branch coverage on
  the baseline (Python 3.14) was 94.61%. The `--cov-fail-under` gate in CI and
  the new `[tool.coverage.report] fail_under` in `pyproject.toml` are both set
  conservatively to 90 (up from the previous fixed 80), leaving headroom for
  cross-version variation on the 3.10–3.14 matrix. `CONTRIBUTING.md` updated to
  match.
- **Release identity and provenance hardening.** Released `v0.5.0` remains
  immutable; default-branch VCS installs now report `0.6.0.dev0`. Manifest
  schema `1.4.0` adds the canonical structured-ensemble-evidence digest to
  the input-hash domain when evidence participates; hashes are comparable
  only within one schema version.
- **A11 typed evidence gate.** A single state on `rho_ss = I/d` remains
  `UNDEFINED / EXPLORATION`. Public `ensemble_confirmation=True` is rejected
  fail-closed. Only validated immutable `EnsembleEvidence` with `PASS` and
  reason `ENSEMBLE_MPEMBA_CONFIRMED` may suppress the floor. The evidence
  binds manifest/run/input digests, family ordering, comparison and
  uncertainty methods, software version, and distinct producer/reviewer
  attestations; its digest enters the run hash and its payload remains in
  `DiagnosticReport.extras`.
- **PyPI publication privilege-separated.** Manual dispatch is build/QA only
  without OIDC. The upload job is restricted to a published release, explicit
  tag checkout, the protected `pypi` environment, the enable flag, and repeated
  source/artifact/tag/commit identity checks. External PyPI/Zenodo/SWHID gates
  in issue #50 remain open.

### Added
- **Classifier semantics debt B3/B4 made explicit contracts** (issue #70,
  completes the sub-findings PR #81 left open; A5/A6/A8/A9 already on main).
  Both changes are **behaviour-preserving** -- no real-input classification
  result changes; the sacred anchor suite (`tests/test_anchors.py`) and the
  V1-V5 golden classification assertions stay byte-identical green.
  - **B3 -- reserved A-classes A6/A7/A9.** The taxonomy `A1-A12-v3.1` defines
    twelve classes but `_pick_a_class` emits only nine; A6 (accelerated-decay),
    A7 (weak-dissipation singular, Mori 2024) and A9 (prethermalization/ETH) have
    no decision branch yet. They are now recorded in `_consts.RESERVED_A_CLASSES`
    as a discoverable code-level contract analogous to `RESERVED_DIAGNOSTIC_SLOTS`
    (D21-D23), so the "12 classes" name stays honest instead of silently
    unreachable. A static-AST reachability test forces the contract to stay in
    lock-step with the code (wiring A6/A7/A9 later must update the reserved set).
  - **B4 -- advisory (unused) evidence named.** `lep_proximity` (D16),
    `bohr_ap_length` (D11) and `mpemba_expansion_alpha` (D20) are surfaced in the
    `evidence` dict for audit but deliberately do NOT influence
    class/verdict/confidence. `classification.ADVISORY_EVIDENCE_KEYS` names that
    contract; a metamorphic test proves the non-influence (perturbing each key
    across `{0, ±1e9, ±inf, nan}` leaves the decision invariant). Wiring any of
    them (e.g. D18 `initial_state_sensitivity` as an A1 confidence *dampener*) is
    a class-influencing design decision with false-positive risk and is left to a
    dedicated PR with anchor + FP coverage -- not blind-hooked here. Pinned by
    `tests/test_classifier_semantics_debt.py` (8 tests, executed).
- **Anti-overfit gates wired into the claim vocabulary** (issue #71 B2). The
  residual-whiteness (Ljung-Box, `fitting/whiteness.py`) and temporal-holdout
  (`fitting/holdout.py`) gates were implemented and unit-tested but exported by
  nothing and connected to no claim-level concept. They are now exported from
  `liouscope.fitting`, and a new `fitting/claim_gate.py::assess_relaxation_claim`
  maps their verdicts onto the StabilityReport `SAFE/REVIEW/BLOCK` vocabulary
  *fail-closed* (worst-wins): a rejected holdout -> `BLOCK`, non-white residuals
  or an un-run gate -> `REVIEW`, both gates run and passing -> `SAFE`. The result
  feeds straight into `build_stability_report(claim_level=...)`. This is purely
  additive -- `diagnose()` behaviour is unchanged; callers opt in. New
  `tests/test_claim_gate.py` pins the full truth table (10 tests, executed).
- **D21-D23 reserved-slot contract** (issue #71 C2). The `D1-D24` schema name
  spans slots that are defined in the Drive-side canon but not implemented in
  this repository. `_consts.RESERVED_DIAGNOSTIC_SLOTS` now records D21-D23 as an
  explicit, discoverable code-level contract next to `DIAGNOSTIC_SCHEMA_VERSION`,
  each marked "reserved ... not implemented", so the "24 diagnostics" name stays
  honest. Pinned by `tests/test_reserved_slots.py`.

### Changed
- **`MANIFEST_SCHEMA_VERSION` bumped `1.2.0` -> `1.3.0` (provenance-derivation
  contract change; MIGRATION NOTE).** The `run_id`/`input_hash` fix below widens
  the set of hashed inputs, so **for the same physical input the derived
  `input_hash` and `run_id` VALUES now differ from v0.5.0/main** (empirically:
  input_hash `1ac0e089…` -> `20a2f552…`, run_id `face535e…` -> `43016c65…` on the
  reference small system). This is intentional (it fixes the collision described
  under *Fixed*), but it is **backward-incompatible for provenance**: a `run_id`
  archived under schema `1.2.0` (v0.5.0 or earlier) does **NOT** re-derive under
  `1.3.0` — the old key is stable and valid *within* `1.2.0`, it is simply not
  comparable across the bump. The JSON manifest STRUCTURE (field list, types,
  `additionalProperties:false`) is unchanged, so a `1.2.0` manifest still parses;
  only the hash-derivation domain changed, which is exactly what the
  `schema_version` bump signals. **Migration:** consumers that key archives or
  reproducibility checks on `run_id`/`input_hash` must (a) treat hashes as
  comparable only *within* a single `schema_version`, and (b) re-emit manifests
  under `1.3.0` if a stable cross-version key is required — there is no value-level
  back-migration (the pre-bump inputs were never hashed). The
  `MANIFEST_SCHEMA.json` `run_id`/`input_hash` descriptions now state the
  schema_version-scoping explicitly so the constraint is machine-discoverable.
  (AGENTS.md Definition-of-Done #6: run-manifest contract touched -> schema bump +
  migration note.)
- **D17 gap-rate consistency is now dimension-coherent** (issue #69). D17 was
  `|beta_D - Delta| / Delta`, comparing the relative-entropy fit rate `beta_D`
  directly with the spectral gap `Delta`. Relative entropy near the steady state
  `pi` is *quadratic* in `(rho - pi)` when `pi` is faithful (full-rank), so it
  decays at `2*Delta`; when `pi` is rank-deficient the null-space-leakage term is
  *linear*, so it decays at `1*Delta`. That metric multiplier `m in {1, 2}`
  (verified empirically across V1-V5: `m ~ 2.0` for faithful `pi` V1/V2/V4,
  `m ~ 1.05` for rank-deficient `pi` V3/V5) inflated D17 (dephasing `~3.3`,
  amp-damp `~1.1`) and made the A1 "gap-controlled" label *unreachable for every
  system*. D17 now uses `beta_D_linear`, the dominant decay rate of the **linear
  trace-distance curve** (LIOU-F-018), which decays at the bare mode rate and is
  therefore dimension-coherent with `Delta`. The relative-entropy rate `beta_D`
  is unchanged (still the headline relaxation rate; the fit/bootstrap/anchor
  pipeline is untouched). The classifier now exposes `beta_D`, `beta_D_linear`,
  `gap` and the implied multiplier `d17_metric_multiplier` in the evidence dict
  (the factor is explicit and auditable, not hidden), and a genuine single-mode
  gap-controlled system can now earn A1 "gap-controlled" (family "none" -- see
  issue #70 A6 below, which corrected the A1 family from F1). The A1 early-branch
  is ordered *after* the F1-F5 gap-failure families and only *before* the
  relative-entropy-shape branches (M2/M3a/M3b): `gap_rate_consistency` +
  `linear_fit_model` are initial-state-dependent, so a strongly non-normal
  phantom/skin operator with an `rho_0` that excites only the slow gap mode must
  NOT shadow its true (operator-intrinsic) F5/F2 mechanism -- it stays A10/F5
  (resp. A4/F2), not A1 (Equalita #79 review). No V1-V5 mechanism label changes
  (golden-pinned end-to-end against baseline `3fef6a1`); only the D17 *number* is
  corrected. New end-to-end regression
  `tests/test_validation_systems/test_d17_gap_coherence.py` (20 tests, incl. V1-V5
  a_class golden + an adversarial non-normal-phantom shadow test) plus synthetic
  ordering regressions in `tests/test_classification.py`. Additive
  fields `RelaxationResult.beta_D_linear/linear_fit_model` and
  `LepResult.beta_D_linear` are defaulted, so serialised reports stay valid;
  `compute_lep_layer(beta_D=...)` -> `compute_lep_layer(beta_D_linear=...)` and
  the `gap_rate_consistency(beta_D=...)` param -> `rate=...` (both internal).
- **U1 solver uncertainty now reads from a named nominal floor**
  (issue #71 B5). `compute_uncertainty_layer` reported a bare `1e-10` magic
  number for U1 when no `solver_residual` was supplied; it now reads
  `_consts.U1_NOMINAL_FLOOR` and the docstring states plainly that U1 is a
  conservative *placeholder* (not a measured residual) unless the caller runs
  an ODE-tolerance sweep -- which `diagnose()` does not. Same numeric value, no
  behaviour change; the semantics are now explicit in code.
- **Classifier semantics debt cleared: A1 family, F5 dimensional coherence,
  LEP degeneracy, dead EXCLUDED verdict** (issue #70, A5/A6/A8/A9). Four
  semantic corrections to the A1-A12 classifier, each with a physics rationale:
  - **A6 -- A1 now maps to family `"none"`, not `F1`.** `F1` is the Mori-Shirai
    *overlap gap-FAILURE* mechanism (PRL 125, 230604); A1 (asymptotic-gap-
    controlled, primitive QMS) is precisely the *no-gap-failure* case. Tagging
    the healthy gap-controlled label with a gap-FAILURE family was a category
    error. `"none"` = "No gap-failure mechanism flagged" is A1's correct family.
    (`tests/test_classification.py` + `test_d17_gap_coherence.py` updated to pin
    A1/`"none"`; the #69 dimension-coherence logic is untouched.)
  - **A8 -- the F5 phantom-relaxation rule is now dimension-coherent and
    scale-invariant.** The old rule `pseudospectral_radius > 2 * gap_to_gns_ratio`
    compared a RATE (the D13 radius `max{|z|: z in sigma_eps(L)}`) against a
    dimensionless ratio, so rescaling the Liouvillian `L -> cL` (a pure change of
    time unit) flipped the A10/F5 verdict. The rule now compares the
    dimensionless pseudospectral *reach* `radius / gap` (how far the
    eps-pseudospectrum extends relative to the asymptotic decay rate `Delta`,
    the physical phantom signature, Znidaric 2023) against `2 * gap_to_gns_ratio`
    -- both sides dimensionless, scale-invariant to leading order. A vanishing
    gap is treated as infinite reach (the gapless/critical phantom limit). New
    metamorphic rescale tests over `c in [1e-3, 1e3]` pin no verdict flip, plus a
    regression proving the old bare-radius rule *would* have flipped.
  - **A9 -- `lep_proximity` (D16) no longer blind to exact degeneracies.** The
    min-separation scan skipped every pair with `sep <= atol`, so an exactly
    degenerate eigenvalue pair -- the STRONGEST exceptional-point signal (two
    eigenvalues coalescing) -- was invisible, and a fully degenerate spectrum
    returned `inf` ("maximally far from an EP"), the exact inverse of the
    physics. Coalescence now yields proximity `0.0`; the candidate-count loop
    uses the same data and a `max(10*min_sep, atol)` window, so the two loops are
    mutually consistent. (D16 measures eigenvalue proximity only; genuine
    defective EP vs semisimple degeneracy is disambiguated by the Petermann
    factor D9, as before.) New edge tests cover exact/full/sub-atol degeneracy.
  - **A5 -- the unreachable `EXCLUDED` verdict was removed** from the `Verdict`
    Literal (and `_consts.VERDICT_EXCLUDED`). A single-pass, maximum-evidence
    classifier reports the best-fit A-class with its support and never reports a
    class it is simultaneously ruling out, so a per-class active-rejection
    verdict is not expressible in this architecture -- the value was permanently
    unreachable through `diagnose()`. The old `confidence < 0.30 -> EXCLUDED`
    branch was also semantically wrong (low confidence = epistemic "unresolved"
    = `NOT_EXCLUDED`, not counter-evidence); low confidence now correctly yields
    `NOT_EXCLUDED`. **API note:** `Verdict` narrows from 5 to 4 members; this is
    a type-surface narrowing only (runtime `diagnose()` output is unchanged --
    it never emitted `EXCLUDED`). Genuine active exclusion is deferred to a
    future per-hypothesis scoring mode. B3 (unreachable classes A6/A7/A9) and B4
    (unused D16/D18/D11/D12/D20 evidence) are analysed but NOT wired in this PR
    (see PR body) -- both are class/verdict-influencing design decisions that can
    introduce false positives and are deferred to dedicated PRs with anchor
    coverage.

### Fixed
- **A1 PUBLICATION_GRADE now requires a positive symmetrised-gap certificate,
  not threshold exhaustiveness** (issue #80; classification-semantics change in
  a dedicated PR per AGENTS.md §3; the twin of the #88 fix). The A1 early
  branch awarded CONFIRMED/PUBLICATION_GRADE (confidence 0.95) whenever
  `gap_rate_consistency < 0.05` + single-exp held and *none* of the F1-F5
  thresholds fired — so the publication-grade claim rested on the
  unprovable exhaustiveness of the F1-F5 threshold set (both #79 reviewers'
  residual): a hypothetical weakly-non-normal gap failure below all thresholds
  with a single-exp-at-gap trajectory would self-certify. The burden of proof
  is now reversed (issue #80 to-do 2): a new evidence key
  `sym_gap_corroborated` (1.0 iff a *measured* symmetrised gap shows no
  F3-grade reduction — certified GNS `gap_to_gns_ratio <= 1.2`, or KMS
  `gap_to_kms_ratio <= 1.2`) gates the 0.95 score; uncorroborated A1 caps at
  **0.70 → CANDIDATE/CONFIRMATION** (honest grade: the single-exp-at-gap
  observable is measured, operator-intrinsic gap control is not).
  `gap_to_kms_ratio` thereby graduates from advisory (#89) to
  class-influencing; using it as an F3 *veto* stays deferred.
  **Anchor-preserving:** the gap-controlled thermal reference has
  `Delta_GNS = Delta_KMS = Delta` exactly (corroborated → 0.95 unchanged);
  V1-V5 golden labels are A5/A11 and never take the A1 confidence path; the
  sacred anchor suite stays green. Synthetic adversary + KMS-certificate +
  double-floored fail-closed tests added. `claim_status: pending` until
  cross-family review confirms the semantics.
- **A2/F3 no longer fires off the `gns_gap` conservative-floor sentinel**
  (issue #88; classification-semantics change in a dedicated PR per AGENTS.md
  §3). `diagnostics.spectral.gns_gap` deliberately floors `Delta_GNS` to ~0
  when the GNS symmetrisation certifies no contraction (the documented 2026-07
  audit-A1 behaviour for non-detailed-balance steady states carrying
  coherences; that floor is unchanged). But `_gather_evidence` turned the
  sentinel into `gap_to_gns_ratio ~ 1e10..inf`, and the F3 branch plus
  `_confidence` promoted the exploded ratio straight to **A2/F3 CONFIRMED /
  PUBLICATION_GRADE** — a publication-grade Mori-Shirai-2023 mechanism claim
  keyed on the *absence* of a certificate, on inputs as tame as a textbook
  Rabi-driven amplitude-damped qubit with `Delta_KMS == Delta` (provably no
  real symmetrised-gap reduction). Fix (issue #88 option 1: positive-evidence
  burden): the evidence dict now carries `gns_certified` (1.0 iff
  `gns_gap >= GNS_CERTIFIED_RTOL * gap`, new named constant `1e-8` in
  `_consts`), and both the F3 branch and the A2 high-confidence rule require
  it — the uncertified sentinel falls through to the state-dependent shape
  branches (honest floor: absence of evidence). `gap_to_kms_ratio` is now
  surfaced as *advisory* audit context (issue #88 option 2 — wiring it as an
  F3 veto is deferred to its own FP study); the advisory metamorphic contract
  covers it. **Verdict flips (uncovered input class only):** the #88 repro
  family (Rabi qubit, drives 0.3/0.7/1.5) flips A2/F3 CONFIRMED/
  PUBLICATION_GRADE (0.85) → A8 or A10-via-M3a CANDIDATE/CONFIRMATION (≤0.70).
  **Anchor-preserving:** V1/V3/V4 have diagonal steady states (finite,
  certified `gns_gap`), V5 is caught by the Mpemba branch first; the sacred
  anchor suite and all V1-V5 golden assertions stay green. A *certified*
  reduction still fires A2/F3 at 0.85 (positive-control test). New e2e guard
  `tests/test_classifier_f3_sentinel.py`; `claim_status: pending` until
  cross-family review confirms the semantics.
- **Hermiticity / normalisation validation gates are now absolute, not
  ~10^4x looser than advertised.** `numerics.linalg.is_hermitian` (and
  therefore `is_density_matrix`), the Hamiltonian Hermiticity check in
  `core.lindblad.build_liouvillian` ("H must be Hermitian within 1e-9 atol"),
  and the unit-norm check in `core.jumps.engineered_target_jumps` all called
  `np.allclose`/`np.isclose` with only `atol=` set. NumPy's default
  `rtol=1e-5` silently widened each gate to `atol + 1e-5*|entry|` (~1e-5 for
  O(1) density matrices / Hamiltonians), so a matrix non-Hermitian at 1e-6 —
  a thousand times the documented 1e-9 tolerance — passed as valid. All three
  now pass `rtol=0.0` so the `atol` means exactly what the docstring/error
  says. Behaviour-preserving for every physical operator (Hermitian to machine
  precision); only genuinely malformed input near the old blind spot is now
  rejected. New regression `tests/test_numerics.py::
  test_is_hermitian_atol_is_absolute_not_relative`.
- **GLS AR(1) log-likelihood is now the exact Prais-Winsten likelihood.**
  `fitting.gls.fit_gls_ar1` whitens with the *exact* AR(1) transform (keeping
  observation 0 scaled by `sqrt(1-rho^2)`, all `n` points), but reported
  `gaussian_log_likelihood(whitened)` — the iid-Gaussian likelihood of the
  whitened residuals, which omits the transform's log-Jacobian
  `+0.5*log(1-rho^2)`. The reported value was therefore a hybrid of the exact
  and conditional likelihoods. Because each model M0..M3b fits its own `rho`
  and its per-model `log_likelihood` feeds `aicc()`/`choose_model` (which drives
  mechanism classification), the missing rho-dependent term biased cross-model
  AICc toward under-fitting high-`rho` fits. The Jacobian is now added.
  **Anchor-preserving:** the sacred `tests/test_anchors.py` suite and the V1-V5
  golden classification assertions stay green (the correction does not flip the
  selected model on any canonical fixture); it only affects near-tied AICc
  comparisons on other data. No manifest-contract or citable-claim change (the
  run manifest does not record per-model likelihoods; `input_hash` is derived
  from inputs, not outputs). New regression `tests/test_fitting.py::
  test_gls_ar1_log_likelihood_is_exact_prais_winsten`.
- **`run_id` / `input_hash` now cover every output-affecting input.** Both
  provenance keys were derived only from `(L_super, rho_initial)` (plus `seed`
  and `framework_version` for `run_id`), so two `diagnose()` calls that differed
  only in an analysis knob — `bootstrap_B`, `t_grid`, `include_mpemba`,
  `solver_path`, or an explicitly-supplied `rho_steady_state` — produced
  byte-different reports carrying an *identical* `run_id`. That is a collision:
  the `MANIFEST_SCHEMA` documents `run_id` as "deterministically derived from
  input parameters", but a larger `bootstrap_B` (wider BCa CI) reused the same
  key, breaking archival keyed on `run_id`. `compute_input_hash` now folds all
  output-affecting arguments; the `MANIFEST_SCHEMA` `run_id`/`input_hash`
  descriptions are made accurate. Determinism for identical inputs is unchanged
  (repeated runs still collide, as required). New regression
  `tests/test_manifest.py::test_run_id_distinguishes_analysis_config`.
- **`seed_everything` fails closed on out-of-range seeds** before mutating any
  global state. Legacy `np.random.seed` only accepts `0 <= seed < 2**32`; a seed
  of `2**32` passed the previous non-negative-int guard, then raised mid-function
  *after* `PYTHONHASHSEED` and `random.seed` had already been changed (partial,
  inconsistent state). The bound is now checked up front. New regression in
  `tests/test_input_guards.py`.
- **D14/D15 time grid stays strictly ascending under extreme non-normality.**
  `_physics_time_grid` built its log-spaced early segment as
  `geomspace(t_decay * 1e-6, t_early, …)`; when the numerical abscissa exceeds
  the gap by more than six decades the start overtakes `t_early` and the segment
  ran *backwards*, clustering the fine sampling after the transient peak instead
  of before it. The start is now clamped below `t_early` in that regime (a no-op
  for all normal spectra). New regression in `tests/test_transient.py`.
- **D13 `pseudospectral_radius` fails closed on non-finite operators.** The
  grid search fed `L` straight into `eigvals`/`svdvals` with no finite/square
  guard, unlike the rest of `numerics` (`linalg.py`, `cptp.py`); a NaN/inf
  operator surfaced as an opaque LAPACK error deep in the svd loop instead of a
  located, argument-named `ValueError`. Added `require_finite_square_2d` at the
  entry point. New regression in `tests/test_resolvent.py`.
- **Metadata / provenance consistency.** `codemeta.json` no longer overclaims
  "D21-D24 implemented" (it now matches `CITATION.cff`: D1-D20 + D2b/D7b/D11b +
  the opt-in D24, with D21-D23 as reserved schema slots); `MANIFEST_SCHEMA.json`
  `$id` points at the canonical `marcohost33-maker/Liouscope` repo instead of a
  stale org; and `build_stability_report` casts `cp_choi_min_eig` through
  `float()` like every other numeric it stores, so a `np.float32`/`np.longdouble`
  input cannot break JSON serialization.
- **Removed a dead classifier branch.** The A1 `gap_rate_consistency < 0.05 and
  aicc_model == "M0"` pre-check returned the identical `("A1", "none")` that the
  following `< 0.20` line already returns, and the A1 confidence keys on
  `gap_rate_consistency` alone — so it changed neither label nor score. Deleting
  it removes an implied distinction that does not exist (no behaviour change).
- **D16 `lep_proximity` now fails closed on non-finite eigenvalues** (issue #82,
  part 2). NaN / +-inf eigenvalues were silently swallowed by the pairwise
  comparisons and yielded a finite `(proximity, count)` result -- e.g. a NaN
  input returned `(1.0, 1)`, a spurious LEP classification instead of an error.
  `lep_proximity` (and therefore `compute_lep_layer`) now raises `ValueError`,
  listing the offending indices, before the scan. This closes a Silent-Failure-
  Gate violation; the exact/full/sub-atol degeneracy behaviour is unchanged.
  Pinned by new negative-path tests in `tests/test_lep.py`.
- **A8/F5 scale-invariance test no longer overclaims** (issue #82, part 1). The
  test `test_a8_f5_rule_is_scale_invariant_under_rescale` hand-scaled *both* the
  gap and the pseudospectral radius by the same factor, making `radius/gap`
  trivially constant, yet its name claimed full scale-invariance. It is renamed
  to `test_a8_f5_decision_rule_invariant_under_exactly_scaled_evidence` (it pins
  the *decision rule* on idealised evidence only) and a new end-to-end
  metamorphic test recomputes the D13 pseudospectral radius from a genuinely
  rescaled operator `c*M`. Because D13 uses a **fixed** `eps = 1e-3`, the reach
  `radius/gap` is scale-invariant **to leading order only** (measured drift
  ~1.34x over six decades, converging to the true spectral-radius/gap at large
  scale); the classifier docstring already said "to leading order" -- only the
  test/PR wording overclaimed. Relative-eps rescaling for exact invariance is a
  physics-design follow-up, out of scope here.
- **D19/A11 Mpemba detector no longer false-positives on trivially symmetric
  initial states** (issue #68). Three coupled defects were corrected:
  - `diagnostics/mpemba.py::overlap_c1` normalised the slowest-mode overlap by
    `||l_1||`, so only its *zero test* was meaningful. It now returns the true
    **biorthogonal** expansion coefficient `|<l_1, rho_0>| / |<l_1, r_1>|`
    (denominator floored at `EPS_DIV`), the physical weight of the slow mode.
  - A **non-triviality guard** (`is_trivial_overlap`, wired through `diagnose()`
    via the new `rho_steady_state` argument of `compute_mpemba_layer`) now
    distinguishes a *symmetry-protected* zero overlap from an anomalous skip. In
    the sector decomposition set by the **eigenprojectors of `rho_ss`** (robust
    to a degenerate / maximally mixed steady state, unlike an eigenvector basis),
    a `rho_0` whose active blocks are disjoint from the slowest mode's can never
    populate it — trivially fast relaxation, not Mpemba. Amplitude damping with
    the default `rho_0 = I/2` (and the excited state `|1><1|`) previously
    returned `A11 F4 CONFIRMED PUBLICATION_GRADE`; both now fall through the A11
    branch. `MpembaResult` gains a `trivial_overlap` field.
  - A single initial state skipping the slowest mode is now **CANDIDATE-grade,
    not PUBLICATION_GRADE**: `A11` confidence is capped at 0.70 because
    confirmation needs a reference family (e.g. thermal states across
    temperatures) the single-state pipeline does not provide. The README
    dephasing example (`rho_ss = I/2`, which collapses to a single eigenprojector
    so the triviality guard cannot fire) stays `A11` but as
    `CANDIDATE`/`CONFIRMATION`, no longer self-certifying.
  - New/updated coverage in `tests/test_mpemba.py` (biorthogonal coefficient,
    guard truth table, a fine-tuned qutrit true-positive oracle, canonical
    false-positive regressions) and `tests/test_classification.py`.
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
- Consolidated the duplicate support-policy ADR: the governance detail from
  `docs/ADR_SUPPORT_POLICY.md` (release classification, evidence gate, the
  required-steps checklist) is merged into `docs/adr/0001-python-support-policy.md`,
  and the old path is reduced to a pointer stub so existing links still resolve.
  Single numbered ADR going forward.

## [0.5.0] - 2026-06-25

Release cut: the Canon v0.5 diagnostics & contracts wave plus the cross-family
CPTP-Choi hardening (PR #55, B1/B2) and the repo-wide mypy gate fix that
shipped on `main` after `v0.4.1`. MINOR bump per SemVer: backward-compatible,
additive API surface (8 validated formelbuch entries LIOU-A-011/A-012/A-013,
F-018/F-019/F-020, RPT-001, NG-003; new `StabilityReport v2.1` projection and
`cptp` Choi gate). No `diagnose()` / `DiagnosticReport` break, no manifest
schema bump (the new StabilityReport is a separate, additive projection).

### Fixed
- **CPTP Choi gate (`liouscope.numerics.cptp`) hardened against non-GKSL /
  corrupted input** (cross-family math review of PR #55, B1/B2). The gate now
  fail-closes on two seams a naive Choi test waved through:
  - **B1 — trace preservation is checked at the propagator, not the generator.**
    The TP residual was `||<<I| M_L||` (the *generator*), which is dt-/scale-blind:
    a sub-tolerance generator violation amplified by a large `dt` is a gross
    propagator trace violation that an absolute-tolerance generator check passes.
    Repro: generator residual `7.07e-10` (< the old `1e-9`) but propagator trace
    scaled ~148x -> real TP residual `~208`; the old gate reported `is_tp=True`.
    Now TP is `||<<I| Phi - <<I||` on `Phi = exp(dt*L)` -> `is_tp=False`.
  - **B2 — Choi Hermiticity is checked before the PSD test.** A non-Hermiticity-
    preserving map has a non-Hermitian Choi matrix; Hermitising it (`(J+J^dag)/2`)
    before taking the minimum eigenvalue *masked* the defect. Repro: a depolarizing
    channel plus a small non-HP term has Hermitised `min_eig > 0` (old PSD check
    "CP") yet `||J - J^dag|| > 0`; the old gate reported `is_cp=True`. Now a
    non-Hermitian Choi forces `is_cp=False`.
  - Absolute `1e-9` tolerances replaced by **relative** ones (scaled by `||J||` /
    `sqrt(d)`) so the verdict is scale-invariant. `ChoiGateResult` gains
    `choi_herm_residual` and `is_hp`. The gate's CP claim therefore now also
    fail-closes on non-GKSL / corrupted input, not only on physical channels.
  Regression tests added for both repros in `tests/test_cptp_choi.py`.
- `liouscope.fitting.holdout.train_holdout_split` now rejects a non-strictly-
  increasing (or non-finite) time grid, preventing future-sample leakage into
  training from an unordered series (cross-family review c3).
- `liouscope.fitting.whiteness` docstring: precise Ljung-Box (1978) citation
  (Biometrika 65(2), 297-303) and an explicit statement of the `dof = m - n_params`
  choice (default `n_params=0`, the conservative fail-closed dof; cross-family
  review c2). No behaviour change.
- Repo-wide **mypy gate** failure on the Python 3.12-3.14 CI matrix. NumPy >=2.5
  ships type stubs that use the PEP 695 `type` statement, which mypy rejects
  while parsing `numpy/__init__.pyi` unless its target is >=3.12. Raised
  `[tool.mypy] python_version` from `3.10` to `3.12`. Runtime 3.10/3.11 support is
  still guarded by the test matrix and `ruff target-version = py310`; the change
  only affects type-checking semantics, not packaged code. (Preferred over
  capping `numpy<2.5` so v0.5 development stays on the current NumPy.)

### Added
- **Canon v0.5 diagnostics & contracts** (additive, backward-compatible;
  validated formelbuch entries LIOU-A-011/A-012/A-013, F-018/F-019/F-020,
  RPT-001, NG-003). Each new diagnostic is pinned to an *independent* oracle
  (closed form, QuTiP, or analytic soll-value), never to its own machinery:
  - **CPTP Choi-PSD gate** (`liouscope.numerics.cptp`, LIOU-A-011): verifies
    complete positivity of `exp(dt*L)` via the Choi-matrix minimum eigenvalue
    (`>= -tol`) plus the trace-preservation residual `|| <<I| M_L ||`. Oracles:
    the transpose map (positive but not CP) has Choi `min_eig = -1`; a dephasing
    channel sits on the CP boundary at `min_eig = 0` (entry beleg); and an Euler
    step `I + dt*L` is shown non-CP where `exp(dt*L)` stays CP (entry NR-002:
    Euler positivity is not a CP proof). Dense-only CP proof by design.
  - **Trace distance D_tr** (`liouscope.diagnostics.relaxation.trace_distance`,
    LIOU-F-018): `(1/2)||rho-sigma||_1`, the observable relaxation metric beside
    D5/D6/D7, plus an additive `RelaxationResult.trace_distance_curve`. Oracles:
    orthogonal pure states -> 1, diagonal states -> total variation, Fuchs-van de
    Graaf bounds vs the Uhlmann fidelity, CPTP contractivity, and `qutip.tracedist`.
  - **Temporal holdout split** (`liouscope.fitting.holdout`, LIOU-A-012): an
    out-of-sample anti-overfit gate for the M0-M3b hierarchy (time-ordered tail,
    no shuffling).
  - **Residual-whiteness gate** (`liouscope.fitting.whiteness`, LIOU-A-013):
    Ljung-Box Q with the chi-squared reference from `scipy.stats` (white noise
    passes, AR(1) is rejected).
  - **Metamorphic spectral oracles** (`tests/test_metamorphic_spectral.py`,
    LIOU-F-020): `gap(c*L)=c*gap(L)` and `spec(U L U^dag)=spec(L)` — ground-truth-
    free invariants.
  - **Gap-invariant reproduction** (`tests/test_gap_invariants_canon.py`,
    LIOU-F-019): reproduces the pack mini-oracle parametrisation (amplitude
    damping `gap=gamma/2`; thermal `g_down=0.9, g_up=0.2, omega=1.3 -> gap=0.55`).
    NOTE: the *general* `gamma/2` and `(g_up+g_down)/2` oracles already exist in
    `tests/test_qutip_spectral_oracle.py` (PR#51); this adds the exact pack
    parametrisation. Dephasing `2*gamma` is intentionally NOT re-added (canon).
  - **StabilityReport v2.1 contract** (`liouscope.io.stability_report` +
    packaged `STABILITY_REPORT_SCHEMA.json`, LIOU-RPT-001): a claim-safe,
    machine-auditable projection of a `DiagnosticReport` adding `claim_level`
    (SAFE/REVIEW/BLOCK), `direction`, `cp_evidence_level`, independently
    recomputed invariant residuals, an `evidence_bundle` and `provenance`. New
    diagnostics carry `claim_status="pending"`. Purely additive — it does NOT
    modify the existing run manifest, `MANIFEST_SCHEMA.json` or `DiagnosticReport`
    (no schema-version bump; older artefacts stay valid).
  - **Petermann interpretation caveat** (D9 docstring, LIOU-NG-003): a large
    Petermann factor is necessary-but-not-sufficient for transient amplification;
    `sup_t ||e^{tL}||` is bounded by the Kreiss constant (D10) / numerical
    abscissa (D15), not by the Petermann factor alone.

  Every new public boundary ships negative/edge-input gates (negative `dt`, NaN,
  non-square dim, bad `holdout_frac`, `m>=N`, non-positive dof, out-of-enum
  verdict fields, tampered schema fields). Verified locally on CPython 3.14 via
  the CI command chain: `ruff check src tests benchmarks` (exit 0), `mypy
  src/liouscope` clean at `--python-version 3.12` (the pyproject-default invocation
  aborts on an unrelated numpy-stub/toolchain skew in the local sandbox), full
  suite `pytest --cov-fail-under=80` (375 passed, coverage 93.9%), anchors
  `pytest tests/test_anchors.py` (21 passed), `pytest -m qutip` (8 passed).
- Independent-oracle cross-checks for the **non-normality layer D8-D11**
  (`tests/test_nonnormality_oracle.py`). The previous tests
  (`tests/test_nonnormality.py`) only asserted *signs* and self-consistency
  (`eta > 0`, `K > 0`, `length >= 1`), so a wrong normalisation or eigenvalue
  filter would pass. The new module pins each diagnostic to a closed-form or an
  independent numerical oracle, never to the library's own machinery: **D8
  Henrici** against the unitarily-invariant `sqrt(||A||_F^2 - sum|lambda|^2)`
  (Henrici 1962; closed form `|b|` for `[[0,b],[0,-1]]`); **D9 Petermann**
  against the 2x2 adjugate condition-number `tr(B0^H B0)/|li-lj|^2` (e.g. Phys.
  Rev. Research 5, 033042 (2023)) plus the normality floor (Petermann = 1 for a
  pure-dephasing Liouvillian); **D10 Kreiss** against the continuous-time
  reference facts (K = 1 for a normal Hurwitz matrix; K >= 1 always; closed form
  `sqrt(1+b^2)` for `[[0,b],[0,-1]]`) and an independent 2D Nelder-Mead
  resolvent-norm maximisation (Kreiss matrix theorem; Mitchell, SIAM J. Matrix
  Anal. Appl. 41(4), 2020); **D11 Bohr AP** against hand-constructed spectra with
  a known longest arithmetic progression and the `log_2(d)` Pauli bound (Basso,
  arXiv:2510.07267, 2025). Each oracle includes a non-vacuous negative control
  (wrong-zero Henrici, missing-denominator Petermann via eigenvalue gap != 1,
  grid-cannot-exceed-oracle Kreiss, no-three-term-AP Bohr). Test-only; no
  production code, anchor, or `DiagnosticReport` output changes. Verified locally
  via the CI command chain on CPython 3.14: `ruff check src tests` (exit 0),
  `mypy src/liouscope` (exit 0), `pytest -q` full suite (307 passed, 1 skipped —
  pre-existing local `jsonschema`-extra skip), `pytest -m qutip` (6 passed), new
  module `tests/test_nonnormality_oracle.py` (26 passed).
- Independent-oracle cross-checks for the **spectral diagnostic layer**
  (`tests/test_qutip_spectral_oracle.py`). The previous QuTiP cross-checks only
  validated the Liouvillian *builder* (matrix construction); the diagnostic
  *outputs* (D1 gap, D3 oscillating gap, steady state, GNS/KMS symmetrised gap)
  were only asserted for self-consistency and loose magnitudes
  (`abs(gap - 0.2) < 0.05 or abs(gap - 0.4) < 0.05`). The new module pins them on
  three canonical GKSL systems (amplitude damping, coherently driven dephasing,
  detailed-balance thermal qubit) against two independent oracles: closed-form
  analytic spectra (always runs) and `qutip.liouvillian(...).eigenenergies()` /
  `qutip.steadystate(...)` (`@qutip_required`). Includes the Mori-Shirai
  prediction that the symmetrised gap coincides with the standard gap at
  equilibrium (PRL 130, 230404 / arXiv:2212.06317). Test-only; no production
  code, anchor, or `DiagnosticReport` output changes. Verified locally via the
  CI command chain on CPython 3.14.4: `ruff check src tests` (exit 0),
  `mypy src/liouscope` (exit 0), anchors (21 passed), full suite with coverage
  (281 passed, 1 skipped — pre-existing local `jsonschema`-extra skip,
  coverage 93.54% ≥ 80% gate), `pytest -m qutip` (6 passed).
- CI test matrix extended to **Python 3.14** (`.github/workflows/ci.yml`) and
  `3.14` added to the `pyproject.toml` Trove classifiers. Verified locally via
  the exact CI command chain on CPython 3.14.4: `pip install -e .[dev,qutip]`
  (qutip 5.3.0 ships cp314 wheels), `ruff check src tests benchmarks` (exit 0),
  `mypy src/liouscope` (exit 0), anchor regressions (21 passed), full suite
  (274 passed, 1 skipped, coverage 93.54% ≥ 80% gate). Closes the freshness gap
  where the library ran clean on 3.14 but CI never exercised it.

### Documentation
- Added `docs/RELEASE_AUDIT_v0.4.1.md`, the post-`v0.4.1`-tag public/citable-release
  readiness audit (archive/provenance gates + post-v0.4.1 CI hardening). It is
  documentation-only and is **not** part of the tagged `v0.4.1` source snapshot
  (the `v0.4.1` tag at commit `1965f2b` predates this file); it lives in
  `[Unreleased]` as accompanying post-release documentation. Updated
  `docs/CANON_STATUS.md` §5 to point public/citable-release work at this v0.4.1
  audit instead of the superseded `docs/RELEASE_AUDIT_v0.4.0.md` §5.

### Fixed
- `codemeta.json` was stale: it reported `version: "0.2.0"`, described only
  "twenty diagnostics D1-D20", and set `codeRepository` to the non-existent
  `github.com/coworker-research/liouscope`. Synchronized to the repo canon:
  `version` → `0.4.1` (matching `src/liouscope/_version.py` and `CITATION.cff`),
  diagnostic description → 24 diagnostics D1-D24 (D1-D20 original submission set,
  D21-D24 post-submission; schema `D1-D24-Übersicht-v3`, per `_consts.py` /
  `MANIFEST_SCHEMA.json`), and `codeRepository` →
  `github.com/marcohost33-maker/Liouscope`. Metadata-only; no runtime/API change.
- `pyproject.toml` `[project.urls]` (Homepage/Repository/Issues) pointed to
  `github.com/coworker-research/liouscope`, which does not exist (HTTP 404 — the
  `coworker-research` org has no such repo). The published wheel/sdist metadata
  therefore carried dead project links. Corrected to the canonical repository
  `github.com/marcohost33-maker/Liouscope`, matching `CITATION.cff`,
  `AGENTS.md` ("Visibility: PRIVATE (marcohost33-maker/Liouscope)"), and the
  `CHANGELOG.md` version-compare links. Metadata-only; no runtime/API change.

## [0.4.1] — 2026-06-16

Release cut: numerics correctness + production-hardening that shipped on `main`
after `v0.4.0`, PRs #43–#45 (resolvent conjugate-transpose fix, resolvent
hardening + large-matrix scaling, classifier taxonomy doc fix, and coverage
lifts across the fit, Liouvillian, sparse, classification, and numerics
layers). PATCH bump per SemVer: the change set is a numerics correctness fix
plus tests and documentation corrections. The one new public symbol
(`numerics.resolvent.ResolventConvergenceWarning`) is a diagnostic warning on a
numerics utility, not a new feature on the `diagnose()` / `DiagnosticReport`
API surface, and no anchor or report output changes — see the per-entry
"anchors unaffected" notes below.

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
  Hilbert dimensions up to 128, was already correct). (The power-iteration
  convergence handling is hardened further under "Changed" below.)

### Added
- Regression tests for the fit-model layer (`tests/test_models.py`): the
  closed-form M0–M3b evaluations and all initial-guess seeds, including the
  log-linear M0 regression with its too-few-positive-samples fallback and the
  FFT-based M3b dominant-frequency pick (supplied-omega and short-signal
  branches). Raises `fitting/models.py` coverage from 69% to 97% (the only
  remaining line is a defensive `else` unreachable for equal-length inputs).
- Branch coverage for the foundational Liouvillian builder and steady-state
  solver (`tests/test_lindblad.py`): the `jump_ops=None` default, the
  `order != "F"` guard, zero-rate-jump skipping, the rate-length guard
  (distinct from the jump-shape guard), the non-square-superoperator rejection,
  the no-exact-null-space smallest-singular-vector fallback, and the
  traceless-null-space "cannot normalise" RuntimeError. Raises
  `core/lindblad.py` coverage from 81% to 99%.
- Fallback-path tests for the Prony M3b seed (`tests/test_fitting.py`):
  short-signal, non-uniform-sampling, and too-few-samples-for-model-order
  fallbacks — pinning the robustness guards that let `prony_seed` degrade
  gracefully instead of raising. Raises `fitting/prony.py` coverage to 88%.
- Edge-branch tests for `sparse.build.build_sparse_liouvillian`
  (`tests/test_sparse.py`): the order/shape/rate-length validation errors, the
  `jump_ops=None` and `rates=None` defaults, and zero-rate-jump skipping.
  Raises `sparse/build.py` coverage from 71% to 100%.
- Regression tests pinning the A1–A12 mechanism classifier decision tree
  (`tests/test_classification.py`): synthetic-input coverage of every
  `_pick_a_class` branch, the `_pick_verdict_tier` thresholds (including the
  `EXCLUDED` and `UNDEFINED` paths unreachable through natural confidence
  values), and the `_confidence` scoring rules. Raises `classification.py`
  coverage from 71% to 100%.
- Tests for the previously-untested large-matrix branches of
  `numerics.resolvent` (`tests/test_numerics.py`): the SuperLU sparse solve
  (n > 256) and the power-iteration resolvent norm (n > 128, the regression
  guard for the fix above), plus a clustered-singular-value case proving the
  norm stays accurate to < 1e-3 when the top two singular values nearly
  coincide.

### Changed
- `diagnostics.classification` documentation corrected to match the
  authoritative `_consts` taxonomy. The module docstring previously described an
  unrelated "evidence families" scheme (F1=spectral, F4=resolvent, …) that
  contradicted `_consts.F_FAMILY_DESCRIPTIONS`, where F1–F5 denote the
  literature-anchored gap-failure *mechanisms* (F1 Mori-Shirai overlap PRL 125
  230604; F2 skin effect PRL 127 070402; F3 symmetrised gap PRL 130 230404;
  F4 quantum Mpemba PRL 127 060401; F5 phantom relaxation arXiv:2306.07876 —
  all references web-validated). The misleading inline comment on the A3 branch
  ("F4 resolvent-amplified non-normality") was fixed: A3 = overlap/eigenvector-
  amplified (Mori-Shirai 2020) correctly maps to family F1, which is what the
  code already returned — a comment/doc defect, not a behaviour change. Added a
  guard test (`test_family_citations_consistent_between_docstring_and_consts`)
  pinning the docstring and `_consts` to the same citations so they cannot drift
  apart again. Also removed a dead `f_family` parameter from the private
  `_pick_verdict_tier` helper (it never influenced the verdict/tier). No
  classifier output changes; anchors unaffected.
- D11b/D12 resolvent diagnostics now scale to large Liouvillians. The inline
  per-frequency dense inverse + SVD in `diagnostics.resolvent.resolvent_peak_curve`
  (201 dense `O(n^3)` solves, intractable for larger lattices and a duplicate of
  the numerics utility) is replaced by a delegation to
  `numerics.resolvent.resolvent_norm`. For `n <= 128` — which includes every
  anchor/example system — this is bit-identical (same dense inverse + SVD); for
  `n > 128` it uses the SuperLU shift-and-invert power iteration, so the
  resolvent peak (D11b) and ridge FWHM (D12) become computable where the dense
  path previously could not finish. The peak matches the dense reference to
  machine precision; low-norm tail points of the profile inherit the power
  iteration's documented ~1e-3 lower-bound behaviour in the clustered regime.
- `numerics.resolvent.resolvent_norm` hardened for production: the power
  iteration now emits a `ResolventConvergenceWarning` (new, exported from
  `numerics.resolvent`) when it exhausts its budget on tightly-clustered top
  singular values (the returned value is then a documented lower bound), the
  iteration budget was raised 80 → 200 (moderately clustered spectra now
  converge; tight clusters improve from ~2e-4 to ~1e-6 relative error), and the
  size cutoffs / iteration constants are now named module constants instead of
  magic numbers. New tests cover the warning path and a convergent clustered
  case.
- `numerics.resolvent.resolvent_norm` docstring now documents the method as the
  standard pseudospectra shift-and-invert approach (Trefethen, *Pseudospectra
  of Linear Operators*, SIAM Rev. 1997) and records why plain power iteration
  is sufficient for the dominant value (value-convergence is robust to top
  singular-value clustering, unlike eigenvector-convergence — verified
  empirically), so Lanczos/`svds` is intentionally not used.
- Packaging metadata modernised to PEP 639: `pyproject.toml` now declares the
  license as the SPDX expression `license = "Apache-2.0"` with
  `license-files = ["LICENSE"]`, and the deprecated `License ::` trove
  classifier is removed (the SPDX expression is the single source). The
  `setuptools` build requirement is raised to `>=77.0` (first release with PEP
  639 support). This clears the deprecation that setuptools enforces after
  2026-02-18 for the old `license = {text = ...}` table, and is the same
  Metadata-2.4 machinery behind the `twine check` `license-file` note recorded
  for this release. No runtime, API, or dependency change.

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

<!-- Version-compare references (Keep a Changelog 1.1.0). The v0.5.0 tag is
     created at release time; the [0.5.0] link resolves once it is pushed. -->
[Unreleased]: https://github.com/marcohost33-maker/Liouscope/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/marcohost33-maker/Liouscope/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/marcohost33-maker/Liouscope/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/marcohost33-maker/Liouscope/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/marcohost33-maker/Liouscope/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/marcohost33-maker/Liouscope/releases/tag/v0.2.0
