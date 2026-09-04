# Diagnostic layers and the mechanism taxonomy

LiouScope's schema (`D1-D24-Übersicht-v3`) organises two dozen diagnostics
into layers, and a classifier maps their joint pattern onto twelve mechanism
classes. This page explains the vocabulary; the per-diagnostic formulas live
in the module docstrings under `liouscope.diagnostics`.

## What is code-backed, and what is not

Honesty about implementation status is part of the contract:

- **D1–D20** (plus sub-diagnostics D2b/D7b/D11b) are code-backed — the
  peer-review submission set.
- **D21–D23** are schema-defined post-submission slots that are **not
  implemented** in this repository; the reserved-slot contract is itself
  tested (`tests/test_reserved_slots.py`).
- **D24** (Zhou mixing-time predictor) ships as an opt-in frozen module with
  claim status `reference-verified-bound-coarser`
  (see {doc}`../how-to/zhou-mixing-time`).

## The layers

| Layer | IDs | Question it answers |
|---|---|---|
| **S — Spectral** | D1–D4 (+D2b) | Where are the eigenvalues? (gap, GNS/KMS symmetrised gaps, oscillating-mode gap, spread) |
| **R — Relaxation** | D5–D7 (+D7b) | What does the trajectory actually do? (entropies, fidelity, asymmetry) |
| **N — Non-normality** | D8–D13 (+D11b) | Why can the spectrum mislead? (Henrici, Petermann, Kreiss, resolvent, pseudospectra) |
| **T — Transient** | D14–D15 | How large is the pre-asymptotic excursion? |
| **C — Classification** | D16–D20 | Which mechanism explains the pattern? (LEP proximity, gap-rate consistency, initial-state sensitivity, Mpemba) |
| **U/G — Uncertainty & Governance** | U0–U2, D24 | How sure are we, and can someone else reproduce it? |

Two details in the C-layer are easy to get wrong and were deliberately
hardened:

- **D17 (gap-rate consistency)** compares the gap against a rate fitted from
  a *linear* trace-distance/fidelity-type relaxation observable
  (`beta_D_linear`), not against a relative-entropy rate — comparing decay
  constants of dimensionally different observables was a category error.
- **D19 (Mpemba overlap)** treats a small overlap $c_1 \approx 0$ only as a
  *candidate* signal: symmetry-protected trivial overlaps are screened out,
  and confirmation requires ensemble evidence (below).

## The A1–A12 mechanism classes

The classifier (`classify_mechanism`, taxonomy `A1-A12-v3.1`) names the
mechanism, each anchored to the literature:

| Class | Mechanism |
|---|---|
| A1 | Asymptotic-gap-controlled (primitive QMS) — *the gap works* |
| A2 | Sym-gap-corrected transient (Mori–Shirai 2023) |
| A3 | Overlap/eigenvector-amplified (Mori–Shirai 2020) |
| A4 | Skin-affected (Haga 2021) |
| A5 | Metastable plateau (Macieszczak 2016) |
| A6 | Accelerated-decay / operator-spreading |
| A7 | Weak-dissipation singular (Mori 2024) |
| A8 | Oscillatory transient (complex pairs) |
| A9 | Prethermalization-affected (ETH regime) |
| A10 | Phantom relaxation (Žnidarič 2023) |
| A11 | Non-normal Mpemba (Entropy 27, 581, 2025) |
| A12 | Mixed / unresolved |

Orthogonally, the **F-families** name *how the gap fails*: F1 Mori–Shirai
overlap, F2 Liouvillian skin effect, F3 symmetrised gap, F4 quantum Mpemba,
F5 phantom relaxation — or `"none"`. `A1` maps to `"none"` by definition:
"no failure" is a first-class classification, not a missing value. Both
catalogues are importable as `liouscope.A_CLASS_DESCRIPTIONS` and
`liouscope.F_FAMILY_DESCRIPTIONS`.

## Verdicts: evidence-graded, fail-closed

Every classification carries a verdict and a tier:

- **Verdicts:** `CONFIRMED` > `CANDIDATE` > `NOT_EXCLUDED` > `UNDEFINED`.
  There is deliberately **no `EXCLUDED`**: the diagnostics establish positive
  evidence for mechanisms; they are not designed to prove absence, and a
  vocabulary that allowed exclusion invited overclaiming.
- **Tiers:** `EXPLORATION` → `CONFIRMATION` → `PUBLICATION_GRADE`.
  `PUBLICATION_GRADE` is bound to positive certificates — e.g. A1 requires a
  certified symmetrised-gap statement, and F3/A2 cannot be promoted off a
  conservative gap *sentinel* value (a sentinel says "we could not certify",
  which must never be read as "certified").
- **Abstention is correct behaviour.** `UNDEFINED` at tier `EXPLORATION`
  means the run does not carry the evidence for a claim. The canonical
  example: a single-state run whose steady state is maximally mixed cannot
  support an A11 Mpemba claim, whatever the overlap says, because Mpemba is
  an *ensemble ordering* statement — see
  {doc}`../how-to/ensemble-evidence` for the typed evidence that lifts the
  floor.

The design rule behind all three bullets is the same: **fail closed.**
When evidence is missing, malformed, or merely asserted, the report degrades
to the weaker claim rather than trusting the caller.

## `support_score` (né `confidence`) is ordinal, not a probability

`ClassificationResult.support_score` is a **deterministic, rule-based support
score** in `[0, 1]` (fixed values such as `0.70`, `0.85`, `0.95` attached to
specific evidence combinations). It is **not** a posterior probability and it
has **not** been calibrated against held-out labelled reference families — do
not read `0.85` as "85 % probability the label is right". Treat the number as
an *ordinal ranking of rule strength* (`0.20 < 0.50 < 0.70 < 0.85 < 0.95`)
and rely on the *verdict/tier* vocabulary (which is evidence-graded and
fail-closed) for claims.

`confidence` is the **legacy alias** for the same value: issue #102 offered
rename-with-honest-semantics or calibrate, and the rename shipped first
(option 1). Both fields carry identical values (pinned by test); a genuinely
*calibrated* score would have to pass the preregistered validation design in
issue #102 (family-split calibration/holdout sets, reliability curves,
adversarial negatives) before it may replace the ordinal one.

## The hypothesis evidence matrix

`ClassificationResult.hypothesis_matrix` (issue #102) reports, for **every**
hypothesis of the taxonomy — each decision rung, the A12 fallback, and the
schema-reserved classes — one entry with:

| key | meaning |
|---|---|
| `status` | `SUPPORTED` / `NOT_SUPPORTED` / `UNEVALUABLE` / `RESERVED` |
| `supporting` | atomic conditions that hold, with the evidence values read |
| `counterevidence` | conditions that fail, with their values |
| `missing` | required evidence keys absent from this run (kept as an audit trail even when the rung is already refuted) |
| `claim_floor` | what this run could claim about the hypothesis |
| `support_score` | the ordinal score this class would receive; `None` unless the entry is `SUPPORTED` |

Each rung is a conjunction, so status precedence is conclusive-first: one
evaluated-**false** condition refutes the rung (`NOT_SUPPORTED`) no matter
what a missing sibling measurement would have said; `UNEVALUABLE` is
reserved for the genuinely open case where no evaluated condition is false
and the missing required evidence could still flip the rung to supported.
This keeps the A12 fallback consistent with the decision ladder on
partially collected runs: when every unfired rung is conclusively refuted,
the fallback is `SUPPORTED` — exactly the ladder's deterministic A12.

The claim floor follows explicit, fail-closed rules: `RESERVED` and
`UNEVALUABLE` floor to `UNDEFINED` (no rule / no evidence — no claim);
`NOT_SUPPORTED` floors to `NOT_EXCLUDED` (a threshold that did not fire is
absence of support, **not** proof of absence); `SUPPORTED` receives the
verdict the hypothesis would get were it the winner — so for the reported
class the floor equals the reported verdict exactly (pinned by test).

Both the decision ladder and the matrix are derived from one declarative
rung specification (`_ladder_spec`), so the audit surface cannot drift from
the decision. The matrix is **report-only**: no verdict consumes it, and the
per-hypothesis numerical-uncertainty / perturbation-robustness columns from
the issue-#102 wishlist are *not faked* — they remain open until the
underlying machinery exists.

### Reserved classes are excluded from coverage denominators

A6/A7/A9 have no code-backed decision rule (`RESERVED_A_CLASSES` records the
per-class rationale). They appear in the matrix as `RESERVED` with a
permanent `UNDEFINED` claim floor, and any "n of N classes" coverage
statement must use the reachable denominator
`liouscope.REACHABLE_A_CLASSES` (9 classes), not the full taxonomy — the
reachability contract is pinned by an AST-level test that scans the actual
decision source.

## Known limitation: the F5 decision path is not rate-unit invariant

The A10/F5 (phantom relaxation) verdict currently consumes the
**rate-dimensioned** legacy diagnostics: the absolute `henrici_eta > 1.0`
gate and grid-based D10/D11b/D13 estimates whose default grids contain
absolute rate constants. Under a pure change of rate units `L → cL` these
values move beyond the physical `~c` scaling, so **LiouScope does not claim
that the A10/F5 verdict is invariant under a change of rate units** (issue
#101 release gate). The scale-relative successors — `henrici_relative`
(D8b), `kreiss_scaled` (D10b), `pseudospectral_radius_rel` /
`pseudospectral_abscissa_rel` (D13) — are computed on every run, are exactly
invariant under `L → cL` (pinned by `tests/test_scale_conformance.py`), and
are surfaced as **advisory evidence** with `claim_status: pending`. They do
not influence any verdict yet: the switch requires the preregistered
calibration study and independent physics review specified in issue #101
(slice C), including gapless-normal negative controls, before any threshold
is chosen.

## The relaxation window is measured in the system's own relaxation time

Everything the relaxation layer reports — D5, D6, D7, the M0..M3b AICc
comparison, `beta_D`, its BCa interval and the D17 gap-rate check — is fitted
on a time grid. A decay rate has dimension `1/time`, so an *absolute* default
window would be an unstated claim about the caller's unit of time.

When `t_grid` is omitted, `diagnose()` therefore spans

```text
t in [0, RELAXATION_HORIZON / Delta],   80 uniform samples
```

with `Delta` the D1 gap and `RELAXATION_HORIZON = 10` — a fixed number of
e-foldings of the slowest mode, which is the only window carried along by the
rescaling `L → cL`. The fitted rates track that rescaling to ≤1.2e-3 relative
over twelve decades of rate units
(`tests/test_relaxation_grid_scale.py`). At `Delta = 1` the grid is
bit-identical to the historical `linspace(0.0, 10.0, 80)`; when no decay scale
is resolved (`Delta <= 0`) that historical window is used, since there is then
no timescale to scale by.

The grid is **uniform whenever a uniform grid suffices** — which is whenever
the spread between the slowest and the fastest mode stays under about
`(n_points − 1) / horizon = 7.9`. Past that the window switches to a two-scale
grid (below), and the residual model switches with it.

Which window produced a given run is recorded on the report, so it never has
to be inferred — including the grid itself, which is the abscissa the exported
D5/D6/D7 curves are sampled on:

```python
report.relaxation.t_grid_source
#   "caller" | "gap_scaled" | "gap_scaled_multiscale" | "legacy_fixed"
report.relaxation.t_grid_span
report.relaxation.t_grid         # the sampling, not just its extent
report.relaxation.residual_model
#   what the fits were ACTUALLY whitened with, not what the grid asked for:
#   "ar1"               uniform grid, discrete AR(1)
#   "car1"              non-uniform grid, every fit whitened continuous-time
#   "car1_fallback_ar1" CAR(1) theta failed on every fit -> AR(1) fallback
#   "car1_mixed"        some fits CAR(1), some fallen back
#   "car1_unavailable"  non-uniform grid and no fit succeeded
```

### What one uniform window cannot do — and what replaced it

The two requirements pull against each other. The window must reach `~1/Δ` to
see the slowest mode relax; the step must stay below `~1/r` to see a mode at
rate `r` at all. Eighty uniform samples over ten e-foldings give

```text
samples_per_fast_efolding = min over modes of  1 / (r · blind_r)
```

where `blind_r` is the largest interval the grid leaves unsampled **while that
mode still has amplitude** — i.e. the largest gap that starts before `1/r`,
counting the lead-in `[0, t[0]]` as a gap. The lead-in matters because
`diagnose()` accepts any non-negative start: on `linspace(100, 101, 101)` a
rate-1 mode is sampled a hundred times per e-folding by its step and is still
long gone by the first sample. On a uniform grid starting at zero this reduces
to `1 / (r_max · dt) ≈ 7.9 · Δ / r_max`, so roughly an **eightfold** spread of
timescales is the most one uniform grid can straddle.

Widening the window does not help: an absolute window resolves the fast mode
and misses the relaxation entirely, which is worse for the quantity this layer
reports (measured `beta_D_linear` `3.5e4` relative from the true gap, against
`0.58` for the uniform gap-scaled window).

What *does* help is a non-uniform grid — and the reason this layer long
declined to use one turned out to be false. The objection was that the GLS
layer "whitens with a single AR(1) coefficient, which presumes a constant
sample interval". That is a property of the **discrete parametrisation**, not
of the noise. The stationary continuous-time process (Ornstein–Uhlenbeck,
equivalently CAR(1)) has `Corr(t, t+d) = exp(−θ·d)` for any `d`, so on an
arbitrary grid one whitens with the per-step `a_k = exp(−θ·dt_k)` and rescales
by `sqrt(1 − a_k²)` to keep the result homoskedastic. Measured on a two-scale
grid with exact OU noise, the median `|lag-1 autocorrelation|` of the whitened
residuals is `0.374` with one constant `ρ` — taken at the median step, that
scheme's best case — against `0.073` with the per-step coefficient. On a
uniform grid both schemes give `0.073`, which is what shows the contrast comes
from the grid rather than from the comparison.

So the default window is now **repaired**, not merely disclosed. When the
uniform grid cannot resolve the fastest mode (`max(−Re λ)`, taken from the
spectrum, never guessed), half the points cover `[0, horizon/r_max]` and the
rest carry the window out to `horizon/Δ`; `liouscope.fitting.car1` supplies the
whitening, the exact `N_eff = n² / Σ_jk exp(−θ|t_j − t_k|)` and the
exact-transition bootstrap resampler that go with it. On the reviewer's case
(rates `1e-6` and `1`) the AICc winner M2 now recovers the fast rate as `1.10`
against a true `1.0`, where the uniform window's M2 reported `2.17e-05` for it,
and the D17 linear rate lands `0.021` from the gap instead of `0.58`.

The disclosure remains for what the two-scale grid still cannot reach:

- a **caller-supplied** `t_grid` that does not resolve its own system;
- an **intermediate** timescale. The two segments resolve the fastest and the
  slowest mode; a mode between them can fall entirely between the coarse late
  samples. On three damped qubits at `1e-6`, `1e-3` and `1` the middle mode is
  sampled `0.0019` times per e-folding and the warning fires, naming that mode.

Below one sample per e-folding the layer emits an
`UnderResolvedTransientWarning` and records
`report.relaxation.samples_per_fast_efolding`. The reported rates then describe
the dynamics the window *does* resolve, and a caller who needs the missing
component must supply a `t_grid` covering it — reading the resulting rates as
describing *that* window.

This closes the time-grid unit dependence only. The `henrici_eta > 1.0` gate
above is unaffected: `henrici_eta` is rate-dimensioned, so on an
amplitude-damped qubit it equals the rescaling factor `c` exactly and still
flips A5 → A10 between `c = 1` and `c = 3` whatever the grid. A third,
independent dependence sits in the least-squares solver's own convergence
controls (issue #111). Neither is asserted away by the grid work.
