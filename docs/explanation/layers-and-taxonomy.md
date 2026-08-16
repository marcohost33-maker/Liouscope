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

The grid is **uniform** by requirement, not by convenience: the GLS layer
whitens residuals with a single AR(1) coefficient, which presumes a constant
sample interval. The transient layer's two-scale grid is appropriate there —
a `sup_t` search with no noise model — but reusing it here would make the
lag-1 correlation position-dependent and silently invalidate the whitening,
the AR(1) bootstrap and `N_eff`.

Which window produced a given run is recorded on the report, so it never has
to be inferred:

```python
report.relaxation.t_grid_source  # "caller" | "gap_scaled" | "legacy_fixed"
report.relaxation.t_grid_span
```

This closes the time-grid unit dependence only. The `henrici_eta > 1.0` gate
above is unaffected: `henrici_eta` is rate-dimensioned, so on an
amplitude-damped qubit it equals the rescaling factor `c` exactly and still
flips A5 → A10 between `c = 1` and `c = 3` whatever the grid. A third,
independent dependence sits in the least-squares solver's own convergence
controls (issue #111). Neither is asserted away by the grid work.
