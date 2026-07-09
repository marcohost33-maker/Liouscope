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
