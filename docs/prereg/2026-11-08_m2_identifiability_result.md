# RESULT + AMENDMENT: M2 identifiability (prereg 2026-11-08)

Prereg: `docs/prereg/2026-11-08_m2_identifiability.md` (commit 8ba189b).
Evaluation script: `benchmarks/pr127_m2_identifiability.py`.

## Run 1 -- the gate exactly as pre-registered (commit fad0f55)

Windows x86-64, Python 3.12.10, numpy 2.5.4, scipy 1.18.1.
Upper rate bound `beta * dt_0 <= 1 / MIN_SAMPLES_PER_FAST_EFOLD = 1`.

| criterion | result | verdict |
|---|---|---|
| S1 `_chain1` -> M0, beta_D 0.753447 | M0, 0.75344699988 (Linux Docker: 0.75344699992) | PASS |
| S2 `_chain1` M2 not identified | yes, "rate 7.67161 outside [0.000175, 2.765]"; the `"jac"` endpoint (1.055e6) too | PASS |
| S3 C1 identified + selected >= 9/10 | 10/10 | PASS |
| S4 C2 identified 0, selected 0 | 0 / 0 | PASS |
| S5 C3 (descriptive) | M0 6, M1 3, M3a 1 | -- |
| S6 amplitude invariance rel 1e-9 | worst 1.12e-9 | **MISS** |
| S7 multistart | C0: 0 identified at all; C1: 49/50 at best cost | PASS |
| S8 full suite: no new failures but the fixture | 5 new failures | **MISS** |

S6: the solver stops at `ftol = 1e-8`; two runs that differ by the rounding
of `c*y` stop at points 1e-9 apart. 6b1cd93 with `"jac"`: 2.66e-9. The
pre-registered tolerance sat below the stopping level of BOTH versions. The
test uses 1e-7 (100x the spread, 3e6x below the failures it separates).

S8 (full suite at fad0f55: 7 failed, 2085 passed, 14 skipped; 2 of the 7 are
the expected fixture): `tests/test_validation_systems/test_d17_gap_coherence.py`

| system | M2 on relative entropy (both versions) | beta2 * dt_0 | 6b1cd93 | run 1 | multiplier m (oracle: <= 2.6) |
|---|---|---|---|---|---|
| V1 | interior optimum, AICc -1331 vs M0 -799 | 1.82 | M2, A5 | M0, **A12** | 2.19 -> **2.67** |
| V3 | interior optimum, AICc -862 vs M0 -383 | 1.28 | M2, A5 | M0, **A12** | 1.05 -> 1.07 |
| V4 | interior optimum, AICc -1044 vs M0 -634 | 1.22 | M2, A5 | M0, **A1** | 2.16 -> **2.76** |

These are not run-offs: the second rate converges to an interior optimum
1.2-1.8 x 1/dt_0, carried by 2-3 samples. Refusing it forces M0 to absorb the
early transient and moves `beta_D` up; on V1 and V4 the relative-entropy
multiplier then exceeds its physical ceiling (2 for a faithful steady state,
test oracle 2.6). C2 shows the same cost: 6b1cd93 selected M2 in 10/10 seeds
with the slow rate right (0.4994-0.5009) even where the fast rate ran off to
8e5 / 5e6; run 1 selected M1 in 10/10 with rate 0.6018-0.6051.
Run-1 numbers were produced by `benchmarks/pr127_m2_identifiability.py` at
script SHA-256 `7951877497f0...` (the version committed with this note).

Conclusion of run 1: `MIN_SAMPLES_PER_FAST_EFOLD = 1` is the right bound for
"the grid resolves this mode" (its use for the default grid), but too strict
for "this fitted rate is identified": it removes real, partially sampled
components and makes `beta_D` physically worse.

## Amendment A1 (committed before run 2)

Deviation from the prereg, made AFTER seeing run 1 and therefore post hoc:

* Upper rate bound `beta * dt_0 <= 10` (constant `_M2_MAX_RATE_STEPS`), the
  value the decision dossier of 2026-11-07 (item M4, written before this
  work) gives for "structurally not identifiable: profile flat". A component
  with `beta * dt_0 > 10` keeps less than e^-10 = 4.5e-5 of its amplitude at
  the second sample: it is a one-point component.
* S2 replaced: the gate refuses the `"jac"` endpoint of `_chain1` (1.055e6);
  the scaled fit (7.67) is identified, and `_chain1` is decided by AICc.
* S4 replaced: C2' = C2 with beta2 = 500 (`beta2 * dt_0 = 63`): identified 0,
  selected 0. C2 (beta2 = 50) becomes descriptive.
* All other criteria unchanged.

Not chosen: any bound between 1.82 and 2.77 that would keep V1/V3/V4 and also
refuse `_chain1`'s scaled fit -- that would be fitted to these four cases.

## Run 2 -- amendment A1 (bound 10 / dt_0)

Windows x86-64, Python 3.12.10, numpy 2.5.4, scipy 1.18.1, script SHA-256
`5f503041f521...`.

| criterion | result | verdict |
|---|---|---|
| S1 `_chain1` -> M0, beta_D 0.753447 | M0, 0.75344699988 | PASS |
| S2' gate refuses the `"jac"` endpoint (1.055e6) | refused, `beta*dt_0 = 3.8e5` | PASS |
| S3 C1 identified + selected >= 9/10 | 10/10 | PASS |
| S4' C2p (beta2 = 500) identified 0, selected 0 | 6 / 6 | **MISS** |
| S5 C3 (descriptive) | M0 6, M1 3, M3a 1 | -- |
| S6 amplitude invariance rel 1e-9 | 1.12e-9 (unchanged, see run 1) | **MISS** |
| S7 multistart | C0: 45/50 identified, 0 with AICc below M0; C1: 49/50 | PASS |
| S8 full suite | see the commit that closes this note | -- |

`_chain1`: the scaled M2 converges to beta2 = 7.67 and is identified; it
loses on its own AICc (`N_eff` = 4.71 <= k + 1 on this uniform grid). So
since A1 the `_chain1` result rests on two independent effects, the scaling
fix and the `N_eff` correction, not on the gate. The gate removes only the
`"jac"`-style one-point endpoint.

S4': with noise 1e-3 a component at beta2 = 500 is fitted at the plateau
edge, beta2 = 48.1-58.1 (`beta*dt_0` = 6.1-7.4), in 6/10 seeds; its SLOW rate is then
right (0.50001-0.50086). The 4 refused seeds run to beta2 ~ 307-319 and the
selection falls to M1 with rate 0.6014-0.6024, 20 % off. The bound cannot
see a flat rate profile at the plateau edge; a profile-likelihood interval
(dossier option B item 2) would. Recorded as `xfail` in
`tests/test_pr127_m2_identifiable.py`, not counted as passing.

C2 (beta2 = 50, descriptive): identical pattern, 7/10 M2 at 0.4994-0.5009,
3/10 M1 at 0.6018-0.6025. 6b1cd93 selected M2 in 10/10 at 0.4994-0.5009.
**Trade-off for the reviewer:** refusing a one-point component is what turns
a correct slow rate into a 20 % biased one on these curves; admitting it is
what produced the platform-dependent `_chain1` model choice. Which to prefer
is a methodology decision, not made here.

## S8 after amendment A1 (code at e076f2b)

| run | result |
|---|---|
| Windows x86-64, Python 3.12.10, numpy 2.5.4, scipy 1.18.1, baseline 6b1cd93 | 2063 passed, 14 skipped, exit 0 |
| same, e076f2b | 2093 passed, 14 skipped, 2 xfailed, exit 0 |
| Linux x86-64 (Docker python:3.12-slim, Python 3.12.16, numpy 2.5.4, scipy 1.18.1, qutip), e076f2b | ruff, mypy, anchors (21) exit 0; 2107 passed, 2 xfailed, exit 0 |

The +30 are the new tests; the 2 xfailed are the S4' limit above. S8: PASS
under A1 (no new failure; the two fixture tests were rewritten as foreseen).

Side check, not pre-registered: `benchmarks/pr127_car1_aicc_identifiability.py`
(unchanged script) rerun on this branch, Windows: decision unchanged
(`nobs_on_nonuniform` SUPPORTED, 6/6 cells); every truth-M2 cell unchanged;
truth M0 loses 1-2 false M2 selections in 5 of 6 cells. The committed result
file was left as produced by its own commit.

## Review finding P1-1 (after 744da3d): refusal must not move the selection

Independent review: the A1 bound also refuses M2 fits whose SLOW rate is
right, and `_select_model` then fell back to M0/M1 without a signal. The
premise "a faster component is gone after one sample" does not hold on a
noise-free curve. Reproduced on Windows (Python 3.12.10, numpy 2.5.4,
scipy 1.18.1) at 744da3d:

| case | 744da3d | 6b1cd93 / bound dropped | oracle |
|---|---|---|---|
| noise-free `exp(-0.5t)+a2 exp(-100t)`, `linspace(0,10,80)` (`beta2*dt_0` 12.7) | M1 0.603309 (a2 0.5), M0 1.009265 (a2 2.0) | M2 0.5 | 0.5 |
| driven dephased qubit (`H = sigma_x`, `sigma_-`, `sqrt(gphi/2) sigma_z`, `|+>`, `linspace(0,20,80)`), `beta_D_linear` | 1.93006 / 1.71351 (gphi 20 / 30) | 1.20733 / 1.13622 | Liouvillian gap 1.20733 / 1.13622 |
| same, `beta_D` | M0 5.03697 / 4.05635 | M2 2.04259 / 1.86908 (6b1cd93) | -- |
| noise 1e-3, seeds 0-9, wrong slow rate (M1 0.602-0.604) | 2/10 (beta2 50), 7/10 (beta2 500) | 0/10, 0/10 | 0.5 |

Two remedies were built and measured:

* **Drop the upper bound** (the zero-column clause still refuses the
  ~1e6 endpoint, whose beta2 column underflows to exactly 0 beyond
  `beta*dt_0 ~ 745`). All cases above come out right, but
  `test_multistart_chain1_never_finds_an_identified_m2_beating_m0` (S7,
  `slow`, run by CI) turns red: 4/50 starts reach an identified M2 (fast
  rate 65-280, slow 0.71731) that beats M0 by ~260 AICc units. That is the
  start dependence #127 removed. Not adopted.
* **Option B5 (adopted):** `_select_model` reranks with each refused fit
  admitted at its would-be score (`FitResult.refused_aicc`); if the winner
  changes, the result is `INDETERMINATE` (rate NaN). All wrong values above
  become NaN; S7 stays green (0/50 identified M2 beat M0; the 4 are refused
  and would yield INDETERMINATE if reached from the default start).

B5 withholds rates that the bound-free variant gets right (noise-free case,
qubit case). Which to prefer -- withholding, dropping the bound with S7
amended, or a profile-likelihood test for the fast rate (dossier option B
item 2) -- is a methodology decision and is not made here. Note for it: on
`_chain1` the M2 optimum (slow 0.71731) is closer to the asymptotic
relative-entropy rate `2 * gap = 0.70` than the reported M0 0.753447.

Proof runs for B5 (code at 1a384ea):

| run | result |
|---|---|
| Windows x86-64, Python 3.12.10, numpy 2.5.4, scipy 1.18.1 | ruff, mypy exit 0; 2123 passed, 14 skipped, 2 xfailed, exit 0 |
| Linux x86-64 (Docker python:3.12-slim, Python 3.12.16, numpy 2.5.4, scipy 1.18.1, qutip), tree of c729ca6 | ruff, mypy clean; anchors 21 passed; 2137 passed, 2 xfailed |
| B5 wrapper neutralised (mutation), P1-1 test file | 17 failed, 43 passed, 2 xfailed |

The +30 against the e076f2b runs above are the P1-1 tests.
