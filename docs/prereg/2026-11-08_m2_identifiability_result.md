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
edge, beta2 ~ 48-53 (`beta*dt_0` ~ 6), in 6/10 seeds; its SLOW rate is then
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
