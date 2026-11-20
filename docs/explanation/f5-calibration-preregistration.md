# Preregistration: rate-unit invariance and the F5 calibration study

*Issue #101 slice C. `claim_status: pending`. Registered 2026-11-20, before
any threshold was fitted.*

This page fixes the design of the study that has to pass before the A10/F5
decision path may change. It declares the question, the control families, the
split, the metrics and the acceptance criteria in advance. It also records the
baseline: what the classifier does today on the calibration split.

It changes no threshold and no verdict. Until the study passes and an
independent physics review has confirmed it, LiouScope does not claim that a
mechanism class is invariant under a change of rate units.

## The question

A Liouvillian carries a rate dimension. Measuring rates in another unit,
`L -> cL` with `c > 0` and every time `t -> t / c`, describes the same
physics. The mechanism class must therefore not depend on `c`. That is an
exact symmetry, not a statistical expectation, which makes it a metamorphic
relation: the correct class is unknown, but the two outputs must agree.

Three hypotheses are tested, in this order.

- **H-INV.** The class and the verdict of a system are the same for every
  rate factor.
- **H-NEG.** F5 does not fire on systems without the phantom mechanism.
- **H-POS.** F5 fires on systems with it.

H-INV comes first because H-NEG and H-POS are not well defined while the
answer depends on the unit.

## How it is measured

`benchmarks/issue101_rate_unit_census.py` runs the full pipeline on `c * L`
for `c` in `1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3`, with the relaxation time
grid `linspace(0, 10, 80) / c`. With the grid rescaled, every trajectory is
the same curve in rescaled time. A difference between two rate factors then
comes from an estimator grid or a threshold that is absolute in the rate.

The library default grid is an absolute `linspace(0, 10, 80)`. A caller who
passes no grid gets a second, separate unit dependence from it. The census
measures that with `--default-grid`; fixing it is PR #115, not this study.

`tests/test_issue101_rate_unit_census.py` pins the harness: the family
premises, the disjoint splits, the exact application of the relation and a
reproducible result digest.

## Control families

The labels are design labels. They follow the control list of the #101
re-audit addendum and are subject to the physics review like everything else
here.

| Family | System | F5 label | In the harness |
|---|---|---|---|
| N1 | thermal two-level system, detailed balance | negative | yes |
| N2 | pure dephasing, conserved populations, normal | negative | yes |
| N3 | damped qubit next to an undamped qubit, gapless | negative | yes |
| N4 | resonantly driven, amplitude-damped qubit | negative | yes |
| P1 | d = 4 directional ladder (the anchor suite's phantom control) | positive | yes |
| N0 | zero operator | negative | required before freeze |
| P2 | defective zero mode (Jordan block at zero) | positive | required before freeze |
| P3 | Liouvillian skin effect chain, three system sizes | positive | required before freeze |
| N5 | non-normal but not phantom, three system sizes | negative | required before freeze |

P3 and N5 are the system-size families that expose the dimension bias of a
Frobenius-norm scale. The skin-effect family follows Haga, Nakagawa, Hamazaki
and Ueda, Phys. Rev. Lett. 127, 070402 (2021).

A positive control that is the anchor suite's own example (P1) shows that a
rule keeps known behaviour. It does not show that the rule detects the
mechanism in general. P2 and P3 are required for that reason.

## Split

Every system has an index. Even indices are the calibration split, odd
indices the holdout split. A rule may be developed on the calibration split
without limit.

The holdout split is evaluated once, after the candidate rule and the
criteria below are frozen in a commit and the physics review has confirmed
the family labels. If the holdout run fails, the result is reported as it is.
A second attempt needs new holdout indices and an entry in the deviations
log.

## Candidate rules

A candidate rule may consume only quantities that are dimensionless under
`L -> cL`. An absolute threshold on a rate is not admissible. The
scale-relative diagnostics of slice A (`henrici_relative`, `kreiss_scaled`,
`pseudospectral_radius_rel`, `pseudospectral_abscissa_rel`) are candidates,
not conclusions. The baseline below shows that all four are invariant on the
five families. It says nothing yet about whether they separate positives from
negatives.

No threshold is tuned to preserve an existing anchor. If the frozen rule
changes an anchor, the change goes into a dedicated PR with the physics
rationale.

## Acceptance criteria

Declared before fitting. The numeric values of C2 and C3 are proposals and
are confirmed or changed at the physics review, before the holdout is opened.

| ID | Criterion | Threshold |
|---|---|---|
| C1 | Class and verdict identical over all seven rate factors | every system, both splits |
| C2 | F5 fired on a negative system, at any rate factor | none on calibration; none on holdout |
| C3 | F5 fired on a positive system, at every rate factor | every system of P1 and P2; at least 9 in 10 of P3 |
| C4 | Decision keys consumed by the rule | relative range `(max - min) / max` over the rate factors at most `1e-9` |
| C5 | Anchor suite | unchanged, or changed in a dedicated PR |

C1 and C4 have no tolerance for exceptions because they test an exact
symmetry. Any violation is a defect in an estimator or a threshold.

C2 is a count of zero, so its meaning depends on the sample size. With zero
events in `n` independent systems, the one-sided 95 % upper bound on the rate
is about `3 / n`. The holdout therefore needs at least 60 negative systems per
family to support "below 5 %", and the report states the bound, not "no false
positives".

## Baseline, calibration split

Measured 2026-11-20 on x86-64, Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1,
at the head of PR #197, 12 systems per family, time grid `t_ref / c`.
Result digest `49ae198cf616899865fa448d5469c2dfd467debdebb25243e93a01b85b129bd3`.

| Family | Label | n | Class stable | Verdict stable | F5 fired at `c` = 1e-3 … 1e3 |
|---|---|---|---|---|---|
| N1 | negative | 12 | 5 | 5 | 0, 0, 0, 0, 6, 6, 7 |
| N2 | negative | 12 | 7 | 9 | 0, 0, 0, 0, 0, 0, 0 |
| N3 | negative | 12 | 12 | 6 | 0, 0, 0, 0, 0, 0, 0 |
| N4 | negative | 12 | 11 | 11 | 0, 0, 0, 0, 0, 0, 0 |
| P1 | positive | 12 | 0 | 0 | 0, 0, 0, 12, 12, 12, 12 |

Against the criteria:

- **C1 fails.** The class is stable for 35 of 60 systems, the verdict for 31.
- **C2 fails.** Thermal systems (N1) are never F5 at `c <= 1` and are F5 in 6,
  6 and 7 of 12 cases at `c = 10, 100, 1000`. The unit alone creates the label.
- **C3 fails.** The positive control fires in 12 of 12 cases at `c >= 1` and
  in 0 of 12 at `c < 1`.
- **C4 fails for the keys the rule consumes today.** `henrici_eta` has a
  relative range of 1 over the rate factors: it is a rate, compared with the
  pure number 1. The legacy reach `pseudospectral_radius / gap` has a range
  of 0.42, 0.49 and 0.35 on N1, N4 and P1, and of 1 on N2.

The slice A candidates, largest relative range per family. Values at or below
`1e-12` count as zero, because round-off zeros of about `1e-16` otherwise read
as variation:

| Key | N1 | N2 | N3 | N4 | P1 |
|---|---|---|---|---|---|
| `henrici_relative` | 1.1e-15 | 0 | 1.8e-15 | 2.9e-15 | 1.0e-15 |
| `kreiss_scaled` | 2.0e-13 | 0 | 2.8e-13 | 4.5e-13 | 2.8e-14 |
| `pseudospectral_radius_rel` | 5.9e-16 | 1.1e-15 | 3.5e-15 | 1.9e-15 | 3.9e-15 |
| `pseudospectral_abscissa_rel` | 0 | 5.4e-15 | 0 | 0 | 0 |

All four meet C4 on all five families, with a margin of more than three
decades. `pseudospectral_abscissa_rel` is zero to round-off on N1, N3, N4 and
P1, so on these families it is invariant but carries no information.

Not every instability is in the F5 path. On N2 the winner moves between the
A1, A5, A8 and A12 rungs with the rate factor although the time grid is
rescaled. Those rungs read the relaxation fit. Issues #111 and #124 document
that its convergence is not rate-unit invariant; the cause of these
particular changes was not isolated here. C1 covers them too.

### With the library default time grid

The same census with the default grid, an absolute `linspace(0, 10, 80)`,
which is what a caller gets who passes no `t_grid`. Result digest
`1c55a36f8a5a405ee85ebb2da901e54a3dddda543ed705cf4ab08ff93daf2e87`.

| Family | n | Class stable | Verdict stable | F5 fired at `c` = 1e-3 … 1e3 |
|---|---|---|---|---|
| N1 | 12 | 0 | 4 | 0, 0, 0, 0, 6, 6, 7 |
| N2 | 12 | 1 | 4 | 0, 0, 0, 0, 0, 0, 0 |
| N3 | 12 | 5 | 0 | 0, 0, 0, 0, 0, 0, 0 |
| N4 | 12 | 0 | 4 | 0, 0, 0, 0, 0, 0, 0 |
| P1 | 12 | 0 | 0 | 0, 0, 0, 12, 12, 12, 12 |

The class is stable for 6 of 60 systems, against 35 of 60 with the rescaled
grid. The F5 counts are identical in both runs, which is consistent with an F5
path that does not read the time grid. The additional instability then comes
from the fixed observation window: for rates of order one at `c = 1`, the default grid covers
a hundredth of a relaxation time at `c = 1e-3` and ten thousand of them at
`c = 1e3`. That is the subject of PR #115. This study uses the rescaled grid
throughout.

## Procedure

1. **Phase 0, this page.** Harness, families N1 to N4 and P1, baseline.
2. **Phase 1.** Add N0, P2, P3 and N5. Isolate the rate-unit dependence of
   the relaxation-fit rungs seen on N2.
3. **Phase 2.** Develop a candidate rule on the calibration split only.
4. **Phase 3.** Freeze rule and criteria in a commit. Physics review of the
   labels, the rule and C2/C3.
5. **Phase 4.** One holdout run. Report as measured.
6. **Phase 5.** Classifier switch in a dedicated PR, with anchors.

## Deviations log

None.

## References

- T. Haga, M. Nakagawa, R. Hamazaki, M. Ueda, "Liouvillian skin effect:
  slowing down of relaxation processes without gap closing", Phys. Rev. Lett.
  127, 070402 (2021), doi:10.1103/PhysRevLett.127.070402.
- T. Mitchell, "Computing the Kreiss constant of a matrix", SIAM J. Matrix
  Anal. Appl. (2020), doi:10.1137/19M1275127. A fixed grid gives a lower
  bound, not a certified value.
- U. Kanewala, J. M. Bieman, "Testing scientific software: a systematic
  literature review", arXiv:1804.01954. Metamorphic relations as the test
  oracle where the correct output is unknown.
- J. A. Hanley, A. Lippman-Hand, "If nothing goes wrong, is everything all
  right?", JAMA 249, 1743 (1983). The `3 / n` bound for zero events.
