# PREREG: M2 fitted identifiably (follow-up to PR #127, option B)

Status: PRE-REGISTERED. Committed before the identifiability gate, the
regression tests and the selection-level runs below exist. Base commit:
`6b1cd93` (head of PR #127). A later edit of this file is visible in git.

## Disclosure: what was measured before this file

One exploratory run preceded this file and informed the choice of the
optimiser scaling (item 2). It fitted ONLY the M2 candidate, never the model
selection, with five `x_scale` variants (Windows x86-64, Python 3.12,
numpy 2.5.4, scipy 1.18.1):

| case | `jac` (today) | `1.0` | characteristic | log-rate |
|---|---|---|---|---|
| `_chain1` M2 | beta2 = 1.055e6, status 3, logL 472.29 | beta2 = 7.67, logL 484.79 | same as `1.0` | beta2 = 83..99, logL 473.40 |
| resolved two-rate, x 1e-40 / 1e+40 | correct (beta2 2.998) | WRONG (beta2 1.525, status 3) | correct | correct |

"characteristic" is `x_scale = [max|y|, 1/T, max|y|, 1/T]` (amplitudes in
units of the curve, rates in units of the inverse window `T`). It is the
variant carried forward: it reaches the better optimum on `_chain1` and keeps
the amplitude invariance that `x_scale = 1.0` breaks.

## Question

`_chain1` (one transverse-field site with amplitude damping,
`tests/test_pr127_default_window_migration.py`) on the gap-scaled default
window selects M2 today because the second M2 rate runs off to beta2 ~ 1e6,
where its Jacobian column is zero and the solver stops on `xtol` at a
platform-dependent point. Does an identifiable M2 fit (scaling + gate)
remove that artefact without removing genuine two-rate fits?

## Fixed design

* **Scaling (item 2).** M2 is optimised with the characteristic scale above,
  in every caller of `fit_gls_ar1` (fit, bootstrap, jackknife, holdout).
  Every other model keeps `x_scale = "jac"` unchanged.
* **Identifiability gate (item 3).** An M2 fit is a model-selection
  candidate only if ALL hold, otherwise it is "not identified", gets
  `aicc = inf` and is excluded from every selection branch, including the
  all-infinite fallback and the fewest-parameters fallback of
  `_select_model`:
  1. all four parameters finite;
  2. each rate inside the resolved window
     `ALPHA_SEED_FLOOR_FRAC <= beta_i * T` (repo constant 5e-3: slower is
     not distinguishable from a constant on this window) and
     `beta_i * dt_0 <= 1 / MIN_SAMPLES_PER_FAST_EFOLD` (repo constant 1.0;
     `dt_0` = first grid step, where a fast component lives);
  3. the column-equilibrated model Jacobian at the fit has no zero or
     non-finite column and condition number `<= eps**-0.5` (~6.7e7; above
     it `J^T J` is singular in float64).
  The thresholds are existing repo constants or a float64 property; no new
  tuned number.
* **Multistart (item 4).** 50 starts, `numpy.random.default_rng(20261108)`,
  `log(beta_i)` uniform on `[log(5e-3 / T), log(10 / dt_0)]`, amplitudes
  `0.5 * max|y|` each, same scaling and gate as production.

## Cases and seeds

* C0 `_chain1`, default window (80 points, uniform, dt = 0.3617), no noise.
* C1 resolved two-rate: `y = exp(-0.5 t) + 0.5 exp(-3 t)`, `t = linspace(0, 10, 80)`
  (beta2 dt = 0.38), white noise sigma 1e-3, seeds 11..20.
* C2 stepped-over two-rate: as C1 with beta2 = 50 (beta2 dt = 6.3), seeds 11..20.
* C3 single exponential truth: `y = exp(-0.5 t)`, same grid and noise, seeds 11..20.

Selection for C1..C3 through `relaxation._dominant_rate` (shipped
`_select_model`, default data error).

## Metrics recorded

Per fit: `status`, `cost`, `optimality`, `nfev`, equilibrated singular
values, `beta`, identification verdict and reason, `logL`, `aicc`; per case:
`selected_model`, `beta_D`.

## Success criteria (fixed now)

* S1 C0: `aicc_model == "M0"` and `beta_D` within `rel 1e-6` of 0.753447,
  on Windows AND on Linux (Docker). NaN fails.
* S2 C0: the M2 fit is "not identified" (rate window) with today's and with
  the new scaling.
* S3 C1: M2 identified AND selected in >= 9 of 10 seeds.
* S4 C2: M2 identified in 0 of 10 seeds, selected in 0 of 10.
* S5 C3: descriptive only (selected-model counts reported, no pass/fail).
* S6 amplitude invariance: M2 rates on `c * y` (C1 seed 11,
  c in {1e-40, 1, 1e40}) agree to `rel 1e-9`; `tests/test_gls_amplitude_scale.py`
  stays green.
* S7 multistart: C0 -- no start yields an identified M2 with lower AICc than
  M0; C1 seed 11 -- >= 45 of 50 starts identified and within `rel 1e-6` of
  the best cost.
* S8 full suite: no new failure against the `6b1cd93` baseline except tests
  that pin the artefact itself (`tests/test_pr127_default_window_migration.py`),
  which are rewritten; anchors green.

Any miss is reported as a miss, not re-tuned.
