# F5 scale-invariance audit — 2026-08-05

Status: research plan; no classifier change; `claim_status: pending`.

## Verified problem

The current A10/F5 branch is not invariant under a pure change of rate units.
The known defect `henrici_eta > 1.0` is only one part of the problem:

- D8 `henrici_eta` is rate-dimensioned.
- D10's numerical Kreiss estimate uses absolute `sigma_lo=1e-3`, `sigma_hi>=1`
  and `omega_max>=1` grid scales.
- D11b uses an absolute `sigma=1e-3` and an absolute frequency-grid floor.
- D13 uses an absolute `eps=1e-3` and absolute `1e-3` grid-span floors.
- The F5 branch consumes D13 as `pseudospectral_radius / gap`, so a fixed
  absolute epsilon changes the represented perturbation class under `L -> cL`.

For positive `c`, the exact pseudospectrum obeys

`Sigma_eps(c L) = c Sigma_{eps/c}(L)`.

Therefore a rate-unit invariant comparison requires a relative perturbation
budget, for example `eps_abs = eps_rel * scale(L)`, and a grid expressed in the
same scale.  The continuous-time Kreiss constant is itself invariant under
positive time rescaling, but a fixed dimensional sampling grid need not preserve
that property numerically.

## Conceptual correction

The maximum modulus of an epsilon-pseudospectrum is not the most direct quantity
for slow or phantom relaxation: it can be dominated by fast modes far from the
origin.  The classifier study must compare at least:

1. relative Henrici departure `dep_F(L) / ||L||_F`;
2. scale-relative continuous-time Kreiss estimate;
3. relative pseudospectral abscissa or gap-directed pseudospectral intrusion;
4. direct transient amplification `sup_t ||exp(tL)||` on a physics-scaled grid;
5. modal-overlap / eigenvector-localisation evidence relevant to phantom
   relaxation.

No single generic non-normality scalar is sufficient evidence for F5.

## Required staged implementation

### Slice A — additive diagnostics, no verdict changes

- Introduce one shared positive rate scale with explicit zero-operator handling.
- Add relative Henrici and scale-relative pseudospectral/Kreiss outputs.
- Preserve legacy fields for comparison and migration.
- Record absolute and relative epsilon/grid definitions in report provenance.

### Slice B — numerical conformance

Metamorphic tests for `c in {1e-10, 1e-5, 1, 1e5, 1e10}`:

- all dimensionless diagnostics invariant within stated numerical tolerance;
- all rate-valued diagnostics scale by `c`;
- time-domain curves agree after `t -> t/c`;
- normal, gapless-normal and zero-operator controls do not acquire F5 support.

Use grid refinement and convergence estimates; a 25x25 grid point result must be
labelled an approximation/lower bound, not an exact pseudospectral radius.

### Slice C — preregistered classifier calibration

- Separate calibration and holdout systems.
- Include known phantom-relaxation positives, Jordan/skin controls, ordinary
  non-normal negatives, normal gapless negatives and system-size families.
- Predeclare FPR/TPR and invariance criteria before threshold fitting.
- Assess thresholds across system size; do not tune to preserve old anchors.
- Switch the classifier only after independent physics review.

## Release gate

`0.6.0` must not claim an invariant F5 classifier while any dimensioned threshold
or absolute diagnostic grid remains in the F5 decision path.  Until the staged
study is complete, F5 should remain research/candidate evidence and the known
unit-rescaling limitation must be documented.
