# ROADMAP -- LiouScope-Floquet Extension v1.0

**Status:** ROADMAP / POST-v0.2.0
**Attribution:** Coworker Research / Coworkerz
**Date:** 2026-05-13

The Floquet extension is **not** included in v0.2.0. It is documented here so
that the core release stays focused on time-homogeneous GKSL / QMS systems.

## 1. Scope

Periodic Liouvillian

    L(t + T) = L(t)

Extension of LiouScope-Core (time-homogeneous GKSL / QMS) to periodically
driven open quantum systems.

## 2. Core Objects

### 2.1 Floquet map
``Phi_T = T-ordered exp( integral_0^T L(t) dt )``

### 2.2 Effective generator
``L_F = (1/T) log(Phi_T)``

### 2.3 Branch selection / spectral unwinding
Reference: Dinc, Eckardt, Schnell, PRA 111, 062216 (2025), arXiv:2409.17072.
Minimise the effective micromotion to find the best candidate for a
Lindblad-form effective generator.

**Caveat:** Not universally valid. Fails at high driving frequencies where
Fourier peaks become broadly distributed (PRA 2025 Appendix C).

### 2.4 CPTP and generator-validity checks
- Kossakowski matrix eigenvalues >= 0 for Lindblad form
- Trace preservation: ``Tr(L_F[rho]) = 0`` for all ``rho``
- Hermiticity preservation
- Complete positivity via Choi matrix

### 2.5 Micromotion diagnostics
- Floquet-mode structure
- Stroboscopic vs continuous-time comparison
- Micromotion norm as quality indicator

## 3. Hard exclusions (do not claim under any extension)

- **NO** blind Magnus-as-Lindbladian claims
  (Wolf, Eisert, Cubitt, Cirac, PRL 101, 150402 (2008): deciding
  Markovianity is NP-hard).
- **NO** universal Markovianity claim for periodically driven systems.
- **NO** thermodynamic-limit theorems.
- **NO** high-frequency-regime claims without explicit spectral-unwinding
  validation.
- Non-Markovian extensions remain **out of scope**.

## 4. Validation plan

### 4.1 Minimal test cases
1. Two-level driven damped system (analytically solvable)
2. Small spin chain (N=3) with periodic drive
3. Driven Jaynes-Cummings with decay

### 4.2 Cross-checks
- QuTiP ``floquet_master_equation`` comparison
- CPTP checks at each stroboscopic step
- Branch-stability tests (vary branch index, check Kossakowski positivity)
- Micromotion-norm convergence

### 4.3 Diagnostics extension
- **FD1** -- Floquet-Gap (effective generator eigenvalue gap)
- **FD2** -- Stroboscopic Slowdown Score
- **FD3** -- Branch-Sensitivity Index
- **FD4** -- Micromotion Norm
- **FD5** -- CPTP-Violation Proximity

## 5. Planned module structure

```
liouscope_floquet/
  __init__.py
  floquet_map.py          # Phi_T construction, time-ordered exponential
  spectral_unwinding.py   # branch selection, micromotion minimisation
  cptp_checks.py          # Kossakowski, Choi, trace/hermiticity checks
  sambe_sparse.py         # Sambe-space representation for large systems
  diagnostics.py          # FD1-FD5
  tests/
    test_floquet_minimal.py
    test_two_level.py
    test_cptp.py
    test_branch_stability.py
```

## 6. Key references

- Dinc, Eckardt, Schnell, PRA 111, 062216 (2025) -- spectral unwinding.
- Wolf, Eisert, Cubitt, Cirac, PRL 101, 150402 (2008) -- NP-hardness of
  deciding Markovianity.
- Chen, Hu, Zhang et al., PRB 109, 184309 (2024) -- periodically driven OQS.
- Chen et al., PRL 134, 090402 (2025) -- Floquet Liouvillian NESS.
- Mori & Shirai, PRL 130, 230404 (2023) -- symmetrised gap (core reference).

## 7. Relationship to core

```
LiouScope-Core v0.2.0:
  time-independent GKSL / QMS
  D1-D20 + D2b + D11b + D24-Zhou
  A1-A12-v3.1

LiouScope-Floquet (post-v0.2.0):
  periodic L(t + T) = L(t)
  FD1-FD5 (new diagnostics)
  spectral unwinding, Sambe, CPTP checks
  DEPENDS ON Core v0.2.0 stable
```

## 8. Entry criteria

Floquet work starts only after:

1. Core v0.2.0 public release completed.
2. All P0 evidence locks resolved
   (see ``LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml``).
3. Peer-review feedback incorporated.
4. Core test suite remains green.

## 9. Success criteria

LiouScope-Floquet is v0.1.0-ready when:

1. All four minimal test cases pass.
2. QuTiP cross-check within tolerance.
3. CPTP checks pass for all test cases.
4. At least one non-trivial system (N >= 3) demonstrates a gap-failure
   analogous to the core findings.
5. FD1-FD5 diagnostics implemented and tested.
