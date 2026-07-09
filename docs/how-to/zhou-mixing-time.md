# How to use the D24 Zhou mixing-time predictor

**Goal:** compute universal lower and upper bounds on the mixing time from
the gap and the Petermann factor — and report them with the correct claim
status.

## Opt-in by design

D24 is deliberately **not** part of `diagnose()`. It lives in the frozen
opt-in module `liouscope._zhou`:

```python
import numpy as np
from liouscope._zhou import compute_zhou_predictor, mixing_time_upper_bound

result = compute_zhou_predictor(L, epsilon=1e-3)
print(result.mixing_time_lower)   # bracket at accuracy epsilon
print(result.mixing_time_upper)
print(result.converged, result.gap, result.petermann_factor)

# Rescale the upper bound to a different accuracy without re-diagonalising:
print(mixing_time_upper_bound(result, eps=1e-6))
```

If you already ran `diagnose()`, reuse its spectral quantities instead of
recomputing the eigendecomposition:

```python
report = lp.diagnose(L, seed=42)
result = compute_zhou_predictor(
    L,
    epsilon=1e-3,
    gap=report.spectral.gap,                    # D1
    petermann_factor=report.nonnorm.petermann_max,  # D9
)
```

The bounds have the familiar form

$$
t_{\text{lower}} = \frac{\ln(1/\epsilon)}{\Delta}, \qquad
t_{\text{upper}} = \frac{\ln(\sqrt{K_{\max}}/\epsilon)}{\Delta},
$$

with $\Delta$ the Liouvillian gap and $K_{\max}$ the worst-mode Petermann
factor. The gap between them widens exactly when the generator is strongly
non-normal — which is the regime LiouScope exists to flag.

## Say what it is — and what it is not

The predictor's `claim_status` is **`reference-verified-bound-coarser`**:

- The cited reference (Yi-Neng Zhou, *Universal Predictors for Mixing Time
  more than Liouvillian Gap*, arXiv:2601.06256) is independently verified.
- The implementation is in the same family as Zhou's Eq. (16) and exact in
  the normal-mode limit, **but** it uses the Petermann (Schatten-2) factor
  and a global gap/$K_{\max}$ rather than Zhou's per-mode trace-norm factor
  $C_j$. The upper bound is therefore a related, generally **coarser
  surrogate**, not a verbatim Eq. (16) implementation.

If you publish numbers from this module, carry that qualification with them.
The exact differences are documented in the `liouscope._zhou` module
docstring; the numerical behaviour is pinned by `tests/test_zhou.py`.
