# How to cross-check a result against QuTiP

**Goal:** confirm a LiouScope computation with an independent implementation.
QuTiP is an optional extra used *only* for cross-checks — it is not a runtime
dependency and no diagnostic requires it.

## Install

```bash
pip install -e .[qutip]      # or: pip install qutip
```

## Cross-check the generator itself

LiouScope and `qutip.liouvillian` share the column-stacking convention, so the
two matrices must agree elementwise:

```python
import numpy as np
import qutip
import liouscope as lp

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
H, gamma = 0.5 * sx, 0.3

L_ls = lp.build_liouvillian(H, jump_ops=[sz], rates=[gamma])
L_qt = qutip.liouvillian(qutip.Qobj(H), [np.sqrt(gamma) * qutip.Qobj(sz)]).full()

assert np.allclose(L_ls, L_qt, atol=1e-10)
```

Note the rate convention: LiouScope takes the rate $\gamma$ separately, QuTiP
absorbs $\sqrt{\gamma}$ into the collapse operator.

## Cross-check spectrum, gap and steady state

```python
evals_ls = np.linalg.eigvals(L_ls)
evals_qt = np.linalg.eigvals(L_qt)      # same matrix, or use qutip solvers

# Gap: smallest nonzero |Re lambda|
report = lp.diagnose(L_ls, seed=42)
nonzero = evals_qt[np.abs(evals_qt) > 1e-12]
gap_qt = -np.max(nonzero.real)
assert np.isclose(report.spectral.gap, gap_qt, rtol=1e-8)

# Steady state against qutip.steadystate
rho_ss_ls = lp.steady_state(L_ls)
rho_ss_qt = qutip.steadystate(
    qutip.Qobj(H), [np.sqrt(gamma) * qutip.Qobj(sz)]
).full()
assert np.allclose(rho_ss_ls, rho_ss_qt, atol=1e-8)
```

## Cross-check dynamics

For time evolution, compare the LiouScope propagator path against
`qutip.mesolve` expectation values or full states:

```python
t_grid = np.linspace(0.0, 5.0, 50)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho0 = np.outer(plus, plus.conj())

result = qutip.mesolve(
    qutip.Qobj(H), qutip.Qobj(rho0), t_grid,
    c_ops=[np.sqrt(gamma) * qutip.Qobj(sz)],
)
# Compare e.g. trace distance to the steady state per time point against
# the trajectory LiouScope's relaxation layer fits.
```

## What already runs in CI

You rarely need to hand-roll these checks: the repository's differential
suite `tests/test_qutip_spectral_oracle.py` pins spectrum, gap, oscillation
frequency and steady state against QuTiP on amplitude-damping, driven
dephasing and thermal-qubit systems, and the dedicated `ci-qutip.yml` workflow
runs it on Python 3.11/3.12 as a required check. `examples/quickstart.py`
performs the generator cross-check on every run and prints `PASS`/`FAIL`.

If you add a new physics path, follow the same pattern: one analytic oracle
test *plus* one QuTiP differential test, so an error in either implementation
cannot hide.
