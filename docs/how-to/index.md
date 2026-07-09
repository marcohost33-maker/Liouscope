# How-to guides

*Goal-oriented recipes for specific tasks.*

Each guide assumes you already know the basics from the
{doc}`tutorials <../tutorials/index>` and solves one concrete task:

- {doc}`export-validate-manifest` — write, validate and diff run manifests.
- {doc}`qutip-cross-check` — verify a LiouScope result against QuTiP.
- {doc}`ensemble-evidence` — supply typed `EnsembleEvidence` so an A11
  (quantum-Mpemba) call can rise above the single-state evidence floor.
- {doc}`zhou-mixing-time` — opt in to the D24 Zhou mixing-time predictor and
  read its bounds with the correct claim status.

Runnable end-to-end references live in `examples/` and `benchmarks/` in the
repository.

```{toctree}
:hidden:
:maxdepth: 1

export-validate-manifest
qutip-cross-check
ensemble-evidence
zhou-mixing-time
```
