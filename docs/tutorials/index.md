# Tutorials

*Learning-oriented, step-by-step lessons.*

These tutorials assume a working install (`pip install -e .` from the
repository root; add `[qutip]` for the optional cross-checks) and basic
familiarity with NumPy. Every code block is runnable as-is.

- {doc}`first-diagnostic-run` — build a GKSL generator, call
  `diagnose()`, and read every layer of the resulting `DiagnosticReport`.
- {doc}`reproducible-runs` — seeds, the SPEC 7 `rng` keyword, and the
  SHA-256-stable run manifest.

The runnable companion script is `examples/quickstart.py` in the repository.

```{toctree}
:hidden:
:maxdepth: 1

first-diagnostic-run
reproducible-runs
```
