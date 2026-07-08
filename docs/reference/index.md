# API reference

*Information-oriented technical reference, auto-generated from the package
docstrings via `sphinx.ext.autodoc` + `napoleon`.*

This page documents the public surface re-exported from the top-level
`liouscope` package (`liouscope.__all__`). Internal modules are intentionally
not listed here; import from `liouscope` directly.

## Core construction

```{eval-rst}
.. autofunction:: liouscope.build_liouvillian
.. autofunction:: liouscope.steady_state
```

## Diagnostics

```{eval-rst}
.. autofunction:: liouscope.diagnose
.. autofunction:: liouscope.classify_mechanism
```

## Reproducibility

```{eval-rst}
.. autofunction:: liouscope.seed_everything
.. autofunction:: liouscope.derive_seed
```

## Result containers

```{eval-rst}
.. autoclass:: liouscope.DiagnosticReport
.. autoclass:: liouscope.ClassificationResult
.. autoclass:: liouscope.RelaxationResult
.. autoclass:: liouscope.SpectralResult
.. autoclass:: liouscope.NonNormalityResult
.. autoclass:: liouscope.ResolventResult
.. autoclass:: liouscope.TransientResult
.. autoclass:: liouscope.LepResult
.. autoclass:: liouscope.MpembaResult
.. autoclass:: liouscope.UncertaintyResult
.. autoclass:: liouscope.FitResult
.. autoclass:: liouscope.GovernanceMetadata
```

## Ensemble evidence

```{eval-rst}
.. autoclass:: liouscope.EnsembleEvidence
```
