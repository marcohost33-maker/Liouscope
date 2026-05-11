<!-- Thanks for contributing to LiouScope. -->

## Summary

<!-- One paragraph: what changes and why. -->

## Type of change

- [ ] Bug fix (numerical, API, or correctness)
- [ ] New diagnostic / mechanism class / fit model
- [ ] Performance improvement
- [ ] Documentation / metadata
- [ ] Build / CI / security

## Correctness anchors potentially affected

Tick all that apply -- the regression gate in `tests/test_anchors.py` must
still pass for every checked item:

- [ ] A. Column-stacking
- [ ] B. GNS Gram
- [ ] C. Alicki adjoint direction
- [ ] D. Eigensolver
- [ ] E. SuperLU resolvent
- [ ] F. AICc model selection
- [ ] G/H. Statistics (bootstrap / N_eff)
- [ ] I. Conjugate-pair LEP
- [ ] J. supp-check
- [ ] K. HS vs pi-adjoint
- [ ] L/M/N. Versioning / D11 vs D11b / D24

## Reproducibility

- [ ] `pytest --cov=liouscope --cov-fail-under=80` passes locally
- [ ] `ruff check src/ tests/` passes
- [ ] `python benchmarks/reproduce_paper.py` SHA-256 unchanged
      (or change explained below)

<!-- If the SHA-256 changes, explain why and update CHANGELOG.md and
     `LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml` accordingly. -->

## Checklist

- [ ] Tests added / updated for the change
- [ ] Docstrings / README / CHANGELOG updated where relevant
- [ ] No new long-running tests without a `slow` marker
