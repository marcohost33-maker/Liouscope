# Explanation

*Understanding-oriented background and design rationale.*

These pages explain *why* LiouScope is shaped the way it is. The quantitative
methodology lives in the module docstrings (see {doc}`../reference/index`);
the release-gate audit trail lives in `docs/RELEASE_AUDIT_v0.5.0.md` in the
repository.

- {doc}`no-single-number` — why the Liouvillian gap alone misleads, and why
  every LiouScope result is a structured report instead of a rate.
- {doc}`layers-and-taxonomy` — the D1–D24 diagnostic layers, the A1–A12
  mechanism taxonomy and the evidence-graded verdict vocabulary.
- {doc}`reproducibility` — the audit chain: seeds, SPEC 7, manifests and
  fail-closed defaults.
- {doc}`f5-calibration-preregistration` — the preregistered design of the
  study that has to pass before the A10/F5 decision path may change, and the
  measured rate-unit baseline.

```{toctree}
:hidden:
:maxdepth: 1

no-single-number
layers-and-taxonomy
reproducibility
f5-calibration-preregistration
```
