---
title: 'LiouScope: Multi-diagnostic relaxation analysis for open quantum lattice systems'
tags:
  - Python
  - open quantum systems
  - Lindblad
  - Liouvillian
  - non-normal dynamics
  - pseudospectrum
  - quantum relaxation
  - Mpemba effect
authors:
  # DRAFT (issue #72): JOSS requires named individual authors, each with an
  # ORCID. The organisational placeholder below mirrors CITATION.cff; the human
  # author list and ORCIDs MUST be completed before submission. Do not invent
  # ORCIDs.
  - name: Coworker Research
    affiliation: 1
affiliations:
  - name: Coworker Research
    index: 1
date: 8 July 2026
bibliography: paper.bib
---

<!--
STATUS: DRAFT skeleton introduced by issue #72. Target length for submission is
750-1750 words. Sections follow the JOSS 2026 requirements (Summary, Statement
of need, State of the field, Software design, Research impact, AI usage
disclosure). Placeholders marked TODO must be resolved with verifiable evidence
before submission; no benchmark numbers are asserted here that are not yet
reproducible from the repository.
-->

# Summary

`LiouScope` is a Python framework for diagnosing *how* and *why* a
time-homogeneous Markovian open quantum system relaxes toward its steady state.
Such systems are described by a Gorini–Kossakowski–Sudarshan–Lindblad (GKSL)
generator [@gks1976; @lindblad1976], a linear superoperator $\mathcal{L}$ acting
on density matrices. A widespread shortcut summarises relaxation by a single
number — the *Liouvillian gap*, the spectral abscissa of $\mathcal{L}$. This is
often misleading: because $\mathcal{L}$ is generally **non-normal**, its
eigenvalues do not control transient behaviour, and quantities such as the
pseudospectrum can reveal large transient amplification, slow effective
relaxation, or anomalous phenomena (e.g. the quantum Mpemba effect) that the gap
alone hides [@trefethen2005].

`LiouScope` replaces the single "decay rate" with a layered, auditable
`DiagnosticReport`. It computes a suite of complementary diagnostics
(the D1–D24 schema) — spectral, non-normality, resolvent/pseudospectral,
transient, relaxation-fit and uncertainty layers — and assigns a mechanism label
from a twelve-class taxonomy (A1–A12). Every run records a reproducible manifest
(seed, framework/schema/taxonomy versions, platform and a run-invariant input
hash), so results are re-derivable and citable.

# Statement of need

Researchers studying dissipative quantum many-body systems, driven-dissipative
phase transitions and quantum thermal machines routinely need to know whether the
Liouvillian gap actually predicts the observed relaxation time — and, when it does
not, *which* mechanism is responsible. Answering this by hand requires stitching
together spectral decompositions, pseudospectral estimates, non-normality
measures, transient-amplification bounds and multi-exponential fits, each with its
own numerical pitfalls (column-stacking conventions, non-Hermitian eigensolvers,
ill-conditioning). `LiouScope` packages these into one reproducible pipeline with
explicit uncertainty and a fail-closed claim gate, so a diagnosis is auditable
rather than a single opaque number.

# State of the field

General open-quantum-systems toolkits such as QuTiP [@qutip1; @qutip2] provide
excellent primitives for constructing Liouvillians and integrating master
equations, and are the de-facto standard for time evolution. `LiouScope` is
complementary: rather than propagating states, it *characterises the generator
itself*, focusing on the non-normal and pseudospectral structure that determines
whether the spectral gap is a faithful relaxation predictor. To our knowledge no
widely used package packages this specific multi-diagnostic relaxation analysis
with a versioned mechanism taxonomy and auditable run manifests. `LiouScope`
depends only on NumPy and SciPy for its core, and can cross-check selected results
against QuTiP as an optional extra.

# Software design

`LiouScope` is organised around a small public surface — `build_liouvillian`,
`steady_state`, `diagnose` and `classify_mechanism` — that returns immutable,
fully typed result containers. Internally the diagnostics are grouped into six
layers, each an independent module with its own correctness anchors. Design
choices are deliberately conservative for numerical reproducibility: column-stacked
(`order='F'`) superoperators throughout, a non-Hermitian eigensolver (`zgeev`) for
the generally non-symmetric $\mathcal{L}$, and a run manifest that hashes inputs so
a manifest alone reproduces a run. Reproducibility follows the Scientific-Python
SPEC 7 convention: `diagnose` and `seed_everything` accept an `rng` keyword
(integer, `SeedSequence`, `Generator` or `BitGenerator`) alongside the legacy
`seed`. A regression suite of historical correctness anchors (A–N) pins reference
behaviour and gates every release.

# Research impact

<!-- TODO before submission: JOSS 2026 requires evidence of research impact
(at minimum first-party use, e.g. a preprint or reproducible benchmark set over
D1-D20). Cite the accompanying methods preprint and any benchmark artefacts here
once they exist and are archived. Do not assert benchmark numbers that are not
reproducible from this repository. -->

`LiouScope` is intended to make relaxation diagnoses in open quantum systems
reproducible and comparable across studies by pinning both the numerical result
and its mechanism label to versioned schemas. The released software is archived
and citable via Zenodo [@liouscope_zenodo].

# AI usage disclosure

Portions of this software were developed with the assistance of AI coding agents
under human review, following the repository's agent working agreements
(`AGENTS.md`). All numerical results are gated by an automated correctness-anchor
regression suite and human-reviewed pull requests; AI-authored changes are labelled
and land through the same required CI checks as human contributions.

# Acknowledgements

TODO: acknowledgements and funding statements to be completed before submission.

# References
