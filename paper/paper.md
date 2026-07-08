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
  # <!-- TODO Marco: replace "Coworker Research" with the named individual
  #      author(s) and add a real, resolver-verified `orcid:` line per author.
  #      Do not submit with the organisational placeholder. -->
  - name: Coworker Research
    affiliation: 1
affiliations:
  - name: Coworker Research
    index: 1
date: 8 July 2026
bibliography: paper.bib
---

<!--
STATUS: DRAFT for issue #72. Target length for submission is 750-1750 words.
Sections follow the JOSS 2026 requirements (Summary, Statement of need, State of
the field, Software design, Research impact, AI usage disclosure). Every factual
claim below is checkable against the repository at the tagged release; no
benchmark numbers are asserted that are not reproducible from the package.
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
`DiagnosticReport`. It computes a suite of complementary diagnostics — the
peer-review submission set **D1–D20**, defined within a broader `D1–D24` schema
whose upper entries are reserved and not implemented here — spanning spectral,
non-normality, resolvent/pseudospectral, transient, relaxation-fit and
uncertainty layers, and assigns a mechanism label from a twelve-class taxonomy
(`A1–A12`, version `A1-A12-v3.1`). Every run records a reproducible manifest
(seed, framework/schema/taxonomy versions, platform, solver path and a
run-invariant `input_hash`), so a result is re-derivable and citable rather than
an opaque scalar.

# Statement of need

Researchers studying dissipative quantum many-body systems, driven-dissipative
phase transitions and quantum thermal machines routinely need to know whether the
Liouvillian gap actually predicts the observed relaxation time — and, when it does
not, *which* mechanism is responsible. Answering this by hand requires stitching
together spectral decompositions, pseudospectral estimates, non-normality
measures, transient-amplification bounds and multi-exponential fits, each with its
own numerical pitfalls (column-stacking conventions, non-Hermitian eigensolvers,
ill-conditioning near degeneracies). The results are easy to over-interpret: a
small gap can coexist with fast transient decay, and a large gap can hide slow
metastable relaxation. `LiouScope` packages these analyses into one reproducible
pipeline with explicit uncertainty and a fail-closed claim gate, so that a
diagnosis is auditable and comparable across studies rather than a single opaque
number produced by ad-hoc scripts.

# State of the field

General open-quantum-systems toolkits such as QuTiP [@qutip1; @qutip2] provide
excellent primitives for constructing Liouvillians and integrating master
equations, and are the de-facto standard for *time evolution* of open systems.
`LiouScope` is complementary rather than competing: instead of propagating
states, it *characterises the generator itself*, focusing on the non-normal and
pseudospectral structure — analysed in the tradition of Trefethen and Embree
[@trefethen2005] — that determines whether the spectral gap is a faithful
relaxation predictor. The two are naturally used together: a practitioner might
build a model and evolve it in QuTiP, then hand the same Liouvillian to
`LiouScope` for a structured relaxation diagnosis. To the authors' knowledge, no
widely used package couples this specific multi-diagnostic relaxation analysis
with a versioned mechanism taxonomy and auditable run manifests. `LiouScope`
therefore depends only on NumPy and SciPy for its core computations, and can
optionally cross-check selected results against QuTiP, which is exercised as a
dedicated continuous-integration job rather than a runtime dependency.

# Software design

`LiouScope` exposes a small, stable public surface — `build_liouvillian`,
`steady_state`, `diagnose` and `classify_mechanism` — that returns immutable,
fully typed result containers (`SpectralResult`, `NonNormalityResult`,
`ResolventResult`, `TransientResult`, `RelaxationResult`, `UncertaintyResult`
and the aggregating `DiagnosticReport`). Internally the diagnostics are grouped
into six layers, each an independent module with its own correctness anchors, so
that a single diagnostic can be validated, replaced or extended without
destabilising the others.

Design choices are deliberately conservative for numerical reproducibility. The
superoperator is built in the **column-stacking convention** (`order='F'`, via
Roth's identity), with a runtime guard that rejects any other ordering; a
general non-Hermitian eigensolver (`numpy.linalg.eig`, dispatching to LAPACK's
`zgeev` for the complex, non-symmetric $\mathcal{L}$) is used where the operator
is not Hermitian, while genuinely Hermitian sub-problems use symmetric solvers.
Reproducibility follows the Scientific-Python SPEC 7 convention: `diagnose` and
`seed_everything` accept an `rng` keyword (integer, `SeedSequence`, `Generator`
or `BitGenerator`) alongside a legacy `seed`. Each run emits a manifest that
hashes its inputs into a run-invariant `input_hash`, so a manifest alone
identifies and reproduces a run; a regression suite of historical correctness
anchors (labelled **A–N**) pins the reference behaviour of the D1–D20 set and
gates every release.

A distinctive element is the **fail-closed claim gate** around anomaly
classification. Rather than emit an anomaly label whenever a pattern is
consistent with it, ambiguous cases are reported as `UNDEFINED / EXPLORATION`.
For example, a single-state quantum-Mpemba (`A11`) diagnosis on a maximally
mixed steady state is suppressed unless a validated `EnsembleEvidence` object —
binding its manifest digest, initial-state family (`F1–F5`), paired run
identifiers, metric and uncertainty method, and distinct producer/reviewer
attestations — supplies a passing reference-family comparison. A bare caller
assertion is explicitly not evidence and raises. This gate encodes the library's
core thesis at the API level: no single number, and no unearned claim. The
package is released under Apache-2.0.

# Research impact

`LiouScope` is an early-stage research tool. Its impact claim is deliberately
modest and evidence-bound: at the time of writing the package is used and
benchmarked **first-party** by its authors on the D1–D20 diagnostic set, and it
has **no external scientific adoption yet** — a fact stated here plainly rather
than obscured. What the software does provide today is *reproducibility
infrastructure*: by pinning both the numerical result and its mechanism label to
versioned schemas and to a hash-stable run manifest, it makes relaxation
diagnoses comparable across machines, versions and studies, which is a
precondition for any future cross-group use. The released software is archived
and citable via Zenodo [@liouscope_zenodo] and distributed on PyPI. Correctness
is defended by the A–N anchor regressions and by an independent QuTiP
cross-check job in continuous integration; these are internal verification
mechanisms, not external validation, and are described as such.

# AI usage disclosure

This project was developed with substantial assistance from AI coding agents,
and discloses that openly. Portions of the source code, tests and documentation
— including this paper — were co-developed with Claude-based coding agents
operating under the repository's written working agreements (`AGENTS.md`). The
human author retains authorship and responsibility: every AI-assisted change
lands through a pull request that the author reviews and merges, is labelled by
agent provenance and confined to an agent branch namespace (`claude/*`,
`codex/*`), and must pass the same required continuous-integration checks —
lint, type-checking, the full test suite with a coverage floor, and the sacred
A–N anchor regressions — as any human contribution. As an additional quality
layer the project uses cross-family *independent model review* (review by a
model from a different provider); this is an internal review aid and is
explicitly **not** a substitute for external scientific validation or
peer review. No experimental or numerical result in this work was produced
without a reproducible, human-inspectable code path.

# Acknowledgements

We thank the maintainers of NumPy, SciPy and QuTiP, on whose open-source work
`LiouScope` builds, and the Scientific-Python community for the SPEC 7 random-
number-generation guidance adopted here.
<!-- TODO Marco: add funding statement (or state "no external funding") and any
     personal acknowledgements before submission. -->

# References
