# LiouScope -- Negative Results Register

**Source:** v2.0 Konsolidierter Gesamtbericht (Teil 18 / 0Meta 4 SURVIVAL_ITEMS)
plus the hardened-review-v2.1 self-audit (NR-153..158) plus repository-
specific findings encountered while materialising the package.

This file is the **negative-results spiegel** for the repository: it enumerates
mistakes that were committed, then corrected, and patterns that must not be
re-introduced. Pair with ``tests/test_anchors.py`` -- every anchor regression
gate maps to one or more NR entries.

---

## NR-001 to NR-010 -- Hallucination audit (canonical, Spec Teil 18.1)

| NR | Failure mode | Correction |
|----|--------------|------------|
| NR-001 | AdS/CFT connection to Liouvillian gap claimed | Out of scope. |
| NR-002 | Kerr-QNM analogy stated as rigorous | Illustrative only. |
| NR-003 | MLSI verification claimed as feature | MLSI is out of scope. |
| NR-004 | Gradient-flow structures claimed | Only reversible sector. |
| NR-005 | BiCGSTAB used as resolver | SuperLU (anchor E / E10). |
| NR-006 | GNS via ``(L + L^H)/2`` | ``G = rho_ss^T (x) I`` (FIX-1). |
| NR-007 | Pauli-sector rate as ``Delta_s`` | Distinct objects (FIX-4). |
| NR-008 | ``order='C'`` in flatten / reshape | Column-stacking only (FIX-2). |
| NR-009 | M3 monolithic (poly + osc) | Split M3a / M3b with Prony seed. |
| NR-010 | Non-Markovian QME claimed | Out of scope (NR-11). |

## NR-013 to NR-023 -- Validator anti-regressions (MOC004)

See ``LIOUSCOPE_EVIDENCE_LOCK_REGISTER.csv`` for the gate codes. Every entry
here is encoded as a regression test in ``tests/test_anchors.py``.

| NR | Failure mode | Validator rule |
|----|--------------|----------------|
| NR-013 | Non-Markovian overreach | VR-013 -- core scope only |
| NR-014 | Floquet overreach | VR-014 -- separate roadmap |
| NR-015 | Gradient-flow overclaim | VR-015 -- only reversible |
| NR-016 | Thermodynamic-limit overclaim | VR-016 -- finite-size only |
| NR-017 | ED scaling oversold | VR-017 -- declare ceiling |
| NR-018 | beta_D fit misuse | VR-018 -- record fit metadata |
| NR-019 | Boundary / bulk merged | VR-019 -- strict split |
| NR-020 | Single initial state confounder | VR-020 -- Haar sweep |
| NR-021 | Regime fit unsplit | VR-021 -- classify first |
| NR-022 | Formula / basis ambiguous | VR-022 -- lock both |
| NR-023 | Parallel ARPACK unsafe replay | VR-023 -- serial reference |

## NR-027, NR-035, NR-036, NR-038, NR-040, NR-041..060 -- Code NRs

(Documented in the main v2.0 consolidated report; the relevant ones are
mirrored in the gates ``G01..G30`` of ``LIOUSCOPE_EVIDENCE_LOCK_REGISTER.csv``.)

---

## NR-112 to NR-135 -- T17 hardening NRs (Drive v2.0 FINAL canon)

| NR | Lesson |
|----|--------|
| NR-112 | Positive correlation is **not** automatic evidence -- the kappa_trans lesson. |
| NR-113 | ``kappa(V)`` is the correct skin-effect diagnostic, **not** gap-to-abscissa. |
| NR-114 | Real-physics Liouvillians have massive eigenvalue degeneracy (54.7% T9-fallback on XY chains). |
| NR-132 | Scale-invariance bug in Bohr-AP. |
| NR-133 | Multi-model fit without a quality gate. |
| NR-134 | Meta-consolidation is itself a distinct value layer. |
| NR-135 | Prior turn skipped its own recommendation. |

---

## NR-153 to NR-160 -- Hardened-review-v2.1 corrections

The following NRs were discovered while comparing my own previous reactions to
the Drive-canonical Evidence Manifest v1.2. They are now permanently logged.

| NR | Failure mode in earlier reaction | Resolution applied in this repo |
|----|----------------------------------|---------------------------------|
| NR-153 | A previous reaction cited "Konsolidierter Finaler Statusbericht V2" and "Umfassender Finaler Projektbericht" as sources. ``README_CANON_0LIOU_PLUS_v1_0.md`` lists both as *DO NOT USE AS CANON*. | The canonical-source convention is encoded explicitly in this file. The repo never references those documents. |
| NR-154 | D2b (KMS-Gap) and D11b (Resolvent Peak ``R_lb,max``) were absent from an earlier "D1-D20 + D24" schema string. | ``DIAGNOSTIC_SCHEMA_VERSION = "D1-D24-Übersicht-v3-2026-04-24"`` includes both. Tests ``test_anchor_B_gns_gram_form`` and ``test_anchor_M_d11_is_bohr_ap`` cover them. |
| NR-155 | Alicki-adjoint tensor-contraction direction (``rho`` vs ``rho^{-1}``) was the most critical T17 fix and was missing from an earlier review. | Anchor C is in ``tests/test_anchors.py`` and the symmetrisation lives in ``src/liouscope/numerics/adjoint.py``. |
| NR-156 | ``core_scope`` written as "GKSL" only. Manifest v1.2 says "finite-dimensional time-homogeneous GKSL / QMS". | Now spelt "GKSL / QMS" in README, CITATION, codemeta and ``LIOUSCOPE_RELEASE_EVIDENCE_MANIFEST.yaml``. |
| NR-157 | Paper was referenced as v09 or as "v1.0". Drive shows the canonical paper is ``paper1_v10.tex`` (0LS6). | README cross-references the 0LS6 submission package. |
| NR-158 | 0LS7 Drive folder is empty -- previously taken as anomaly. | Documented as "reserved for next session/phase", not a missing artefact. |
| NR-159 | Strong quantum Mpemba effect is sensitive to preparation errors (Mackinnon & Paternostro, NJP 2026). | Added preparation-sensitivity caveat to ``src/liouscope/diagnostics/mpemba.py`` docstring; classification weights for A11 demand both ``c_1`` close to zero **and** a clean initial state. |
| NR-160 | Hypocoercivity (Carlen-Maas, PRL 134, 140405, 2025) appeared in the paper related work but was missing as a NR. | Logged here as a forward-looking reference for related-work scope; not implemented as a diagnostic in v0.2.0. |

---

## NR-201 to NR-202 -- Repository-build NRs (this session)

| NR | Failure mode | Resolution |
|----|--------------|------------|
| NR-201 | First GNS construction lived in the Schrödinger picture and produced a positive eigenvalue of ``L_sym`` (unphysical) on V1 qutrit. | Switched to Heisenberg-picture construction: ``L_HS_sym = (L^H + G^{-1} L G) / 2``. Now Hermitian under Gram-conjugation and kills ``vec(I)``. |
| NR-202 | Initial ``sigma_-`` formula ``(sigma_x - i sigma_y)/2`` yielded ``|1><0|`` (raising), so amplitude-damping V3 had its steady state on the excited level. | Corrected to ``(sigma_x + i sigma_y)/2 = |0><1|``. Fix in ``core.jumps`` and ``examples.v3_amplitude_damped_qubit``. |

---

## Anti-regression protocol

For every NR above:

1. Read the affected entry before refactoring the relevant module.
2. Either keep the existing anchor regression test or add a new one in
   ``tests/test_anchors.py``.
3. After the change: ``make anchors`` must pass.
4. Append a line to this register if a new failure mode is uncovered.
