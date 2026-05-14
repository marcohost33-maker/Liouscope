Architecture
============

Six-layer pipeline (S/N/R/U/C/G)
--------------------------------

================== ================== ====================================
Layer              Letter             Diagnostics
================== ================== ====================================
Spectral            S                 D1, D2, D2b, D3, D4
Non-normality       N                 D8, D9, D10, D11
Relaxation          R                 D5, D6, D7, D7b, M0-M3b
Uncertainty         U                 U0, U1, U2
Classification      C                 A1-A12, F1-F5
Governance          G                 SHA-256 run-id, version pinning
================== ================== ====================================

Twenty diagnostics D1-D20
-------------------------

See :doc:`api` for the public symbols. The diagnostic-schema version
locked across releases is::

    DIAGNOSTIC_SCHEMA_VERSION = "D1-D24-Übersicht-v3-2026-04-24"

Mechanism taxonomy A1-A12
-------------------------

.. list-table::
   :header-rows: 1

   * - Code
     - Mechanism
   * - A1
     - Asymptotic-gap-controlled
   * - A2
     - Symmetrised-gap-corrected transient
   * - A3
     - Overlap/eigenvector-amplified
   * - A4
     - Skin-affected
   * - A5
     - Metastable plateau
   * - A6
     - Accelerated decay / operator spreading
   * - A7
     - Weak-dissipation singular
   * - A8
     - Oscillatory transient
   * - A9
     - Prethermalization-affected
   * - A10
     - Phantom relaxation
   * - A11
     - Non-normal Mpemba
   * - A12
     - Mixed / unresolved

``TAXONOMY_VERSION = "A1-A12-v3.1"``.

Gap-failure families F1-F5
--------------------------

* **F1** Mori-Shirai overlap (PRL 125, 230604, 2020)
* **F2** Liouvillian skin effect (PRL 127, 070402, 2021)
* **F3** Symmetrised gap (PRL 130, 230404, 2023)
* **F4** Quantum Mpemba effect (PRL 127, 060401, 2021)
* **F5** Phantom relaxation (arXiv:2306.07876, 2023)

Correctness anchors A-N
-----------------------

Every anchor maps to a documented historical bug or external audit finding
and is encoded as a regression test in ``tests/test_anchors.py``. Failure of
any anchor blocks release. See ``LIOUSCOPE_NEGATIVE_RESULTS_REGISTER.md``
for the full anti-regression log.

Mpemba demotion (Mackinnon-Paternostro NJP 28, 2026)
----------------------------------------------------

Strong quantum Mpemba is sensitive to preparation errors. The classifier
demotes an A11 verdict to A12 whenever
``lep.initial_state_sensitivity > 0.05`` (see
:data:`liouscope.diagnostics.classification.MPEMBA_SENSITIVITY_THRESHOLD`).
