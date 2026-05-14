Reproducibility
===============

Every :func:`liouscope.diagnose` call emits a
:class:`liouscope.GovernanceMetadata` record containing a SHA-256 ``run_id``
derived deterministically from the inputs, seed and framework version. The
contract is described in detail in the top-level ``REPRODUCIBILITY.md``.

Locked versions
---------------

================================ ===============================================
String                           Constant
================================ ===============================================
``A1-A12-v3.1``                  :data:`liouscope.TAXONOMY_VERSION`
``D1-D24-Übersicht-v3-2026-04-24`` :data:`liouscope.DIAGNOSTIC_SCHEMA_VERSION`
``1.2.0``                        manifest schema (``MANIFEST_SCHEMA.json``)
================================ ===============================================

Reproducing the paper's table
-----------------------------

.. code-block:: bash

   pip install -e .
   python benchmarks/reproduce_paper.py

The script prints a SHA-256 over the V1-V5 result table that must match
``20104ea1180307a60eadf5df294a76815133255a263a7e3062db72dc514f5cc7`` on
the reference environment (Python 3.11.15, numpy 2.4.4, scipy 1.17.1).
Drift in this hash is treated as a numerical regression.

Benchmark manifest
------------------

``LIOUSCOPE_BENCHMARK_MANIFEST.yaml`` declares the canonical benchmark
entries BM-001 through BM-003b. Run all of them with::

    make benchmarks

The deterministic JSON outputs land under ``benchmarks/output/`` and are
cross-checked against the manifest's ``reproduce.output_hash`` field.

Archival
--------

* **Zenodo DOI**: auto-minted on every tagged GitHub release via the
  ``.zenodo.json`` configuration.
* **Software Heritage**: submit via ``swh save origin`` against the
  GitHub URL; the resulting SWHID (ISO/IEC 18670:2025) populates the
  ``identifiers`` block in ``CITATION.cff``.
