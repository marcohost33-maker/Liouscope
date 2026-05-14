Quickstart
==========

Install
-------

.. code-block:: bash

   pip install liouscope[qutip,figures]

Minimal usage
-------------

.. code-block:: python

   import numpy as np
   import liouscope as ls

   sx = np.array([[0, 1], [1, 0]], dtype=complex)
   sz = np.array([[1, 0], [0, -1]], dtype=complex)
   H = 0.5 * sx
   L = ls.build_liouvillian(H, [sz], [0.3])
   plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
   rho0 = np.outer(plus, plus.conj())

   report = ls.diagnose(L, rho_initial=rho0, bootstrap_B=100, seed=42)

   print("Delta            :", report.spectral.gap)
   print("GNS gap           :", report.spectral.gns_gap)
   print("KMS gap           :", report.spectral.kms_gap)
   print("Mechanism class  :", report.classification.a_class)
   print("Run ID            :", report.governance.run_id[:16])

QuTiP cross-check
-----------------

A typical defensive pattern: assert that ``build_liouvillian`` agrees with
``qutip.liouvillian().full()`` on a small system to catch silent
column-stacking bugs.

.. code-block:: python

   import qutip

   H_qt = qutip.Qobj(H)
   c_ops = [np.sqrt(0.3) * qutip.Qobj(sz)]
   L_qt = qutip.liouvillian(H_qt, c_ops).full()
   assert np.allclose(L, L_qt, atol=1e-10), "Column-stacking mismatch!"

Command-line
------------

.. code-block:: bash

   python -m liouscope version
   python -m liouscope info
   python -m liouscope diagnose path/to/L.npy --output report.json
