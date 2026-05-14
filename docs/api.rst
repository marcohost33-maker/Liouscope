API Reference
=============

Top-level
---------

.. autosummary::
   :toctree: _autosummary

   liouscope.build_liouvillian
   liouscope.steady_state
   liouscope.diagnose
   liouscope.classify_mechanism
   liouscope.seed_everything

Result containers
-----------------

.. autosummary::
   :toctree: _autosummary

   liouscope.DiagnosticReport
   liouscope.SpectralResult
   liouscope.NonNormalityResult
   liouscope.RelaxationResult
   liouscope.ResolventResult
   liouscope.TransientResult
   liouscope.LepResult
   liouscope.MpembaResult
   liouscope.UncertaintyResult
   liouscope.ClassificationResult
   liouscope.GovernanceMetadata
   liouscope.FitResult

Diagnostics layers
------------------

.. autosummary::
   :toctree: _autosummary

   liouscope.diagnostics.compute_spectral_layer
   liouscope.diagnostics.compute_nonnormality_layer
   liouscope.diagnostics.compute_relaxation_layer
   liouscope.diagnostics.compute_resolvent_layer
   liouscope.diagnostics.compute_transient_layer
   liouscope.diagnostics.compute_lep_layer
   liouscope.diagnostics.compute_mpemba_layer
   liouscope.diagnostics.compute_uncertainty_layer

Numerics primitives
-------------------

.. autosummary::
   :toctree: _autosummary

   liouscope.numerics.vec
   liouscope.numerics.unvec
   liouscope.numerics.alicki_adjoint
   liouscope.numerics.hs_adjoint
   liouscope.numerics.eig_nonhermitian

Sparse path
-----------

.. autosummary::
   :toctree: _autosummary

   liouscope.sparse.build_sparse_liouvillian
   liouscope.sparse.sparse_steady_state
   liouscope.sparse.sparse_spectrum
   liouscope.sparse.chi1_lower_bound

Post-submission (v0.2.1)
------------------------

.. autosummary::
   :toctree: _autosummary

   liouscope._zhou.compute_zhou_predictor
