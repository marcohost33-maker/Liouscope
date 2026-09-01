# Why "no single number"

## The textbook picture

For a time-homogeneous GKSL generator $\mathcal{L}$, the **Liouvillian gap**

$$
\Delta = -\max\{\operatorname{Re}\lambda : \lambda \in \sigma(\mathcal{L}),\ \lambda \neq 0\}
$$

is the textbook relaxation-time proxy: asymptotically, deviations from the
steady state decay as $e^{-\Delta t}$, so $1/\Delta$ "is" the relaxation
time. This is exact in the long-time limit for the slowest mode — and it is
routinely wrong as a description of what an experiment or a simulation
actually observes.

## Why the gap misleads

Liouvillians are generically **non-normal**: their eigenvectors are not
orthogonal. Non-normality decouples the spectrum from finite-time behaviour
in several distinct ways, each of which LiouScope measures separately:

Transient amplification
: Even when every mode decays, $\|e^{\mathcal{L}t}\|$ can *grow* before it
  decays. The Kreiss constant (D10) gives a rigorous lower bound on that
  transient; the numerical-abscissa ratio (D15) and the sup-norm transient
  (D14) measure it directly.

Eigenvector condition / Petermann factors
: The excitation of a mode is weighted by how non-orthogonal its left and
  right eigenvectors are (D9). A large Petermann factor means a mode that is
  spectrally fast can still dominate the observed dynamics for a long time
  — the Mori–Shirai overlap amplification (family F1).

Pseudospectra
: For non-normal operators, eigenvalues are unstable under perturbation: the
  $\epsilon$-pseudospectrum can reach far beyond the spectrum (D13, resolvent
  peak D11/D12). Physical response is governed by the resolvent, not by
  eigenvalue positions alone.

Mechanism-specific failures
: Symmetrised-gap corrections (F3), the Liouvillian skin effect (F2),
  anomalous Mpemba ordering of initial states (F4) and phantom relaxation
  (F5) each break the gap–relaxation link in a *qualitatively different*
  way. A single number cannot tell you which one you are looking at — and
  the appropriate correction differs per mechanism.

The discrepancies are not small: gap-based predictions can be off by orders
of magnitude in lattice systems of very modest size.

## The design consequence

Because there are *many* distinct failure modes, LiouScope's API refuses to
collapse them:

1. **A structured report, not a rate.** `diagnose()` returns a
   `DiagnosticReport` whose layers (spectral, non-normality, relaxation,
   transient, classification, uncertainty) can disagree with each other —
   and that disagreement *is* the finding. There is deliberately no
   `decay_rate` attribute, and the project treats adding one as an
   API-design regression ("Don't introduce a single-decay-rate surface").

2. **Explicit uncertainty.** Fitted quantities carry bias-corrected bootstrap
   intervals — BCa when the grid is short enough for the leave-one-out
   jackknife that supplies the acceleration term, plain BC on the default
   80-point grid; `relaxation.interval_method` records which one a run
   actually computed (issue #116) — plus AICc model selection over a model
   hierarchy, and GLS with AR(1) residuals. A rate without an interval and a
   model-selection trace is not reported.

3. **Evidence-graded claims.** The classifier outputs a verdict vocabulary
   (`CONFIRMED` / `CANDIDATE` / `NOT_EXCLUDED` / `UNDEFINED`) instead of a
   boolean, and abstention is a first-class, *correct* outcome when the run
   cannot support a claim (see {doc}`layers-and-taxonomy`).

The one place a user might want a single number back — "how long until
mixed?" — is served by the opt-in D24 Zhou predictor, which answers with a
*bracket* $[t_{\text{lower}}, t_{\text{upper}}]$ whose width is itself a
non-normality diagnostic (see {doc}`../how-to/zhou-mixing-time`).
