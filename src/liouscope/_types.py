"""Frozen result dataclasses for the LiouScope public API.

All result containers are frozen dataclasses with keyword-only construction.
The top-level :class:`DiagnosticReport` is what `diagnose()` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._consts import DIAGNOSTIC_SCHEMA_VERSION, TAXONOMY_VERSION

# issue #70 A5: "EXCLUDED" was removed. A single-pass maximum-evidence
# classifier reports the best-fit A-class with its support and never emits a
# genuine active-rejection verdict, so the value was permanently unreachable
# through ``diagnose()`` (advertised a capability the pipeline lacks). Active
# exclusion requires a per-hypothesis scoring mode (deferred). See CHANGELOG.
Verdict = Literal["CONFIRMED", "CANDIDATE", "NOT_EXCLUDED", "UNDEFINED"]
Tier = Literal["PUBLICATION_GRADE", "CONFIRMATION", "EXPLORATION"]
QualityLabel = Literal["stable", "moderate", "exploratory"]
SolverPath = Literal["dense", "sparse_arpack"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectralResult:
    """Spectral layer S: D1, D2, D2b, D3, D4."""

    gap: float                  # D1 Delta
    gns_gap: float              # D2 Delta_s (Mori-Shirai 2023)
    kms_gap: float              # D2b (Fagnola 2025)
    oscillating_gap: float      # D3 min |Im(lambda)| over complex pairs
    spectral_spread: float      # D4 max|Re| minus min|Re| (non-zero modes)
    eigenvalues: np.ndarray     # full sigma(L) sorted by real part
    steady_state: np.ndarray    # rho_ss matrix, d x d
    has_complex_pairs: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class SteadyProjectorResult:
    """Asymptotic spectral (Riesz) projector of a Liouvillian — issue #103.

    Built from an ORDERED Schur decomposition plus a Sylvester solve, not from
    a single arbitrary null vector: for a degenerate stationary manifold the
    latter picks one basis vector of the kernel and silently misrepresents the
    asymptotic conditional expectation.

    ``semisimple`` is fail-closed. A defective zero/peripheral mode has no
    spectral projector onto a complementary invariant subspace in the sense
    used here, so downstream centred quantities must report NaN rather than a
    plausible-looking number. ``claim_status: pending``.
    """

    projector: np.ndarray       # P_inf on the vectorised space (order='F')
    rank: int                   # dim of the peripheral/asymptotic subspace
    semisimple: bool            # every peripheral mode diagonalisable
    peripheral_eigenvalues: np.ndarray
    tolerance: float            # |Re lambda| <= tolerance counts as peripheral
    separation: float           # gap to the fastest peripheral-excluded mode


@dataclass(frozen=True, slots=True, kw_only=True)
class CenteredTransientEstimate:
    """D14b: ``sup_t ||e^{tL} - P_inf||_2`` with grid audit metadata (#103).

    A finite time grid yields a LOWER BOUND. ``edge_maximizer`` records that
    the sampled maximum sat on the last grid point, i.e. the true peak may lie
    beyond the window. ``claim_status: pending``.
    """

    value: float
    projector_norm: float
    rank: int
    semisimple: bool
    t_min: float
    t_max: float
    n_points: int
    edge_maximizer: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationalTransientEstimate:
    """D14c: trace-norm amplification on the traceless-Hermitian subspace (#103).

    Evaluated over a finite, recorded family of PHYSICAL state differences
    (differences of density matrices), so the value is an explicit LOWER BOUND
    on the induced 1->1 norm, never a certified supremum. For a CPTP semigroup
    it must not exceed 1 within tolerance — that contractivity control is the
    diagnostic's own sanity check. ``claim_status: pending``.
    """

    value: float
    n_states: int
    seed: int
    t_min: float
    t_max: float
    n_points: int
    edge_maximizer: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class KreissGridEstimate:
    """D10b: scale-relative Kreiss grid lower bound with audit metadata.

    Issue #101 slice A / 2026-08-05 re-audit: coarse-grid results must be
    labelled as estimates (lower bounds), not certified constants, and must
    record grid ranges/resolution, whether the maximiser sat on a grid edge,
    and refinement-convergence information. ``claim_status: pending``.
    """

    value: float                # refined grid lower bound (dimensionless)
    coarse_value: float         # pre-refinement grid lower bound
    edge_maximizer: bool        # coarse maximiser on a grid edge -> sup may lie outside
    refinement_delta: float     # (value - coarse_value) / value; convergence metadata
    sigma_rel_lo: float         # dimensionless sigma grid range (units of rate_scale)
    sigma_rel_hi: float
    omega_rel_max: float        # dimensionless omega grid half-span
    n_sigma: int
    n_omega: int
    rate_scale: float           # ||L||_F used to (de)dimensionalise the grid


@dataclass(frozen=True, slots=True, kw_only=True)
class NonNormalityResult:
    """Non-normality layer N: D8 (+D8b), D9, D10 (+D10b), D11.

    The scale-relative D8b/D10b fields (issue #101 slice A) are additive and
    defaulted to NaN/False so older callers and serialised results stay valid;
    they are ``claim_status: pending`` and advisory-only (no classifier branch
    consumes them).
    """

    henrici_eta: float          # D8 (rate-dimensioned, legacy)
    petermann_max: float        # D9 max K_j
    petermann_factors: np.ndarray
    kreiss: float               # D10 (legacy absolute-grid lower bound)
    bohr_ap_length: int         # D11 Bohr arithmetic-progression depth
    bohr_ap_pauli_bound: float
    henrici_relative: float = float("nan")   # D8b eta_N / ||L||_F in [0, 1]
    kreiss_scaled: float = float("nan")      # D10b scale-relative grid lower bound
    kreiss_scaled_edge_maximizer: bool = False
    kreiss_scaled_refinement_delta: float = float("nan")


@dataclass(frozen=True, slots=True, kw_only=True)
class FitResult:
    """Single-model fit summary for an entry of the M-hierarchy."""

    model: str                  # "M0", "M1", "M2", "M3a", "M3b"
    params: np.ndarray
    log_likelihood: float
    aicc: float
    n_eff: float
    residual_ar1_rho: float
    success: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RelaxationResult:
    """Relaxation layer R: D5-D7b plus M0..M3b winner."""

    von_neumann_entropy: float            # D5 final
    relative_entropy_curve: np.ndarray    # D6 trajectory
    fidelity_curve: np.ndarray            # D7 trajectory
    entanglement_asymmetry: float | None  # D7b
    fits: dict[str, FitResult]
    aicc_model: str                       # winning model
    beta_D: float                         # fitted exponential rate of best model
    bca_ci_beta: tuple[float, float]      # BCa 95% CI
    # F-018 (LIOU-F-018): half trace-norm distance to rho_ss along the
    # trajectory, the observable relaxation metric alongside D5/D6/D7. Optional
    # and defaulted so the field is purely additive (older callers/serialised
    # reports remain valid).
    trace_distance_curve: np.ndarray | None = None
    # LIOU-#69: dominant rate of the LINEAR trace-distance curve. Unlike beta_D
    # (fit on relative entropy, which carries a metric multiplier m in {1, 2}),
    # this rate decays at the bare mode rate and is dimension-coherent with the
    # spectral gap Delta -- it is the rate the D17 gap-rate consistency check
    # uses. Additive + defaulted so older callers / serialised reports stay valid.
    beta_D_linear: float = float("nan")
    linear_fit_model: str = "none"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolventResult:
    """Resolvent layer: D11b, D12, D13 (+ scale-relative variants).

    The scale-relative fields (issue #101 slice A) are additive and defaulted
    to NaN so older callers/serialised results stay valid; they are
    ``claim_status: pending`` and advisory-only. ``rate_scale == 0`` (zero
    operator) leaves all of them NaN by documented semantics.
    """

    resolvent_peak: float                 # D11b (legacy absolute sigma)
    ridge_fwhm: float                     # D12 (legacy)
    pseudospectral_radius: float          # D13 eps-pseudospectrum radius (legacy)
    pseudospec_eps: float
    rate_scale: float = float("nan")               # ||L||_F shared scale
    sigma_rel: float = float("nan")                # dimensionless D11b offset
    resolvent_peak_scaled: float = float("nan")    # rate_scale * peak (dimensionless)
    ridge_fwhm_rel: float = float("nan")           # fwhm / rate_scale
    pseudospec_eps_rel: float = float("nan")       # eps_abs = eps_rel * rate_scale
    pseudospectral_radius_rel: float = float("nan")  # radius / rate_scale
    pseudospectral_abscissa: float = float("nan")    # max Re z in sigma_eps (rate-valued)
    pseudospectral_abscissa_rel: float = float("nan")  # abscissa / rate_scale


@dataclass(frozen=True, slots=True, kw_only=True)
class TransientResult:
    """Transient layer: D14, D15."""

    trans_amplitude_ratio: float          # D14 sup_t ||e^{tL}||_2 (UNSTRUCTURED HS)
    kappa_trans: float                    # D15 omega(L) / Delta
    numerical_abscissa: float             # omega(L)
    # Issue #103: D14 above is an unstructured Hilbert-Schmidt semigroup norm
    # over the full complex Liouville space. It carries the asymptotic
    # projector baseline (||P_inf||_2 = sqrt(d * Tr rho_ss^2) in [1, sqrt(d)])
    # and its extremiser need not be Hermitian, traceless or a difference of
    # physical states. The fields below separate those two confounds. Additive
    # and defaulted so synthetic callers stay valid; ``claim_status: pending``,
    # no classifier consumption until calibration (#102).
    trans_amplitude_centered: float = float("nan")      # D14b sup_t ||e^{tL} - P_inf||_2
    trans_amplitude_decaying: float = float("nan")      # D14d sup_t ||e^{tL}|_decay||_2
    steady_projector_norm: float = float("nan")         # ||P_inf||_2 (the removed baseline)
    steady_projector_rank: int = -1                     # dim of the asymptotic subspace
    steady_projector_semisimple: bool = False           # False => D14b is NaN (fail-closed)
    trans_amplitude_operational: float = float("nan")   # D14c trace-norm LOWER BOUND


@dataclass(frozen=True, slots=True, kw_only=True)
class LepResult:
    """LEP layer: D16, D17, D18.

    Note: conjugate-pair LEP candidates are INCLUDED (FIX-3, anchor I).
    """

    lep_proximity: float                  # D16 min pair separation
    gap_rate_consistency: float           # D17 |beta_D_linear - Delta| / Delta
    initial_state_sensitivity: float      # D18 std over Haar ensemble
    lep_candidate_count: int
    # LIOU-#69: the LINEAR-metric rate actually fed to D17 (dimension-coherent
    # with the gap). Additive + defaulted so synthetic callers stay valid.
    beta_D_linear: float = float("nan")


@dataclass(frozen=True, slots=True, kw_only=True)
class MpembaResult:
    """Mpemba layer: D19, D20."""

    overlap_c1: float                     # D19 slowest-mode overlap
    is_mpemba_candidate: bool
    expansion_alpha: float                # D20 Phi_n scaling exponent
    trivial_overlap: bool = False         # symmetry-protected zero overlap (not Mpemba)


@dataclass(frozen=True, slots=True, kw_only=True)
class UncertaintyResult:
    """Uncertainty layer U: U0, U1, U2."""

    fit_uncertainty: float                # U0
    solver_uncertainty: float             # U1
    size_uncertainty: float | None        # U2 (None for single-N)
    bootstrap_B: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassificationResult:
    """Classification layer C: A-class plus F-family plus verdict.

    Anchor L: ``taxonomy_version`` and ``schema_version`` are stamped here.
    """

    a_class: str                          # "A1" .. "A12"
    f_family: str                         # "F1" .. "F5" or "none"
    verdict: Verdict
    tier: Tier
    confidence: float                     # 0..1
    evidence: dict[str, float]
    taxonomy_version: str = TAXONOMY_VERSION
    schema_version: str = DIAGNOSTIC_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class GovernanceMetadata:
    """Governance layer G: SHA-256 run-id and reproducibility info."""

    run_id: str
    framework_version: str
    python_version: str
    platform: str
    numpy_version: str
    scipy_version: str
    seed: int
    solver_path: SolverPath
    quality_label: QualityLabel
    timestamp: str
    input_hash: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticReport:
    """Top-level report produced by :func:`liouscope.diagnose`."""

    spectral: SpectralResult
    nonnorm: NonNormalityResult
    relaxation: RelaxationResult
    resolvent: ResolventResult
    transient: TransientResult
    lep: LepResult
    uncertainty: UncertaintyResult
    classification: ClassificationResult
    governance: GovernanceMetadata
    mpemba: MpembaResult | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZhouPredictorResult:
    """v0.2.1 post-submission Zhou universal mixing-time predictor (D24)."""

    mixing_time_lower: float
    mixing_time_upper: float
    epsilon: float
    converged: bool
    gap: float = float("nan")
    petermann_factor: float = float("nan")
