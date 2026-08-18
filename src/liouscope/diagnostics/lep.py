"""LEP layer: D16 Liouvillian-exceptional-point proximity, D17 gap-rate
consistency, D18 initial-state sensitivity.

Anchor I: Conjugate-pair eigenvalues are INCLUDED in the LEP scan. The
original v0.1 implementation excluded them (FIX-3 / external audit), which
hid genuine non-diagonalisability cases (Tay 2023, Claeys 2022).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._types import LepResult
from ..core.lindblad import steady_state
from ..io.seed import RNGLike, SeedLike, derive_seed
from ..numerics.kronecker import unvec, vec
from ..numerics.scale import spectral_zero_tolerance


def lep_proximity(
    eigenvalues: np.ndarray,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[float, int]:
    """Minimum eigenvalue-pair separation, including complex-conjugate pairs.

    Returns ``(min_sep, candidate_count)`` where ``candidate_count`` is the
    number of pairs within a proximity window of the closest pair.

    Issue #70 A9: a Liouvillian exceptional point (LEP) is where two eigenvalues
    (and their eigenvectors) coalesce, so eigenvalue separation -> 0 is the
    signature of *approaching* an EP and an EXACT degeneracy is the STRONGEST
    possible proximity signal. The pre-#70 scan skipped every pair with
    ``sep <= atol`` (``if sep > atol``), which discarded exactly that signal:
    an exactly (or numerically) degenerate pair was invisible, and a fully
    degenerate spectrum returned ``inf`` -- "maximally FAR from an EP" -- the
    exact inverse of the physics. The scan below keeps all ``i < j`` pairs; a
    minimum separation at or below ``atol`` is clamped to ``0.0`` (coalesced =
    proximity 0, not inf). The candidate-count loop uses the SAME data and a
    proximity window ``max(10 * min_sep, atol)`` so that when ``min_sep == 0``
    the window is ``atol`` (counting the coalesced cluster) rather than a
    degenerate zero-width window -- the two loops are now mutually consistent.

    Issue #82 follow-up: non-finite eigenvalues are not a physical LEP signal and
    must not be silently converted into ``inf`` / zero candidates. NaN/inf input
    now fails closed with ``ValueError`` before the pairwise scan, so upstream
    eigensolver corruption cannot masquerade as "maximally far from an EP".

    D16 measures eigenvalue proximity only; it cannot by itself distinguish a
    genuine (defective) EP from a semisimple / symmetry-protected degeneracy.
    That disambiguation is the job of the non-normality layer (Petermann factor
    D9), which the classifier combines with this signal.
    """
    eigenvalues = np.asarray(eigenvalues)
    if not np.all(np.isfinite(eigenvalues)):
        bad = np.flatnonzero(~np.isfinite(eigenvalues)).tolist()
        raise ValueError(
            "lep_proximity requires finite eigenvalues; "
            f"non-finite entries at indices {bad}"
        )
    # Issue #108: the coalescence threshold is a rate-dimensioned SEPARATION,
    # so an absolute floor made "are these two eigenvalues coalesced?" depend on
    # the choice of rate unit -- under L -> cL with small c every separation
    # fell below the floor and reported proximity 0.0, the STRONGEST possible EP
    # signal, for an ordinary well-separated spectrum. Both the clamp and the
    # candidate-count window now scale with the spectral radius.
    tol = (
        spectral_zero_tolerance(eigenvalues, atol=atol, name="eigenvalues")
        if rtol is None
        else spectral_zero_tolerance(eigenvalues, rtol=rtol, atol=atol, name="eigenvalues")
    )
    n = eigenvalues.size
    if n < 2:
        return float("inf"), 0
    min_sep = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            sep = float(abs(eigenvalues[i] - eigenvalues[j]))
            if sep < min_sep:
                min_sep = sep
    # Exact / numerically-degenerate closest pair == strongest EP signal.
    if min_sep <= tol:
        min_sep = 0.0
    window = max(10.0 * min_sep, tol)
    pairs_close = 0
    for i in range(n):
        for j in range(i + 1, n):
            sep = float(abs(eigenvalues[i] - eigenvalues[j]))
            if sep <= window:
                pairs_close += 1
    return min_sep, pairs_close


def gap_rate_consistency(rate: float, gap: float) -> float:
    """D17: ``|rate - Delta| / Delta``. Returns inf if gap is zero or rate is
    non-finite.

    ``rate`` must be a LINEAR-metric relaxation rate (LIOU-#69): the spectral
    gap ``Delta = -max Re(lambda != 0)`` is a bare single-mode decay rate, so
    the consistency check is only dimension-coherent when it is compared against
    a rate measured on a *linear* distance metric (trace distance / fidelity),
    which decays at that same bare rate. The relative-entropy rate ``beta_D``
    carries a metric multiplier m in {1, 2} (m=2 for a faithful pi, m=1 for a
    rank-deficient pi) and must NOT be passed here -- doing so inflates D17 by m
    and makes the A1 "gap-controlled" label unreachable (issue #69).
    """
    if gap <= 0:
        return float("inf")
    if not np.isfinite(rate):
        return float("inf")
    return float(abs(rate - gap) / gap)


def initial_state_sensitivity(
    L_super: np.ndarray,
    rho_steady_state: np.ndarray,
    *,
    n_samples: int = 10,
    t_eval: float = 1.0,
    seed: int | None = None,
    rng: RNGLike | SeedLike | None = None,
) -> float:
    """D18: std of relaxation distance over a Haar-random initial-state ensemble.

    Samples ``n_samples`` Haar-random pure states, evolves to ``t_eval`` and
    measures
    ``||rho(t) - rho_ss||_F``. Returns the standard deviation across samples.
    ``seed`` (legacy, default 7) and the SPEC 7 ``rng`` keyword are mutually
    exclusive; ``rng`` is normalised via :func:`liouscope.io.seed.derive_seed`.
    """
    L_super = np.asarray(L_super)
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    gen = np.random.default_rng(derive_seed(rng, seed, default=7))
    expm_t = sla.expm(L_super * t_eval)
    distances = np.empty(n_samples)
    for k in range(n_samples):
        psi = gen.normal(size=d) + 1j * gen.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())
        rho_t = unvec(expm_t @ vec(rho0), d=d)
        distances[k] = float(np.linalg.norm(rho_t - rho_steady_state, ord="fro"))
    return float(np.std(distances))


def compute_lep_layer(
    L_super: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    beta_D_linear: float,
    gap: float,
    rho_steady_state: np.ndarray | None = None,
    seed: int | None = None,
    rng: RNGLike | SeedLike | None = None,
    n_haar: int = 10,
) -> LepResult:
    """Run D16, D17, D18 together.

    ``beta_D_linear`` is the LINEAR-metric relaxation rate (from the trace-
    distance curve), which is dimension-coherent with the spectral ``gap``; see
    :func:`gap_rate_consistency` for why the relative-entropy rate must not be
    used here (issue #69). ``seed`` (legacy, default 7) and the SPEC 7 ``rng``
    keyword are mutually exclusive and only feed the D18 Haar ensemble.
    """
    L_super = np.asarray(L_super)
    if rho_steady_state is None:
        rho_steady_state = steady_state(L_super)
    proximity, candidates = lep_proximity(eigenvalues)
    consistency = gap_rate_consistency(beta_D_linear, gap)
    sensitivity = initial_state_sensitivity(
        L_super, rho_steady_state, n_samples=n_haar, seed=seed, rng=rng
    )
    return LepResult(
        lep_proximity=proximity if np.isfinite(proximity) else float("inf"),
        gap_rate_consistency=consistency,
        initial_state_sensitivity=sensitivity,
        lep_candidate_count=candidates,
        beta_D_linear=float(beta_D_linear),
    )
