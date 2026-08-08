"""Transient layer: D14 unstructured HS semigroup norm, D14b centred, D14c
operational, D15 kappa_trans.

Issue #103 splits the legacy D14 into three questions that were previously
conflated in one number:

* **D14** ``sup_t ||e^{tL}||_2`` — mathematical, unstructured, carries the
  asymptotic projector baseline. Kept for backward comparison.
* **D14b** ``sup_t ||e^{tL} - P_inf||_2`` — the same norm with the asymptotic
  projector removed, so growth means growth of the DECAYING part.
* **D14d** ``sup_t ||e^{tL}|_decay||_2`` — the semigroup restricted to the
  decaying invariant subspace. NOT the same as D14b: ``||I - P|| = ||P||`` for
  an oblique projector, so only D14d actually starts at 1.
* **D14c** trace-norm amplification restricted to traceless-Hermitian
  differences of physical density matrices — the operational question, whose
  answer must be <= 1 for CPTP dynamics.

D14b, D14c and D14d are advisory (``claim_status: pending``) and are not consumed by
the classifier; the F2 branch keeps the legacy D14 until the calibration study
in issue #102.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .._types import (
    CenteredTransientEstimate,
    OperationalTransientEstimate,
    SteadyProjectorResult,
    TransientResult,
)


class TransientGridWarning(UserWarning):
    """Emitted when ``trans_amplitude_ratio`` may have under-resolved the sup.

    Raised when the propagator norm is still rising at the right edge of the
    (auto-scaled) time grid, which means the true transient peak likely lies
    beyond the sampled window and the returned value is a lower bound.
    """


def numerical_abscissa(L_super: np.ndarray) -> float:
    """Numerical abscissa ``omega(L) = max eigenvalue of (L + L^H)/2``."""
    A = 0.5 * (np.asarray(L_super) + np.asarray(L_super).conj().T)
    return float(np.max(np.linalg.eigvalsh(A)))


def _physics_time_grid(
    L_super: np.ndarray,
    gap: float,
    *,
    n_points: int = 64,
    horizon: float = 8.0,
) -> np.ndarray:
    """Build a physics-scaled two-scale time grid for ``sup_t ||e^{tL}||``.

    The relevant timescales of a stable Liouvillian propagator are set by

    * the spectral gap ``Delta`` (D1): the slowest mode decays as
      ``e^{-Delta t}``, so the propagator has returned close to its asymptotic
      value only after ``t ~ several / Delta``;
    * the numerical abscissa ``omega(L)``: the *initial* growth rate, which
      sets where a sharp transient overshoot peaks (early, ``t ~ 1/omega``).

    A fixed grid such as ``linspace(0.01, 5.0, 30)`` silently misses the
    supremum whenever ``1/Delta`` is far outside ``[0.01, 5]`` (small gap ->
    peak/plateau beyond ``t = 5``; this is exactly the strongly non-normal
    regime the diagnostic targets). We instead span ``[0, ~horizon/Delta]``
    and place a log-spaced early segment (to resolve a sharp growth peak that a
    uniform grid would step over) plus a linear late segment.

    ``gap <= 0`` (no isolated steady state / degenerate spectrum) falls back to
    a window driven purely by ``omega(L)`` so the call still terminates.
    """
    L_super = np.asarray(L_super)
    omega = max(numerical_abscissa(L_super), 0.0)
    if gap > 0.0:
        gap_eff = gap
    else:
        # No usable decay scale: drive the window off the growth scale only.
        gap_eff = omega if omega > 0.0 else 1.0
    t_decay = horizon / gap_eff
    fast = max(omega, gap_eff)
    t_early = min(horizon / fast, t_decay)
    n_early = max(2, n_points // 2)
    n_late = max(2, n_points - n_early)
    # The early segment must ascend from a tiny time up to ``t_early`` to resolve
    # a sharp growth peak. For extreme non-normality (numerical abscissa exceeding
    # the gap by >6 decades) the nominal start ``t_decay * 1e-6`` overtakes
    # ``t_early``, which would make ``geomspace`` run *backwards* and cluster the
    # fine sampling after the peak instead of before it. Clamp the start below
    # ``t_early`` in that regime; the guard is a no-op for all normal spectra.
    early_start = t_decay * 1.0e-6
    if early_start >= t_early:
        early_start = t_early * 1.0e-3
    early = np.geomspace(early_start, t_early, n_early)
    late = np.linspace(t_early, t_decay, n_late)
    # np.asarray keeps mypy happy on py3.10 numpy stubs (np.unique -> Any there).
    return np.asarray(np.unique(np.concatenate(([0.0], early, late))), dtype=np.float64)


def trans_amplitude_ratio(
    L_super: np.ndarray,
    *,
    gap: float | None = None,
    t_grid: np.ndarray | None = None,
) -> float:
    """D14: ``sup_t ||e^{tL}||_2`` — UNSTRUCTURED Hilbert-Schmidt semigroup
    norm estimate over the full complex Liouville space.

    Issue #103: despite the historical name this is **not** a state-amplitude
    ratio without qualification. Two confounds ride along:

    * **Projector baseline.** ``e^{tL} -> P_inf`` with
      ``||P_inf||_2 = sqrt(d * Tr(rho_ss^2))``, i.e. 1 for the maximally mixed
      steady state up to ``sqrt(d)`` for a pure one. The value can exceed 1 by
      geometry alone, with no transient overshoot. See
      :func:`centered_transient_amplitude` (D14b).
    * **Norm geometry.** The extremiser ranges over all of complex Liouville
      space and need not be Hermitian, traceless, or a difference of physical
      density matrices. A CPTP map contracts trace distance between physical
      states, so growth here does not imply increased operational
      distinguishability. See :func:`operational_trace_amplitude` (D14c).

    Preserved unchanged for backward comparison; the two diagnostics above are
    additive and answer the separated questions.

    The time grid is chosen in this priority order:

    1. an explicit ``t_grid`` (caller fully controls sampling), else
    2. a physics-scaled grid built from the spectral ``gap`` (D1) and the
       numerical abscissa (see :func:`_physics_time_grid`), else
    3. a legacy coarse fallback ``linspace(0.01, 5.0, 30)`` when neither
       ``gap`` nor ``t_grid`` is supplied (kept for backward compatibility).

    When the sampled norm is still increasing at the right edge of an
    auto-scaled grid the returned value is only a lower bound; a
    :class:`TransientGridWarning` is emitted in that case.
    """
    L_super = np.asarray(L_super)

    auto_scaled = False
    if t_grid is None:
        if gap is not None:
            t_grid = _physics_time_grid(L_super, gap)
            auto_scaled = True
        else:
            t_grid = np.linspace(0.01, 5.0, 30)
    t_grid = np.asarray(t_grid, dtype=float)

    norms = np.array(
        [float(sla.svdvals(sla.expm(L_super * t))[0]) for t in t_grid]
    )
    sup = float(norms.max())

    if auto_scaled and t_grid.size >= 2:
        # Underestimation guard. Warn only when the maximum sits on the final
        # sample AND the norm is still rising there by a *non-negligible*
        # relative amount -- i.e. the true peak plausibly lies beyond the
        # window. A merely asymptotic monotone approach to the supremum (the
        # generic behaviour of a contractive propagator settling onto its
        # transient ratio) rises by << rel_tol per step and must NOT warn,
        # otherwise the guard fires on every well-resolved system.
        rel_tol = 1.0e-3
        edge_rise = norms[-1] - norms[-2]
        rising_significantly = edge_rise > rel_tol * max(norms[-1], 1.0)
        if int(np.argmax(norms)) == norms.size - 1 and rising_significantly:
            warnings.warn(
                "trans_amplitude_ratio: propagator norm still rising at the "
                f"right grid edge t={t_grid[-1]:.4g} (norm {norms[-1]:.6g} > "
                f"{norms[-2]:.6g}); the returned sup is a lower bound. "
                "Pass an explicit, wider t_grid to resolve the peak.",
                TransientGridWarning,
                stacklevel=2,
            )
    return sup


def steady_projector(
    L_super: np.ndarray,
    *,
    tol: float | None = None,
) -> SteadyProjectorResult:
    """Asymptotic spectral (Riesz) projector ``P_inf`` of ``L_super`` — issue #103.

    For a trace-preserving semigroup with a unique steady state
    ``e^{tL} -> P_inf = |rho_ss><I|`` in column-stacking convention, and
    ``||P_inf||_2 = sqrt(d * Tr(rho_ss^2)) in [1, sqrt(d)]``. Any full-space
    ``sup_t ||e^{tL}||`` therefore inherits that baseline even with no
    transient overshoot at all — which is the confound this projector removes.

    Construction: eigenvalues with ``|Re lambda| <= tol`` are moved to the
    leading block of an ORDERED Schur form, and the oblique projector onto that
    invariant subspace along the decaying complement is

        P = Z @ [[I, X], [0, 0]] @ Z^H,    T11 X - X T22 = T12

    (a Sylvester solve). This is stable and handles a degenerate stationary
    manifold correctly; taking one null vector of ``L`` would not.

    Fail-closed: a defective peripheral mode (algebraic > geometric
    multiplicity) sets ``semisimple=False``. Callers must then refuse to report
    a centred amplitude rather than return a plausible-looking number.
    """
    L = np.asarray(L_super)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"steady_projector: expected a square operator, got {L.shape}")
    if not np.all(np.isfinite(L)):
        raise ValueError("steady_projector: non-finite entries in L_super")

    n = L.shape[0]
    scale = float(np.linalg.norm(L, "fro"))
    if tol is None:
        # Relative to the operator scale so the split is rate-unit invariant
        # (issue #101): an absolute threshold would reclassify modes under
        # L -> cL. n * eps**0.5 is the usual conservative eigenvalue floor.
        tol = scale * float(n) * float(np.sqrt(np.finfo(float).eps))
    if scale == 0.0:
        # Zero generator: e^{tL} = I for all t, the whole space is stationary.
        return SteadyProjectorResult(
            projector=np.eye(n, dtype=complex), rank=n, semisimple=True,
            peripheral_eigenvalues=np.zeros(n, dtype=complex),
            tolerance=float(tol), separation=float("inf"),
        )

    T, Z, sdim = sla.schur(L, output="complex", sort=lambda x: bool(abs(x.real) <= tol))
    k = int(sdim)
    eigs = np.diag(T).copy()
    peripheral = eigs[:k]
    rest = eigs[k:]
    separation = float(np.min(np.abs(rest.real))) if rest.size else float("inf")

    if k == 0:
        return SteadyProjectorResult(
            projector=np.zeros((n, n), dtype=complex), rank=0, semisimple=True,
            peripheral_eigenvalues=peripheral, tolerance=float(tol), separation=separation,
        )
    if k == n:
        return SteadyProjectorResult(
            projector=np.eye(n, dtype=complex), rank=n,
            semisimple=_peripheral_is_semisimple(L, peripheral, tol),
            peripheral_eigenvalues=peripheral, tolerance=float(tol), separation=separation,
        )

    T11, T12, T22 = T[:k, :k], T[:k, k:], T[k:, k:]
    # T11 X - X T22 = T12  <=>  solve_sylvester(T11, -T22, T12)
    X = sla.solve_sylvester(T11, -T22, T12)
    block = np.zeros((n, n), dtype=complex)
    block[:k, :k] = np.eye(k)
    block[:k, k:] = X
    P = Z @ block @ Z.conj().T

    return SteadyProjectorResult(
        projector=P, rank=k,
        semisimple=_peripheral_is_semisimple(L, peripheral, tol),
        peripheral_eigenvalues=peripheral, tolerance=float(tol), separation=separation,
    )


def _peripheral_is_semisimple(
    L: np.ndarray, peripheral: np.ndarray, tol: float
) -> bool:
    """Algebraic == geometric multiplicity for every distinct peripheral mode.

    Geometric multiplicity is read off the singular values of ``L - lambda I``
    (rank deficiency), which is the numerically stable route; comparing Schur
    off-diagonals would confuse a defective mode with a merely ill-conditioned
    basis.
    """
    n = L.shape[0]
    remaining = list(peripheral)
    while remaining:
        lam = remaining[0]
        cluster = [z for z in remaining if abs(z - lam) <= max(tol, 1e-12)]
        remaining = [z for z in remaining if abs(z - lam) > max(tol, 1e-12)]
        algebraic = len(cluster)
        centre = complex(np.mean(cluster))
        svals = sla.svdvals(L - centre * np.eye(n))
        cutoff = max(tol, float(svals[0]) * n * float(np.finfo(float).eps))
        geometric = int(np.sum(svals <= cutoff))
        if geometric < algebraic:
            return False
    return True


def centered_transient_amplitude(
    L_super: np.ndarray,
    *,
    gap: float | None = None,
    t_grid: np.ndarray | None = None,
    projector: SteadyProjectorResult | None = None,
) -> CenteredTransientEstimate:
    """D14b: ``sup_t ||e^{tL} - P_inf||_2`` — issue #103.

    Removes the asymptotic projector baseline from D14, so a value above one
    means genuine transient growth of the DECAYING part rather than a pure
    steady state inflating the norm.

    Fail-closed on a defective peripheral mode: ``value`` is NaN and
    ``semisimple`` is False.
    """
    L = np.asarray(L_super)
    proj = projector if projector is not None else steady_projector(L)
    P = proj.projector
    p_norm = float(sla.svdvals(P)[0]) if proj.rank > 0 else 0.0

    if t_grid is None:
        t_grid = (
            _physics_time_grid(L, gap) if gap is not None else np.linspace(0.01, 5.0, 30)
        )
    t_grid = np.asarray(t_grid, dtype=float)

    if not proj.semisimple:
        return CenteredTransientEstimate(
            value=float("nan"), projector_norm=p_norm, rank=proj.rank,
            semisimple=False, t_min=float(t_grid[0]), t_max=float(t_grid[-1]),
            n_points=int(t_grid.size), edge_maximizer=False,
        )

    norms = np.array(
        [float(sla.svdvals(sla.expm(L * t) - P)[0]) for t in t_grid]
    )
    return CenteredTransientEstimate(
        value=float(norms.max()), projector_norm=p_norm, rank=proj.rank,
        semisimple=True, t_min=float(t_grid[0]), t_max=float(t_grid[-1]),
        n_points=int(t_grid.size),
        edge_maximizer=bool(int(np.argmax(norms)) == norms.size - 1),
    )


def decaying_transient_amplitude(
    L_super: np.ndarray,
    *,
    gap: float | None = None,
    t_grid: np.ndarray | None = None,
    tol: float | None = None,
) -> CenteredTransientEstimate:
    """D14d: ``sup_t ||e^{tL}|_decay||_2`` on the decaying invariant subspace.

    Issue #103 offers ``sup_t ||e^{tL} - P_inf||`` "or equivalently restrict
    the semigroup to the decaying complement". **These are not equivalent.**
    For a non-trivial oblique projector ``||I - P|| = ||P||`` (Kato; the two
    complementary projectors of an oblique splitting share their norm), and
    ``e^{0L} - P_inf = I - P_inf``. The centred form therefore still evaluates
    to the projector norm at ``t = 0`` and keeps exactly the baseline the issue
    set out to remove.

    Restricting the generator to the decaying invariant subspace and measuring
    in an ORTHONORMAL basis of that subspace does remove it: the value is
    exactly 1 at ``t = 0`` and exceeds 1 only under genuine transient growth of
    the decaying part. That is this function; both are reported so the
    difference stays visible instead of being argued away.

    Implementation: ordered Schur with the decaying spectrum leading, so the
    leading columns of ``Z`` span the decaying invariant subspace and the
    restriction is the leading triangular block. ``claim_status: pending``.
    """
    L = np.asarray(L_super)
    if not np.all(np.isfinite(L)):
        raise ValueError("decaying_transient_amplitude: non-finite entries in L_super")
    n = L.shape[0]
    scale = float(np.linalg.norm(L, "fro"))
    if tol is None:
        tol = scale * float(n) * float(np.sqrt(np.finfo(float).eps))

    proj = steady_projector(L, tol=tol)
    p_norm = float(sla.svdvals(proj.projector)[0]) if proj.rank > 0 else 0.0

    if t_grid is None:
        t_grid = (
            _physics_time_grid(L, gap) if gap is not None else np.linspace(0.01, 5.0, 30)
        )
    t_grid = np.asarray(t_grid, dtype=float)

    if not proj.semisimple or proj.rank == n:
        # Defective peripheral mode, or nothing decays at all.
        return CenteredTransientEstimate(
            value=float("nan"), projector_norm=p_norm, rank=proj.rank,
            semisimple=proj.semisimple, t_min=float(t_grid[0]),
            t_max=float(t_grid[-1]), n_points=int(t_grid.size), edge_maximizer=False,
        )

    # Decaying spectrum first -> leading block IS the restriction.
    T, _Z, sdim = sla.schur(L, output="complex", sort=lambda x: bool(abs(x.real) > tol))
    k = int(sdim)
    T_decay = T[:k, :k]
    norms = np.array([float(sla.svdvals(sla.expm(T_decay * t))[0]) for t in t_grid])
    return CenteredTransientEstimate(
        value=float(norms.max()), projector_norm=p_norm, rank=proj.rank,
        semisimple=True, t_min=float(t_grid[0]), t_max=float(t_grid[-1]),
        n_points=int(t_grid.size),
        edge_maximizer=bool(int(np.argmax(norms)) == norms.size - 1),
    )


def _physical_difference_states(d: int, *, seed: int, n_random: int) -> list[np.ndarray]:
    """Traceless-Hermitian differences of density matrices, trace-norm 1.

    Deterministic basis pairs (population and coherence directions) plus Haar
    random pure-state pairs. These are genuine ``rho_a - rho_b``, i.e. exactly
    the objects whose trace distance a CPTP map must contract.
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []

    def add(X: np.ndarray) -> None:
        nrm = float(np.sum(sla.svdvals(X)))
        if nrm > 0:
            out.append(X / nrm)

    for i in range(d):
        for j in range(i + 1, d):
            pop = np.zeros((d, d), dtype=complex)
            pop[i, i], pop[j, j] = 1.0, -1.0
            add(pop)
            plus = np.zeros(d, dtype=complex)
            plus[i], plus[j] = 1 / np.sqrt(2), 1 / np.sqrt(2)
            minus = np.zeros(d, dtype=complex)
            minus[i], minus[j] = 1 / np.sqrt(2), -1 / np.sqrt(2)
            add(np.outer(plus, plus.conj()) - np.outer(minus, minus.conj()))

    for _ in range(n_random):
        a = rng.normal(size=d) + 1j * rng.normal(size=d)
        b = rng.normal(size=d) + 1j * rng.normal(size=d)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        add(np.outer(a, a.conj()) - np.outer(b, b.conj()))
    return out


def operational_trace_amplitude(
    L_super: np.ndarray,
    *,
    gap: float | None = None,
    t_grid: np.ndarray | None = None,
    seed: int = 0,
    n_random: int = 16,
) -> OperationalTransientEstimate:
    """D14c: trace-norm amplification on physical state differences — issue #103.

    ``sup_{t, X} ||e^{tL}[X]||_1 / ||X||_1`` over a finite, recorded family of
    traceless-Hermitian ``X = rho_a - rho_b``. The induced 1->1 norm is not
    computable in closed form here, so this is an explicit LOWER BOUND.

    Unlike D14 this answers an operational question — does the dynamics ever
    make two physical states MORE distinguishable? For a CPTP semigroup the
    answer must be no (value <= 1 within tolerance), which makes the quantity
    its own contractivity control.
    """
    L = np.asarray(L_super)
    n = L.shape[0]
    d = int(round(np.sqrt(n)))
    if d * d != n:
        raise ValueError(
            f"operational_trace_amplitude: L_super dimension {n} is not a perfect square"
        )

    if t_grid is None:
        t_grid = (
            _physics_time_grid(L, gap) if gap is not None else np.linspace(0.01, 5.0, 30)
        )
    t_grid = np.asarray(t_grid, dtype=float)

    states = _physical_difference_states(d, seed=seed, n_random=n_random)
    if not states:
        return OperationalTransientEstimate(
            value=float("nan"), n_states=0, seed=seed, t_min=float(t_grid[0]),
            t_max=float(t_grid[-1]), n_points=int(t_grid.size), edge_maximizer=False,
        )

    # Column-stacking (order='F') — Anchor A convention of build_liouvillian.
    V = np.column_stack([X.reshape(-1, order="F") for X in states])
    per_time = np.empty(t_grid.size)
    for idx, t in enumerate(t_grid):
        Y = sla.expm(L * t) @ V
        per_time[idx] = max(
            float(np.sum(sla.svdvals(Y[:, c].reshape(d, d, order="F"))))
            for c in range(Y.shape[1])
        )
    return OperationalTransientEstimate(
        value=float(per_time.max()), n_states=len(states), seed=seed,
        t_min=float(t_grid[0]), t_max=float(t_grid[-1]), n_points=int(t_grid.size),
        edge_maximizer=bool(int(np.argmax(per_time)) == per_time.size - 1),
    )


def kappa_trans(omega_L: float, gap: float) -> float:
    """D15: ``kappa_trans = omega(L) / Delta`` (Patch E5).

    Returns the unbounded ratio. ``omega(L) > 0`` is the generic case in
    non-normal Liouvillians.
    """
    if gap <= 0:
        return float("inf")
    return float(omega_L / gap)


def compute_transient_layer(
    L_super: np.ndarray,
    gap: float,
    *,
    t_grid: np.ndarray | None = None,
) -> TransientResult:
    """Run D14, D15 with the spectral gap from D1.

    The ``gap`` is forwarded to :func:`trans_amplitude_ratio` so the D14 time
    window is physics-scaled to the system's own relaxation timescale instead
    of a fixed coarse grid. An explicit ``t_grid`` still overrides the scaling.
    """
    omega_L = numerical_abscissa(L_super)
    ratio = trans_amplitude_ratio(L_super, gap=gap, t_grid=t_grid)
    kappa = kappa_trans(omega_L, gap)

    # Issue #103, additive: separate the projector baseline and the norm
    # geometry from the legacy unstructured value. Both are advisory
    # (claim_status: pending) and are NOT consumed by the classifier — F2 keeps
    # using the legacy D14 until the calibration study in #102.
    centered = centered_transient_amplitude(L_super, gap=gap, t_grid=t_grid)
    decaying = decaying_transient_amplitude(L_super, gap=gap, t_grid=t_grid)
    operational = operational_trace_amplitude(L_super, gap=gap, t_grid=t_grid)

    return TransientResult(
        trans_amplitude_ratio=ratio,
        kappa_trans=kappa,
        numerical_abscissa=omega_L,
        trans_amplitude_centered=centered.value,
        trans_amplitude_decaying=decaying.value,
        steady_projector_norm=centered.projector_norm,
        steady_projector_rank=centered.rank,
        steady_projector_semisimple=centered.semisimple,
        trans_amplitude_operational=operational.value,
    )
