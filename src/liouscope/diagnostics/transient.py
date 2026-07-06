"""Transient layer: D14 trans-amplitude ratio, D15 kappa_trans."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .._types import TransientResult


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
    """D14: ``sup_t ||e^{tL}||_2``.

    Returns the supremum operator norm of the propagator (which equals the
    relative trans-amplitude ratio for a unit-norm initial state).

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
    return TransientResult(
        trans_amplitude_ratio=ratio,
        kappa_trans=kappa,
        numerical_abscissa=omega_L,
    )
