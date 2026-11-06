"""Relaxation layer R: D5 VNE, D6 rel-entropy, D7 fidelity, D7b ent. asym.,
plus the M0-M3b fit hierarchy orchestrated through AICc with N_eff.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.optimize import least_squares

from .._consts import (
    EPS_SUPP,
    INTERVAL_METHOD_BC,
    INTERVAL_METHOD_BCA_DELETE1,
    INTERVAL_METHOD_UNAVAILABLE,
    JACKKNIFE_MAX_GRID_POINTS,
    MODEL_SELECTION_INDETERMINATE,
)
from .._types import FitResult, RelaxationResult
from ..core.lindblad import steady_state
from ..fitting.aicc import aicc, choose_model
from ..fitting.bootstrap import _jackknife, bca_ci, parametric_bootstrap
from ..fitting.car1 import is_uniform_grid, neff_car1, whiten_car1
from ..fitting.gls import _whiten as _whiten_ar1
from ..fitting.gls import fit_gls_ar1
from ..fitting.models import (
    M0,
    M1,
    M2,
    M3a,
    M3b,
    initial_guess_m0,
    initial_guess_m1,
    initial_guess_m2,
    initial_guess_m3a,
)
from ..fitting.neff import estimate_neff_geyer
from ..fitting.prony import prony_seed
from ..numerics.kronecker import unvec, vec
from ..numerics.linalg import support_check
from ..numerics.propagation import (
    BACKEND_AUTO,
    UnrepresentableTrajectoryError,
    propagate_trajectory,
)
from .spectral import compute_spectral_layer

# ``UnrepresentableTrajectoryError`` is re-exported: it is raised through
# ``compute_relaxation_layer`` and callers catch it from this module.
__all__ = [
    "MIN_SAMPLES_PER_FAST_EFOLD",
    "RELAXATION_HORIZON",
    "RELAXATION_N_POINTS",
    "UnderResolvedTransientWarning",
    "UnrepresentableRelaxationWindowError",
    "UnrepresentableTrajectoryError",
    "compute_relaxation_layer",
    "decay_rates",
    "default_relaxation_grid",
    "entanglement_asymmetry",
    "fastest_decay_rate",
    "fidelity",
    "relative_entropy",
    "samples_per_fast_efolding",
    "trace_distance",
    "von_neumann_entropy",
]


# Default relaxation window, expressed in the system's OWN slowest relaxation
# time ``1 / Delta`` (D1) rather than in absolute time units.
#
# A decay rate carries dimension 1/time, so any *absolute* default window is
# implicitly a claim about the caller's unit of time. The legacy default
# ``linspace(0.0, 10.0, 80)`` made the whole relaxation layer -- D5, D6, D7,
# the M0..M3b AICc comparison, ``beta_D``, its BCa interval and the D17
# gap-rate check -- silently dependent on that choice: under the pure change of
# rate units ``L -> cL`` (identical physics, different unit of time) ``beta_D``
# drifted by >20%, the D17 linear rate ``beta_D_linear`` was wrong by a factor
# ~430 at ``c = 1e-6``, and ``diagnose()`` returned four different A-classes
# across ``c in [1e-6, 1e2]``. Rates in real systems are MHz or GHz, not O(1),
# so the affected regime is the common one, not an exotic corner.
#
# ``[0, HORIZON / Delta]`` fixes the window at a fixed number of e-foldings of
# the slowest mode, which is the only unit-invariant choice: it is carried
# along by the rescaling and therefore samples the same dimensionless curve
# whatever units the caller works in. ``HORIZON = 10`` is ``e^-10 ~ 4.5e-5`` of
# the initial deviation -- deep enough to identify the rate, shallow enough
# that the tail is not almost entirely numerical floor.
#
# Chosen to be exactly backward compatible at ``Delta = 1``: the grid is then
# bit-identical to the legacy ``linspace(0.0, 10.0, 80)``.
RELAXATION_HORIZON: float = 10.0
RELAXATION_N_POINTS: int = 80

# Legacy absolute window, retained ONLY for the degenerate case where no decay
# scale exists (``Delta <= 0``: no isolated steady state, or a gap that floored
# to zero). There is no physical timescale to scale by there, so any window is
# a convention; keeping the historical one avoids inventing a second arbitrary
# constant and keeps those runs comparable with earlier releases.
_LEGACY_T_MAX: float = 10.0


# Minimum samples per e-folding of the WORST-RESOLVED decaying mode before
# that mode counts as stepped over rather than measured. Below 1 the component
# loses more than 63% of its amplitude between consecutive samples, so no fit
# can identify it from this grid.
#
# A curve-shape heuristic was tried first and rejected: the share of the decay
# landing in the first interval measures how much AMPLITUDE the fast component
# carries, not whether it was RESOLVED, and the two are independent (measured
# 0.49 for a fully unsampled 1e-6/1 separation against 0.46 for a mild 0.1/1
# one). The spectrum answers the actual question directly.
MIN_SAMPLES_PER_FAST_EFOLD: float = 1.0

# Number of samples the two-scale grid places inside the fast mode's own
# window, as a fraction of ``n_points``. With ``n_points = 80`` and
# ``horizon = 10`` this puts 40 points across ten e-foldings of the fastest
# mode -- exactly 4 samples per fast e-folding, INDEPENDENTLY of the rates,
# because the segment is measured in that mode's own relaxation time. The
# remaining points carry the window out to ``horizon / gap`` so the slow mode
# is still measured over the same ten e-foldings as before.
_FAST_SEGMENT_FRAC: float = 0.5


def decay_rates(
    L_super: np.ndarray, *, eigenvalues: np.ndarray | None = None
) -> np.ndarray:
    """Distinct positive decay rates ``-Re lambda`` of the Liouvillian.

    Every mode of ``exp(L t)`` decays as ``exp(-r t)`` with ``r = -Re lambda``;
    the zero mode (steady state) and any numerically non-finite eigenvalue are
    dropped. Returned sorted ascending, so ``[-1]`` is the fastest rate.
    Empty when nothing decays.

    ``eigenvalues``, when given, is the spectrum to read the rates from -- the
    one the spectral layer has already computed AND certified (ROUND-21
    REVIEW, external, PR #127). Without it this function launches a bare
    ``np.linalg.eigvals`` on ``L_super``: on a generator where the primary
    complex driver had failed and :func:`certified_eigvals` succeeded only
    through its real-driver or Schur repair route, that call repeats the
    original failure inside the relaxation layer, and where it does not fail
    it reads rates off the rejected primary spectrum rather than the one D1
    was certified on. Forwarding the certified spectrum makes the grid, the
    resolution guard and D1 read ONE set of eigenvalues.
    """
    if eigenvalues is None:
        eigenvalues = np.linalg.eigvals(np.asarray(L_super))
    rates = -np.real(np.asarray(eigenvalues))
    positive = rates[np.isfinite(rates) & (rates > 0.0)]
    return np.sort(positive)


def fastest_decay_rate(
    L_super: np.ndarray, *, eigenvalues: np.ndarray | None = None
) -> float:
    """``max(-Re lambda)``: the rate that sets the finest timescale to resolve.

    Returns NaN when no finite positive rate exists -- fail-closed, so a caller
    sizing a grid from this value falls back to its documented default instead
    of scaling by a fabricated rate. ``eigenvalues`` is forwarded to
    :func:`decay_rates`; see there.
    """
    rates = decay_rates(L_super, eigenvalues=eigenvalues)
    return float(rates[-1]) if rates.size else float("nan")


class UnrepresentableRelaxationWindowError(ValueError):
    """The decay scale IS resolved, but its window overflows float64.

    Deliberately its own type rather than a bare ``ValueError``: a caller has
    to be able to tell this apart from a malformed argument, because the
    remedy is different (rescale the generator, or supply ``t_grid``) and
    because it is NOT the "no usable decay scale" case, which keeps its
    documented absolute-window fallback. Modelled on
    :class:`liouscope.core.lindblad.DegenerateSteadyStateError`, the repo's
    existing fail-closed domain error.
    """


def default_relaxation_grid(
    gap: float,
    *,
    fast_rate: float | None = None,
    horizon: float = RELAXATION_HORIZON,
    n_points: int = RELAXATION_N_POINTS,
) -> np.ndarray:
    """Time grid spanning ``horizon`` e-foldings of the slowest mode.

    Without ``fast_rate`` this is ``linspace(0, horizon / gap, n_points)``:
    a uniform window measured in the system's own relaxation time ``1 / gap``
    (D1). See the module-level note on :data:`RELAXATION_HORIZON` for why the
    default must not be an absolute window.

    With a ``fast_rate`` that the uniform grid cannot resolve, the grid becomes
    **two-scale**: the first ``_FAST_SEGMENT_FRAC`` of the points cover
    ``[0, horizon / fast_rate]`` -- ten e-foldings of the FASTEST mode -- and
    the rest carry the window out to ``horizon / gap`` as before. Both ends of
    the spectrum are then sampled by one grid, at 4 samples per fast e-folding
    for the default constants, whatever the separation.

    This replaces a disclosure with a repair. The previous docstring argued a
    non-uniform grid was impossible because the GLS layer "whitens with a
    single AR(1) coefficient, which presumes a constant sample interval". That
    is a property of the DISCRETE parametrisation, not of the noise: the
    stationary continuous-time process has ``Corr(t, t+d) = exp(-theta d)``, so
    whitening with the per-step ``exp(-theta dt_k)`` is valid on ANY grid.
    :mod:`liouscope.fitting.car1` implements that, and the fitting layer
    switches to it whenever the grid is non-uniform -- measured median lag-1
    autocorrelation of the whitened residuals on a two-scale grid: 0.3738 with
    one constant rho against 0.0728 with the per-step coefficient, the latter
    equal to the uniform positive control, where both schemes give 0.0728.

    The uniform grid is kept, bit-for-bit, whenever it already resolves the
    fast mode (``1 / (fast_rate * dt) >= MIN_SAMPLES_PER_FAST_EFOLD``), so no
    system that was well sampled before changes at all, and the switch happens
    exactly in the regime that previously only got a warning.

    ``gap <= 0`` or a non-finite ``gap`` means no usable decay scale was
    resolved; the historical absolute window is returned in that case. A
    missing, non-finite or non-positive ``fast_rate`` likewise falls back to
    the uniform grid rather than guessing a fast scale.

    A gap that is finite and positive but smaller than ``horizon /
    float64.max`` has no REPRESENTABLE relaxation window: ``horizon / gap``
    overflows. That case does NOT join the "no usable decay scale" branch
    above -- round 18 sent it there and round 19 measured what that costs --
    but raises :class:`UnrepresentableRelaxationWindowError` (round-19 review,
    external, PR #127); see the guard below.
    """
    if not np.isfinite(gap) or gap <= 0.0:
        return np.linspace(0.0, _LEGACY_T_MAX, n_points)
    t_max = horizon / float(gap)
    # ROUND-18 REVIEW (external, PR #127). ``gap`` can pass the finiteness
    # gate above and still leave the window unrepresentable: for a positive
    # ``gap < horizon / float64.max`` (measured: 5.56e-309 with the default
    # horizon) this division overflows to ``inf`` and ``np.linspace(0, inf,
    # n)`` produces ``[nan, inf, inf, ...]``. That grid was returned verbatim
    # whenever ``fast_rate`` was absent or unusable, and the two-scale
    # post-condition below degrades to the SAME invalid ``uniform``, so
    # ``compute_relaxation_layer`` fed ``inf``/``nan`` times into ``expm`` and
    # the fits. The condition is on the DERIVED quantity, not on ``gap``: what
    # is unrepresentable is the window, and no threshold on the rate alone
    # expresses that (it depends on ``horizon``). Failing into the documented
    # legacy window is the same fail-closed answer the non-finite-gap branch
    # already gives -- "no usable decay scale was resolved" -- rather than a
    # fabricated one, and it is reached only from an overflow, so every
    # representable window is untouched.
    if not np.isfinite(t_max) or t_max <= 0.0:
        # ROUND-19 REVIEW (external, PR #127). Round 18 stopped the invalid
        # ``[nan, inf, inf, ...]`` grid and then handed back the absolute
        # legacy window, which is a DIFFERENT wrong answer: the fits still
        # ran, on a trajectory that says nothing about this system.
        #
        # Measured on an amplitude-damping generator rescaled by 1e-307
        # (certified gap 5e-308): ``diagnose()`` reported ``beta_D ~
        # -1.9e-18``, ``beta_D_linear ~ 84``, class A12 and -- the label that
        # makes it undetectable downstream -- ``t_grid_source="gap_scaled"``,
        # while the identical unscaled physics reports rates near 1 and class
        # A5. Numbers were produced, they were wrong, and nothing in the
        # report said so.
        #
        # These are two different states and the layer must not merge them:
        # ``gap <= 0`` or non-finite is NOT DETERMINABLE (no decay scale was
        # resolved), and the documented legacy window is the honest answer
        # there. A finite positive gap whose window overflows is DETERMINED
        # BUT NOT REPRESENTABLE in the caller's rate unit -- the scale exists,
        # float64 cannot hold the window it implies. Withholding is the only
        # answer that does not fabricate one; the caller can rescale the
        # generator or pass an explicit ``t_grid``.
        raise UnrepresentableRelaxationWindowError(
            "default_relaxation_grid: the relaxation window horizon/gap = "
            f"{horizon}/{gap} overflows float64, so no gap-scaled grid exists "
            "in these rate units. The gap IS resolved, so this is not the "
            "'no usable decay scale' case and the absolute legacy window "
            "would describe a different system: on the reviewer's fixture it "
            "produced a full set of finite rates and an A12 class label for a "
            "generator whose unscaled twin is A5. Rescale the generator to "
            "rate units of order one, or pass an explicit t_grid."
        )
    uniform = np.linspace(0.0, t_max, n_points)

    # --- fail-closed guards: any doubt keeps the historical uniform grid ---
    if fast_rate is None or not np.isfinite(fast_rate) or fast_rate <= 0.0:
        return uniform
    n_fast = int(n_points * _FAST_SEGMENT_FRAC)
    if n_fast < 2 or n_points - n_fast < 2:
        return uniform
    dt_uniform = float(uniform[1] - uniform[0])
    if dt_uniform <= 0.0:
        return uniform
    # Already resolved: leave it exactly as it was.
    if 1.0 / (float(fast_rate) * dt_uniform) >= MIN_SAMPLES_PER_FAST_EFOLD:
        return uniform
    t_split = horizon / float(fast_rate)
    if not np.isfinite(t_split) or t_split <= 0.0 or t_split >= t_max:
        return uniform

    early = np.linspace(0.0, t_split, n_fast, endpoint=False)
    late = np.linspace(t_split, t_max, n_points - n_fast)
    # ``np.asarray`` is not decoration: numpy's own stubs type
    # ``np.concatenate`` as ``Any`` up to 2.2.x and as ``ndarray`` from 2.3
    # onward, so without it this function silently returns ``Any`` and every
    # caller downstream loses type checking on the grid. The repo already
    # settles this the same way in ``transient.py`` (``_transient_grid``).
    # Runtime is unaffected: ``concatenate`` already returns a float64 array,
    # so this is a no-op view, not a copy or a cast.
    grid = np.asarray(np.concatenate([early, late]), dtype=float)
    # Structural post-condition rather than trust in the arithmetic above: a
    # grid that is not strictly increasing and finite would corrupt every
    # consumer downstream (the propagator, the whitening, N_eff), so a failure
    # here degrades to the uniform grid instead of propagating.
    if (
        grid.size != n_points
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
        or grid[0] != 0.0
    ):
        return uniform
    return grid


class UnderResolvedTransientWarning(UserWarning):
    """A decaying mode is stepped over by the grid rather than measured.

    A grid can only resolve dynamics slower than the largest interval it leaves
    unsampled in the region where that mode still has amplitude. On the DEFAULT
    path this is now largely repaired rather than merely disclosed: when the
    uniform gap-scaled window cannot resolve the fastest mode,
    :func:`default_relaxation_grid` switches to a two-scale grid that samples
    both ends of the spectrum, and the fitting layer switches to the CAR(1)
    residual model that is valid on it (:mod:`liouscope.fitting.car1`).

    Two situations remain, and this warning exists for them:

    * a CALLER-supplied ``t_grid`` that does not resolve its own system -- for
      instance one starting long after the mode has decayed, where no step size
      compensates for the unsampled lead-in;
    * an INTERMEDIATE timescale on the default two-scale grid. The two segments
      resolve the fastest and the slowest mode; a mode between them can still
      fall between the coarse late samples. Measured on three independent
      damped qubits with rates 1e-6, 1e-3 and 1, the middle mode is sampled
      0.0039 times per e-folding and the warning fires -- correctly.

    Widening the window is still not a fix -- an absolute window resolves the
    fast mode and misses the relaxation entirely (measured on two independent
    damped qubits with rates 1e-6 and 1: the absolute window put
    ``beta_D_linear`` 3.5e4 relative away from the true gap, the gap-scaled
    window 0.58). What changed is that a non-uniform grid IS now available,
    because the whitening it needs was implemented rather than assumed
    impossible.

    When this warning does fire, the rates the layer reports describe the
    dynamics the window resolves; a caller who needs the unresolved component
    must supply a ``t_grid`` covering that timescale (and read the resulting
    rates as describing that window).
    """


def _resolution_detail(
    L_super: np.ndarray,
    t_grid: np.ndarray,
    *,
    eigenvalues: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """How finely the grid samples the WORST-RESOLVED decaying mode.

    Returns ``(samples_per_efolding, rate, blind_interval, blind_start)`` for
    the mode that the grid resolves least well. ``blind_start`` is the time the
    offending interval begins, which is what distinguishes an unsampled lead-in
    (``start == 0 < t[0]``) from an ordinary sample interval -- a distinction a
    comparison against ``t[1] - t[0]`` gets wrong on a two-scale grid, where
    the largest late step exceeds the first step without any lead-in existing.

    For each decay rate ``r`` of ``L_super`` this forms ``1 / (r * blind_r)``,
    where ``blind_r`` is the largest interval the grid leaves unsampled *while
    that mode still has amplitude*, and returns the MINIMUM over modes.

    Two deliberate choices:

    ``blind_r`` counts the lead-in from ``t = 0``, not merely the step. The
    dynamics starts at ``rho_initial`` at ``t = 0`` while the trajectory is
    only evaluated from ``t[0]``, and :func:`liouscope.diagnose` permits any
    non-negative start, so a late-starting grid leaves a lead-in that no step
    size compensates for: a rate-1 mode on ``linspace(100, 101, 101)`` is
    sampled 100x per e-folding by its step and is nonetheless long gone by the
    first sample (amplitude ``e^-100``, and the whole relative-entropy curve
    measures identically zero while the fit still returns a confident-looking
    rate).

    ``blind_r`` only considers intervals that START before ``1 / r``. A
    coarse interval after the mode has decayed is not blindness -- there is
    nothing left to see -- and counting it would make every two-scale grid look
    under-resolved because of its own late segment. On a uniform grid the
    restriction is inert (every interval has the same length), so this reduces
    exactly to the historical ``1 / (r_max * max(dt, t[0]))``; the pinned
    values for uniform and late-starting grids are unchanged.

    Taking the minimum over ALL modes rather than only the fastest is what
    makes the measure honest on a non-uniform grid. The old form read
    ``t[1] - t[0]``, which on a fine-then-coarse grid reports the FINE step and
    would wave through a middle timescale that falls entirely between the
    coarse samples. That case is created, not hypothetical: it is exactly the
    intermediate scale a two-scale grid does not resolve.

    Returns ``inf`` when nothing decays (no finite positive rate), and also
    when every mode is resolved so finely that ``r * blind_r`` underflows to
    zero -- the limit of the ratio, i.e. unbounded sampling resolution. NaN
    is reserved for the grid having no usable interval at all, which is a
    different statement and must not be conflated with the first.
    """
    t = np.asarray(t_grid, dtype=float)
    if t.size < 2 or not np.all(np.isfinite(t)):
        return float("nan"), float("nan"), float("nan"), float("nan")
    rates = decay_rates(L_super, eigenvalues=eigenvalues)
    if rates.size == 0:
        return float("inf"), float("nan"), float("nan"), float("nan")

    # Unsampled intervals as (start, length): the lead-in [0, t[0]] first, then
    # every grid step. Starts are non-decreasing for an increasing grid, so a
    # running maximum of the lengths answers "largest interval starting before
    # T" by a single search.
    lead_in = float(t[0]) if t[0] > 0.0 else 0.0
    # Same ``Any``-leak as in ``default_relaxation_grid`` above: unannotated,
    # everything computed from these two arrays below (the running maximum, the
    # searchsorted index, the returned interval) would be unchecked on numpy
    # <= 2.2 while checking fine on newer stubs.
    starts = np.asarray(np.concatenate([[0.0], t[:-1]]), dtype=float)
    lengths = np.asarray(np.concatenate([[lead_in], np.diff(t)]), dtype=float)
    if not np.all(np.isfinite(lengths)) or np.any(lengths[1:] <= 0.0):
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    lengths = lengths[order]
    # Running maximum AND the index that produced it, so the warning can name
    # where the offending interval is instead of inferring it from a size
    # comparison that a two-scale grid breaks.
    running_max = np.maximum.accumulate(lengths)
    running_argmax = np.zeros(lengths.size, dtype=int)
    best = -np.inf
    best_i = 0
    for i, val in enumerate(lengths):
        if float(val) > best:
            best, best_i = float(val), i
        running_argmax[i] = best_i

    worst = float("inf")
    worst_rate = float("nan")
    worst_blind = float("nan")
    worst_start = float("nan")
    # ROUND-19 REVIEW (external, PR #127). ``measured`` replaces the old
    # ``np.isfinite(worst)`` test at the end, and the two are NOT the same
    # question. "No usable interval" and "every interval resolves the mode
    # perfectly" both left ``worst`` at ``inf``, so the second was reported as
    # the first -- a grid that resolves everything was described as a grid with
    # no usable step. Which of the two happened is exactly what this flag
    # records, and nothing else about the loop changes: for any rate that was
    # already reachable, ``not measured or value < worst`` selects the same
    # mode as ``value < worst`` did.
    measured = False
    for r in rates:
        horizon_r = 1.0 / float(r)
        idx = int(np.searchsorted(starts, horizon_r, side="left"))
        # ``starts[0] == 0.0 < horizon_r`` for any finite positive rate, so the
        # slice is never empty.
        j = max(idx, 1) - 1
        blind = float(running_max[j])
        if blind <= 0.0:
            continue
        # Same review, the reported line. ``r`` and ``blind`` are both finite
        # and strictly positive here, yet their PRODUCT can be zero: below
        # ~5e-324 it underflows, and ``1.0 / 0.0`` raises ZeroDivisionError.
        # Measured: an amplitude-damping generator scaled by 1e-200 on the
        # valid increasing grid ``[0, 1e-200, 2e-200]`` killed both
        # ``samples_per_fast_efolding`` and ``compute_relaxation_layer``
        # before any fit ran -- a crash caused by the choice of rate UNITS, on
        # a quantity that is dimensionless and therefore unit-free.
        #
        # The mathematical limit is the honest value: ``1 / (r * blind)``
        # grows without bound as the product goes to zero, i.e. the mode is
        # sampled arbitrarily often per e-folding. ``inf`` is already this
        # function's documented encoding for "nothing decays over this grid",
        # and the consumer in ``compute_relaxation_layer`` gates its warning on
        # ``np.isfinite``, so an infinitely well resolved mode correctly warns
        # about nothing. Returning NaN instead -- the "invalid" encoding --
        # would report an unmeasurable grid where the grid is in fact perfect.
        product = float(r) * blind
        value = float("inf") if product <= 0.0 else 1.0 / product
        if not measured or value < worst:
            measured = True
            worst = value
            worst_rate = float(r)
            worst_blind = blind
            worst_start = float(starts[running_argmax[j]])
    if not measured:
        # Every interval was zero-length: no usable step.
        return float("nan"), float("nan"), float("nan"), float("nan")
    return worst, worst_rate, worst_blind, worst_start


def samples_per_fast_efolding(
    L_super: np.ndarray,
    t_grid: np.ndarray,
    *,
    eigenvalues: np.ndarray | None = None,
) -> float:
    """Samples per e-folding of the worst-resolved mode.

    Thin wrapper over :func:`_resolution_detail`; see its docstring. The two
    share one implementation deliberately, so the number this returns and the
    number quoted in :class:`UnderResolvedTransientWarning` can never describe
    different intervals. ``eigenvalues`` is forwarded to :func:`decay_rates`.
    """
    return _resolution_detail(L_super, t_grid, eigenvalues=eigenvalues)[0]


def _decay_fraction_from_resolution(samples_per_efolding: float) -> float:
    """Fraction decayed across the blind interval from its sampling ratio.

    ``samples_per_efolding = 1 / (rate * blind_interval)``.  When the
    denominator overflows float64 the representable ratio is exactly 0.0; the
    physical limit is then complete decay, not a division-by-zero error.
    ``expm1`` preserves accuracy in the opposite, well-resolved limit.
    """
    s = float(samples_per_efolding)
    if s == 0.0:
        return 1.0
    return -float(np.expm1(-1.0 / s))


def _propagate(
    L_super: np.ndarray,
    rho0: np.ndarray,
    t_grid: np.ndarray,
    *,
    backend: str = BACKEND_AUTO,
) -> tuple[np.ndarray, str]:
    """Propagate ``rho(t) = exp(L t) rho_0``; return the trajectory and the backend.

    The backend (``"dense_expm"`` reference or ``"expm_action"``, issue #162)
    is chosen by :func:`liouscope.numerics.propagation.select_trajectory_backend`
    under ``backend="auto"``. Both fail closed with
    :class:`UnrepresentableTrajectoryError` instead of returning NaN/inf.
    """
    d = rho0.shape[0]
    result = propagate_trajectory(L_super, vec(rho0), t_grid, backend=backend)
    traj = np.empty((t_grid.size, d, d), dtype=complex)
    for k in range(t_grid.size):
        # ``states`` holds rho_0 exactly at t == 0, already validated in the
        # complex128 working dtype; vec/unvec is a pure reshape, so this is
        # rho_0 itself without a second, unchecked narrowing store.
        traj[k] = unvec(result.states[k], d=d)
    return traj, result.backend


def _trajectory_invariants(traj: np.ndarray) -> tuple[float, float, float]:
    """Measured physical invariants of a propagated trajectory (issue #162).

    Returns ``(max |tr rho - 1|, max ||rho - rho^dag||_max, min eig(herm(rho)))``;
    NaN for an empty trajectory. The last value is the positivity drift: a
    CPTP evolution keeps it >= 0 exactly, round-off pushes it below by a few ulp.
    """
    if traj.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    adjoint = np.conj(np.swapaxes(traj, 1, 2))
    trace_error = float(np.max(np.abs(np.einsum("kii->k", traj) - 1.0)))
    hermiticity = float(np.max(np.abs(traj - adjoint)))
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (traj + adjoint))))
    return trace_error, hermiticity, min_eig


def _evolve(
    L_super: np.ndarray,
    rho0: np.ndarray,
    t_grid: np.ndarray,
    *,
    backend: str = BACKEND_AUTO,
) -> np.ndarray:
    """Propagate ``rho(t) = exp(L t) rho_0`` and return the trajectory."""
    traj, _backend = _propagate(L_super, rho0, t_grid, backend=backend)
    return traj


def von_neumann_entropy(rho: np.ndarray) -> float:
    """D5: ``S(rho) = -Tr(rho log rho)``."""
    rho = np.asarray(rho)
    rho_sym = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rho_sym)
    evals = evals[evals > EPS_SUPP]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log(evals)))


def relative_entropy(rho: np.ndarray, pi: np.ndarray) -> float:
    """D6: ``D(rho || pi) = Tr(rho (log rho - log pi))``.

    Uses ``support_check`` to regularise (anchor J).
    """
    rho = np.asarray(rho)
    pi = np.asarray(pi)
    _, pi_reg = support_check(rho, pi)
    rho_sym = 0.5 * (rho + rho.conj().T)
    pi_sym = 0.5 * (pi_reg + pi_reg.conj().T)
    evals_rho, evecs_rho = np.linalg.eigh(rho_sym)
    evals_pi, evecs_pi = np.linalg.eigh(pi_sym)
    log_pi = (
        evecs_pi
        @ np.diag(np.log(np.clip(evals_pi, EPS_SUPP, None)))
        @ evecs_pi.conj().T
    )
    log_rho = (
        evecs_rho
        @ np.diag(np.log(np.clip(evals_rho, EPS_SUPP, None)))
        @ evecs_rho.conj().T
    )
    val = float(np.real(np.trace(rho_sym @ (log_rho - log_pi))))
    return max(val, 0.0)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """D7: Uhlmann fidelity ``F = Tr sqrt( sqrt(rho) sigma sqrt(rho) )``.

    Computes ``sqrt(rho)`` via spectral decomposition so rank-deficient
    inputs (pure-state limits) do not trigger ``sqrtm`` warnings.
    """
    rho = np.asarray(rho)
    rho_sym = 0.5 * (rho + rho.conj().T)
    evals, evecs = np.linalg.eigh(rho_sym)
    evals = np.clip(evals.real, 0.0, None)
    sqrt_evals = np.sqrt(evals)
    rho_sqrt = (evecs * sqrt_evals) @ evecs.conj().T
    inner = rho_sqrt @ np.asarray(sigma) @ rho_sqrt
    eigs = np.linalg.eigvalsh(0.5 * (inner + inner.conj().T))
    eigs = np.clip(eigs.real, 0.0, None)
    return float(np.sum(np.sqrt(eigs)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """D_tr: half the trace-norm distance ``(1/2) || rho - sigma ||_1`` (LIOU-F-018).

    For Hermitian inputs this equals ``(1/2) sum_i |lambda_i|`` over the
    eigenvalues of ``rho - sigma`` (Nielsen & Chuang). It is the observable
    relaxation metric complementing D5 (von-Neumann entropy), D6 (relative
    entropy) and D7 (Uhlmann fidelity): a metric (``0 <= D_tr <= 1`` for density
    operators, ``D_tr = 0`` iff ``rho == sigma``) that contracts monotonically
    under any CPTP map (data-processing inequality) and upper-bounds the
    distinguishability of ``rho`` and ``sigma`` by measurement (Helstrom).

    The difference is Hermitised before the eigenvalue decomposition so a tiny
    non-Hermitian numerical excursion in either argument cannot leak a complex
    part into the (real, by construction) distance.
    """
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    if rho.shape != sigma.shape:
        raise ValueError(
            f"trace_distance requires equal shapes, got {rho.shape} vs {sigma.shape}"
        )
    diff = rho - sigma
    diff_sym = 0.5 * (diff + diff.conj().T)
    evals = np.linalg.eigvalsh(diff_sym)
    return float(0.5 * np.sum(np.abs(evals)))


def entanglement_asymmetry(rho: np.ndarray) -> float:
    r"""D7b: Rylands et al. 2024 entanglement-asymmetry measure (single block).

    The entanglement asymmetry of a subsystem state ``rho`` with respect to a
    U(1) charge ``Q`` (here the total magnetisation of the block) is

    .. math::

        \Delta S_A \;=\; S(\rho_Q) - S(\rho),
        \qquad \rho_Q \;=\; \sum_q \Pi_q\, \rho\, \Pi_q,

    where ``\Pi_q`` projects onto the eigenspace of ``Q`` with eigenvalue ``q``.
    Block-dephasing ``rho -> rho_Q`` removes exactly the coherences *between*
    distinct charge sectors, so ``\Delta S_A >= 0`` and ``\Delta S_A = 0`` iff
    ``rho`` commutes with ``Q`` (i.e. the state is symmetric). This is the
    published Ares-Murciano-Calabrese / Rylands construction, and it differs
    from a full single-qubit Pauli twirl, which would instead maximally mix the
    twirled qubit and report an entropy *deficit* rather than a symmetry breaking
    (a Bell state ``(|01> + |10>)/sqrt2`` lives in the single sector ``q = 1`` and
    is therefore exactly symmetric, ``\Delta S_A = 0``).

    For the supported two-qubit block (``d = 4``) the charge is the number
    operator ``Q = n_1 + n_2`` in the computational basis, with sectors
    ``q in {0, 1, 2}`` (basis-state populations by Hamming weight). Returns
    ``nan`` outside the supported ``d = 4`` case; the report layer maps that to
    ``entanglement_asymmetry=None``.
    """
    rho = np.asarray(rho)
    d = rho.shape[0]
    if d not in (4,):
        return float("nan")
    # Charge = total magnetisation / particle number of the 2-qubit block:
    # basis index i in {0,1,2,3} -> Hamming weight (00->0, 01/10->1, 11->2).
    charge = np.array([bin(i).count("1") for i in range(d)])
    # rho_Q = sum_q Pi_q rho Pi_q keeps only the intra-sector (equal-charge)
    # blocks of rho; inter-sector coherences are projected out. Because the
    # computational basis diagonalises Q, this is the Hadamard mask that zeroes
    # every entry rho[i, j] with charge[i] != charge[j].
    mask = charge[:, None] == charge[None, :]
    rho_q = np.where(mask, rho, 0.0)
    # von_neumann_entropy Hermitises defensively; the mask preserves Hermiticity
    # exactly (it is symmetric), so rho_q stays a valid density operator.
    return float(max(von_neumann_entropy(rho_q) - von_neumann_entropy(rho), 0.0))


def _fit_with_model(
    model_name: str,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[FitResult, np.ndarray]:
    fns = {"M0": M0, "M1": M1, "M2": M2, "M3a": M3a, "M3b": M3b}
    seeds = {
        "M0": initial_guess_m0,
        "M1": initial_guess_m1,
        "M2": initial_guess_m2,
        "M3a": initial_guess_m3a,
        "M3b": lambda t_, y_: np.asarray(prony_seed(t_, y_)),
    }
    model = fns[model_name]
    p0 = seeds[model_name](t, y)
    fit = fit_gls_ar1(model, t, y, p0)
    # ``k`` is the number of parameters ESTIMATED from this data set, and on a
    # non-uniform grid that includes two residual parameters besides the mean
    # function (PR #127 review, three findings resolved together):
    #
    # * a correlation parameter. CAR(1) fits ``theta``; the AR(1) fallback of
    #   ``fit_gls_ar1`` fits ``rho``. Counting only the first gave the fallback
    #   candidates of a ``car1_mixed`` comparison a smaller penalty for the same
    #   number of fitted parameters;
    # * the stationary variance, profiled per candidate from all ``n`` whitened
    #   residuals. Theta now maximises that same likelihood
    #   (``estimate_car1_theta``), so the count and the likelihood describe one
    #   probability model.
    #
    # The small-sample correction ``2k(k+1)/(N_eff-k-1)`` is nonlinear in ``k``,
    # so an omission is not a constant offset: it favours the higher-dimensional
    # candidates exactly when ``N_eff`` is small. The predicate is the GRID, the
    # one property fixed across all candidates of a selection, not
    # ``isfinite(theta_car1)``, which varies between them.
    #
    # A UNIFORM grid keeps the historical ``k`` = mean-function parameters on
    # purpose. It is self-consistent within one selection, and changing it
    # re-ranks every existing uniform-grid result, which is the separate
    # methodology change of PR #157 (with its anchor evidence).
    uniform_grid = is_uniform_grid(t)
    k = int(p0.size) + (0 if uniform_grid else 2)
    # N_eff must be computed under the SAME residual model the fit whitened
    # with, or the AICc it feeds compares likelihoods from one model against a
    # sample size from another. The Geyer IPS estimator sums autocorrelations
    # indexed by LAG; on a grid whose step varies, lag 1 is not a fixed time
    # separation, so the sequence it sums is not an autocorrelation function.
    # The CAR(1) form is exact there and reduces to the same double sum on a
    # uniform grid. Fail-closed: if it cannot be formed, fall back to Geyer.
    n_eff = float("nan")
    if np.isfinite(fit.theta_car1):
        n_eff = neff_car1(t, fit.theta_car1)
    if not np.isfinite(n_eff):
        n_eff = estimate_neff_geyer(fit.residuals)
    # Round-17 review (PR #121). ``fit_gls_ar1`` flips ``success`` when the
    # final model evaluation ended inside the magnitude guards -- convergence
    # for the wrong reason, the plateau where every derivative is exactly
    # zero. Flipping a flag no consumer reads changed nothing: the failed fit
    # still carried a finite log-likelihood, hence a finite AICc, and could
    # WIN ``choose_model`` and supply the reported decay rate.
    #
    # Made non-selectable HERE rather than enforced at each consumer, and
    # deliberately so: ``_dominant_rate``, ``compute_relaxation_layer``, the
    # bootstrap and the holdout path are four independent obligations to
    # remember, which is precisely the shape of the four-fold miss this same
    # review round found in the zero-mode filters. ``choose_model`` already
    # drops non-finite entries, so ``inf`` is the existing vocabulary for
    # "not a candidate"; the fit itself stays in ``fits`` so the report still
    # shows that the model was tried and why it lost.
    # AICc sample size on a non-uniform grid: the OBSERVED count, not N_eff. The
    # joint CAR(1) likelihood already models the serial covariance, so an ESS
    # derived from that same dependence would count it twice (Hurvich & Tsai
    # 1989; route of PR #157). This is what makes ``k = p + 2`` safe to count:
    # with N_eff below ``k + 1`` the correction is infinite. Measured with known
    # ground truth, 40 seeds on the default two-scale grid, CAR(1) noise at
    # theta = 2: truth M0, ``p+2, N_eff``: 36-40/40 runs had NO finite score;
    # ``p+2, n_obs``: 0/40 and M0 selected 29/40. Truth two-rate at theta = 0.05:
    # ``N_eff`` selected M0 40/40, ``n_obs`` M2 40/40. N_eff stays in the report
    # as uncertainty evidence. A uniform grid keeps N_eff, see above.
    #
    # Pre-registered study (cross-family review LIO-02):
    # benchmarks/pr127_car1_aicc_identifiability.py, result in
    # benchmarks/results/pr127_car1_aicc_identifiability.json (40 seeds,
    # sigma 1e-3, theta in {0.05, 0.5, 2}). On the two-scale grid ``n_obs``
    # selects the truth at least as often as ``N_eff`` in 6 of 6 cells (rule:
    # SUPPORTED). ``N_eff`` leaves 29-39 of 40 runs with no finite score.
    # Truth M2: ``n_obs`` 40/40 at every theta. Truth M0: ``n_obs`` 8, 20, 21 of
    # 40 at theta = 0.05, 0.5, 2. The losses go mostly to M1 and M2.
    # At theta = 0.05 the CAR(1) rate itself is NOT identifiable on this window
    # (fitted within a factor 2 of the truth in 18/40 and 17/40 runs). A noise
    # correlation that slow is close to a constant offset, which M1's ``C``
    # absorbs. This is an open limitation. The INDETERMINATE outcome of
    # ``_select_model`` covers rounding- and solver-level ties only and fired
    # in 0 of 480 of these runs. A statistical indeterminacy rule
    # (near-ties: best two AICc within 2 units in 21/40 of the theta = 0.05, M0
    # runs) is a separate methodology decision and is not made here.
    aicc_n = n_eff if uniform_grid else float(t.size)
    aic = aicc(fit.log_likelihood, k, aicc_n) if fit.success else float("inf")
    fit_result = FitResult(
        model=model_name,
        params=fit.params,
        log_likelihood=fit.log_likelihood,
        aicc=aic,
        n_eff=n_eff,
        residual_ar1_rho=fit.rho_ar1,
        success=fit.success,
        residual_theta_car1=float(fit.theta_car1),
        likelihood_degenerate=fit.likelihood_degenerate,
        # ROUND-2 REVIEW (PR #147). This conversion carried
        # ``likelihood_degenerate`` and dropped ``scale_unavailable``, so the
        # newly supported underflow case -- ``success=True`` with a withheld
        # ``sigma`` -- reached the report indistinguishable from an ordinary
        # fit. Its bootstrap then fails and ``compute_relaxation_layer``
        # records ``bca_ci_beta = (nan, nan)``, exactly what a jackknife
        # failure or a non-converged resample produces. The reason lived only
        # in a RuntimeWarning, which a persisted artefact does not keep.
        scale_unavailable=fit.scale_unavailable,
    )
    return fit_result, fit.params


def _beta_from_params(model_name: str, params: np.ndarray) -> float:
    if model_name == "M0":
        return float(params[1])
    if model_name == "M1":
        return float(params[1])
    if model_name == "M2":
        return float(min(params[1], params[3]))
    if model_name == "M3a":
        return float(params[2])
    if model_name == "M3b":
        return float(params[1])
    return float("nan")


_MODEL_FUNCTIONS = {"M0": M0, "M1": M1, "M2": M2, "M3a": M3a, "M3b": M3b}
# Evaluation budget of the continued optimisation in
# ``_attainable_residual_norm``. A latency guard, not a statistical choice:
# measured on V1, V3 and V5 (both curves) the continuation used 2 to 49
# evaluations except one at 176 (V1) and one runaway at a 2000 cap (V3, M3b
# with omega ~ 2e6) that cost 17.9 s of a 29 s ``diagnose``. Stopping early
# only means a smaller widening for that candidate.
_POLISH_MAX_NFEV: int = 100


def _gamma_rounding(m: int) -> float:
    """Higham's ``gamma_m = m u / (1 - m u)``: the relative forward-error bound
    of a length-``m`` floating-point inner product (``u`` = unit roundoff).
    ``inf`` once ``m u >= 1``, where no bound of this form exists."""
    mu = float(m) * 0.5 * float(np.finfo(float).eps)
    return mu / (1.0 - mu) if mu < 1.0 else float("inf")


def _loglik_rounding_band(
    n_obs: int,
    residual_norm: float,
    data_error_norm: float,
    attainable_norm: float | None = None,
) -> tuple[float, float]:
    """``(up, down)``: how far the profiled Gaussian log-likelihood of a fit
    can lie above / below the reported value, given an error of norm
    ``data_error_norm`` in the data and an optimiser that stopped at a
    residual of norm ``residual_norm`` while ``attainable_norm`` (default:
    the same) is what a converged step could still reach.

    The profile likelihood depends on the data only through the minimised
    residual norm, ``ln L_max = c - n ln ||r||``. For data ``y + delta`` the
    triangle inequality bounds the new minimum by ``||r_min|| +- ||delta||``
    (for every parameter value, hence also for the minimiser), so with
    ``q = ||delta|| / ||r||`` and ``a = ||r_attainable|| / ||r||``::

        ln L_max(y + delta) - ln L_reported  in  [-n ln(1 + q), -n ln(a - q)]

    The lower end needs no ``a``: an unconverged optimiser only ever reports
    a residual that is too LARGE. The upper end is unbounded (``inf``) once
    ``a <= q``: a residual no larger than the data error says nothing about
    how small it could become. A bound, not a first-order estimate, for a
    FIXED whitening ``W`` (pass ``||W r||`` and ``||W||_2 ||delta||``).
    Approximation: ``W`` is held at the fitted residual model (``rho`` /
    ``theta``); that model is profiled per candidate and its own response to
    the data error is not bounded.

    NaN-safe in the fail-toward-tie direction: a NaN or zero residual norm
    gives an unbounded or NaN band, which :func:`_select_model` treats as
    "cannot be excluded".
    """
    if attainable_norm is None:
        attainable_norm = residual_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        q = float(np.float64(data_error_norm) / np.float64(residual_norm))
        a = float(np.float64(attainable_norm) / np.float64(residual_norm))
    n = float(n_obs)
    up = -n * float(np.log(a - q)) if (a - q) > 0.0 else float("inf")
    down = n * float(np.log1p(q)) if q >= 0.0 else float("nan")
    return up, down


def _attainable_residual_norm(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    whiten: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Whitened residual norm the SAME candidate reaches when its optimisation
    is continued from the reported parameters to float64 tolerances, with the
    residual model held fixed -- the solver-error term of AGENTS.md (issue
    #180) for the AICc comparison. The reported fit, its parameters and its
    AICc are NOT changed; this value only widens the candidate's band.

    Measured on ``0.04 exp(-t)`` plus white noise of 20 u: M0 stopped (default
    ``least_squares`` tolerances) with an AR(1) residual correlation of 0.88
    and lost to M1 or M3a by 308 to 721 AICc units in 7 of 8 seeds although M0
    is the truth -- the comparison measured where the optimiser stopped, not
    the data. On validation system V3 (trace distance) continuing M0 lowers
    its residual by a factor 1.7e6. A linear (Gauss-Newton) prediction was
    tried first and rejected: on the near-degenerate M2 fit of V5
    (``A1 ~ -A2``, ``beta1 ~ beta2``) it predicted 0.51 of the residual, the
    continued optimisation reached 0.993, and V5 turned INDETERMINATE for
    nothing. Not a bound (the problem is non-convex): what this optimiser
    reaches, which is the quantity the comparison depends on. Never larger
    than ``||W r||``; an optimiser failure returns 0.0 (unknown -> cannot be
    excluded)."""
    p0 = np.asarray(params, dtype=float)
    y = np.asarray(y, dtype=float)

    def w_residual(p: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.asarray(whiten(y - model(t, p)), dtype=float)

    current = float(np.linalg.norm(w_residual(p0)))
    if not (np.all(np.isfinite(p0)) and np.isfinite(current)):
        return 0.0
    tol = float(np.finfo(float).eps)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = least_squares(
                w_residual, p0, x_scale="jac",
                ftol=tol, xtol=tol, gtol=tol, max_nfev=_POLISH_MAX_NFEV,
            )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
        return 0.0
    polished = float(np.linalg.norm(res.fun))
    if not np.isfinite(polished):
        return 0.0
    return min(polished, current)


def _select_model(
    fits: dict[str, FitResult],
    t: np.ndarray,
    y: np.ndarray,
    data_error: float,
) -> tuple[str, tuple[str, ...]]:
    """Pick the AICc winner among the SUCCEEDED fits, or report that the
    ranking is not resolved above the rounding level of the data.

    Returns ``(selected, tied)``: ``tied`` lists, in ladder order, every
    candidate whose AICc cannot be separated from the minimum once each score
    is widened by :func:`_loglik_rounding_band` for a per-sample data error of
    ``data_error`` (absolute, in the units of ``y``). Cross-family review of
    PR #127 (LIO-03): on a curve that is an exact single exponential to
    float64 precision every candidate fits the same rounding noise, and a
    1-ulp perturbation of the operator flipped the winner from M0 to M3a in 7
    of 25 seeds -- the selection was a property of the platform's rounding,
    not of the data.

    * One candidate left: it is the winner, exactly as ``choose_model``.
    * Several, and ``M0`` among them: ``M0``. Every other candidate nests it
      (M1 at ``C = 0``, M2 at ``A2 = 0``, M3a at ``B = 0``, M3b at
      ``omega = phi = 0``), so the data at their own resolution do not reject
      the simplest model -- the parsimony rule, not an arbitrary tie-break.
    * Several without ``M0``, M0 evaluated and excluded by a true comparison:
      :data:`MODEL_SELECTION_INDETERMINATE`. The data reject the common
      nested model, no tied member nests the others, and the data cannot
      decide between them.
    * Several, M0 not a usable candidate at all (failed fit or infinite
      AICc): the tied candidate with the fewest parameters, ladder order on
      equal count.

    The comparison is orientation-safe: a candidate is EXCLUDED only by a
    true ``lower_j > upper_winner``; a NaN on either side keeps it tied.
    With every succeeded AICc non-finite the historical fallback is kept
    (``choose_model``'s "M0", else the first succeeded model in ladder order).
    """
    selectable = {n: fr for n, fr in fits.items() if fr.success}
    if not selectable:
        return "none", ()
    finite = {n: fr for n, fr in selectable.items() if np.isfinite(fr.aicc)}
    if not finite:
        winner = choose_model({n: fr.aicc for n, fr in selectable.items()})
        if winner not in selectable:
            winner = next(iter(selectable))
        return winner, (winner,)
    winner = choose_model({n: fr.aicc for n, fr in finite.items()})
    y = np.asarray(y, dtype=float)
    n_obs = int(y.size)
    error_norm = float(data_error) * float(np.sqrt(n_obs))

    def band(name: str, polish: bool) -> tuple[float, float]:
        fr = finite[name]
        # The likelihood is that of the WHITENED residuals, so the bound is
        # taken there too: ``||W r||`` against ``||W||_2 ||delta||``. Measured
        # on the phantom ladder (seed 9 of the 1-ulp probe): the raw residual
        # norms of M0 and M3a differ by 10 %, their AICc by 103 -- the AR(1)
        # whitening removes the smooth common part, so a raw-norm band was
        # 4x too narrow and the flip survived.
        theta = float(fr.residual_theta_car1)
        rho = float(fr.residual_ar1_rho)
        if np.isfinite(theta):
            def whiten(v: np.ndarray) -> np.ndarray:
                return whiten_car1(v, t, theta)
        elif np.isfinite(rho):
            def whiten(v: np.ndarray) -> np.ndarray:
                return _whiten_ar1(v, rho)
        else:
            return float("inf"), float("nan")
        with np.errstate(all="ignore"):
            resid = y - _MODEL_FUNCTIONS[name](t, fr.params)
            w_resid = whiten(resid)
            w_resid_norm = float(np.linalg.norm(w_resid))
            attainable = (
                _attainable_residual_norm(_MODEL_FUNCTIONS[name], t, y, fr.params, whiten)
                if polish
                else w_resid_norm
            )
            w_matrix = np.column_stack([whiten(col) for col in np.eye(n_obs)])
            # ||W||_2 <= sqrt(||W||_1 ||W||_inf): sums only, no BLAS-dependent
            # factorisation (AGENTS.md, issue #180), and an upper bound.
            w_abs = np.abs(w_matrix)
            w_norm = float(
                np.sqrt(np.max(np.sum(w_abs, axis=0)) * np.max(np.sum(w_abs, axis=1)))
            )
        return _loglik_rounding_band(
            n_obs, w_resid_norm, w_norm * error_norm, attainable
        )

    # The winner's upper end uses only ``down`` (an unconverged optimiser
    # reports too large a residual, never too small), so it is not polished.
    _up_w, down_w = band(winner, polish=False)
    upper_winner = float(finite[winner].aicc) + 2.0 * down_w
    tied = []
    for name in finite:  # ladder order (dict insertion order)
        if name == winner:
            tied.append(name)
            continue
        # Polishing can only widen ``up`` (it lowers the attainable residual),
        # so it is run only for a candidate the data band alone would exclude.
        up_j, _down_j = band(name, polish=False)
        lower_j = float(finite[name].aicc) - 2.0 * up_j
        if lower_j > upper_winner:
            up_j, _down_j = band(name, polish=True)
            lower_j = float(finite[name].aicc) - 2.0 * up_j
        if not (lower_j > upper_winner):
            tied.append(name)
    if len(tied) == 1:
        return winner, tuple(tied)
    if "M0" in tied:
        return "M0", tuple(tied)
    if "M0" not in finite:
        # M0 was never a usable candidate (failed, saturated or infinite
        # AICc), so the data were not shown to reject the single exponential
        # every candidate nests. Parsimony without that anchor: fewest
        # parameters, ladder order on equal count -- deterministic, unlike
        # the plain argmin of a tie. Pinned by
        # tests/test_pr121_review_round17.py (a saturated M0 must not cost the
        # layer its rate).
        simplest = min(tied, key=lambda n: (int(finite[n].params.size), tied.index(n)))
        return simplest, tuple(tied)
    return MODEL_SELECTION_INDETERMINATE, tuple(tied)


def _finite_abs_max(values: np.ndarray) -> float:
    """``max|v|`` over the FINITE samples (0.0 if there are none). A
    non-finite sample is not a rounding error; the fits deal with it."""
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _curve_data_error(
    values: np.ndarray,
    *,
    d: int,
    formation_scale: float,
    measured_defects: tuple[float, ...] = (),
) -> float:
    """Absolute per-sample rounding level of a curve formed from ``d x d``
    states: ``gamma_{d^2}`` times the magnitude of the quantities the samples
    were formed from (``formation_scale``), raised to any measured defect of
    the same arithmetic (trajectory trace / Hermiticity defect) and to the
    representation level ``u max|y|`` of the values themselves. A non-finite
    defect or scale means the level is unknown and returns ``inf``, which
    makes every candidate "not excludable" -- fail toward M0 / INDETERMINATE,
    never toward a confident winner."""
    u = 0.5 * float(np.finfo(float).eps)
    candidates = [
        _gamma_rounding(d * d) * float(formation_scale),
        u * _finite_abs_max(values),
    ]
    candidates.extend(float(v) for v in measured_defects)
    if not all(np.isfinite(c) for c in candidates):
        return float("inf")
    return max(candidates)


def _dominant_rate(
    t: np.ndarray, curve: np.ndarray, *, data_error: float | None = None
) -> tuple[float, str]:
    """Fit the M0..M3b hierarchy (AICc, no bootstrap) to a decay ``curve`` and
    return ``(dominant_rate, winning_model)``.

    Used to obtain a LINEAR-metric relaxation rate (from the trace-distance
    curve) for the D17 gap-rate consistency check. Unlike relative entropy,
    a linear distance metric decays at the bare mode rate ``Delta`` (no metric
    multiplier), so its dominant rate is dimension-coherent with the spectral
    gap. Returns ``(nan, "none")`` if every model fit fails -- fail-closed, so
    the consistency check reads "unknown" rather than a spurious match.

    ``data_error`` is the absolute per-sample rounding level of ``curve`` (see
    :func:`_select_model`); omitted, only the representation level
    ``u max|curve|`` is assumed, the smallest level any float64 curve has.
    ``(nan, MODEL_SELECTION_INDETERMINATE)`` when the ranking is not resolved
    above it and no nested model is among the tied candidates.
    """
    curve = np.asarray(curve, dtype=float)
    fits: dict[str, FitResult] = {}
    for name in ("M0", "M1", "M2", "M3a", "M3b"):
        try:
            fit_result, _ = _fit_with_model(name, t, curve)
            fits[name] = fit_result
        except (ValueError, RuntimeError):
            continue
    if not fits:
        return float("nan"), "none"
    # Round-17 review: ``choose_model`` falls back to "M0" when nothing has
    # a finite AICc, and the old ``next(iter(fits))`` then handed back an
    # arbitrary FAILED fit's rate. Every candidate saturated == the rate is
    # unknown, which D17 already knows how to read.
    # Round-17 review, CORRECTED after the full suite. The predicate is
    # ``success``, NOT ``isfinite(aicc)``: a non-finite AICc is not the same
    # event as a failed fit. Measured on validation system V4 (thermal
    # two-level), trace-distance curve: all five models converge, yet every
    # AICc is ``inf`` because the Geyer-corrected ``n_eff`` of that smooth,
    # strongly autocorrelated residual series is too small for the
    # small-sample correction. Keying on finiteness withheld D17 on a system
    # where nothing had failed. ``choose_model`` keeps its documented "all
    # entries inf -> M0" fallback; the only thing enforced here is that the
    # winner must come from the SUCCEEDED set. The all-inf fallback that named
    # a model outside it is kept inside, in ladder order (``_select_model``).
    if data_error is None:
        data_error = _curve_data_error(curve, d=1, formation_scale=0.0)
    winner, _tied = _select_model(fits, t, curve, data_error)
    if winner not in fits:  # "none" or MODEL_SELECTION_INDETERMINATE
        return float("nan"), winner
    return _beta_from_params(winner, fits[winner].params), winner


def _beta_index(model_name: str, params: np.ndarray) -> int:
    """Index in ``params`` of the value that :func:`_beta_from_params` returns.

    The BCa CI is taken on ``cis[beta_index]`` and MUST describe the same
    parameter as the point estimate ``beta_D``. For M2 (stretched
    bi-exponential) ``beta_D = min(beta1, beta2)``, so the index depends on the
    fitted params -- not a fixed constant. Previously M2 was hardcoded to index
    1 (beta1), so when beta2 < beta1 the reported CI described a different
    parameter than the point estimate.
    """
    if model_name == "M3a":
        return 2
    if model_name == "M2":
        return 1 if float(params[1]) <= float(params[3]) else 3
    # M0, M1, M3b (and fallback): beta sits at index 1
    return 1


def compute_relaxation_layer(
    L_super: np.ndarray,
    *,
    rho_initial: np.ndarray | None = None,
    rho_steady_state: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    gap: float | None = None,
    eigenvalues: np.ndarray | None = None,
    spectrum_resolved: bool | None = None,
    bootstrap_B: int = 200,
    seed: int = 42,
    trajectory_backend: str = BACKEND_AUTO,
) -> RelaxationResult:
    """Run D5-D7b and the M0..M3b fit hierarchy.

    Parameters
    ----------
    gap
        Spectral gap ``Delta`` (D1), used only to scale the DEFAULT time grid
        to the system's own relaxation time when ``t_grid`` is not given (see
        :func:`default_relaxation_grid`). :func:`liouscope.diagnose` forwards
        the gap it has already computed; a direct caller who omits it gets the
        same value recomputed from ``L_super`` via D1, so the default window is
        rate-unit invariant either way. Ignored when ``t_grid`` is supplied.
    eigenvalues
        The spectrum the fast decay scale and the resolution guard read their
        rates from. :func:`liouscope.diagnose` forwards
        ``SpectralResult.eigenvalues`` -- the certified spectrum D1 came from
        -- so this layer never re-solves ``L_super`` with a bare
        ``np.linalg.eigvals`` that can repeat a primary-driver failure the
        spectral layer had already repaired (ROUND-21 REVIEW, external, PR
        #127). A direct caller who omits both ``gap`` and ``eigenvalues`` gets
        both from one spectral-layer call; a caller who supplies ``gap`` alone
        keeps the historical behaviour, a fresh eigensolve of ``L_super``.
    spectrum_resolved
        ``False`` when the spectral layer's zero-mode certificate is
        applicable but unresolved -- the predicate under which it withholds
        D1/D3/D4 (issues #112/#113). The candidate spectrum is then not a
        measurement, so nothing in this layer is derived from it: no fast
        decay scale (the default window is the documented legacy one, as the
        NaN gap already implies), no resolution guard, no
        ``worst_resolved_*`` identity, and no eigensolve of ``L_super`` either,
        which would only reproduce the same rejected spectrum (PR #154 review).
        ``samples_per_fast_efolding`` and the three ``worst_resolved_*``
        fields are NaN in that case. ``None`` (the default) means "not
        supplied": whenever this layer needs a spectrum and the caller has not
        provided one, it runs the spectral layer itself and reads the verdict
        from that -- also for a caller-supplied ``t_grid`` (PR #154 review,
        round 2: the caller-grid path used to skip the certificate and
        launched a bare eigensolve, so on a stiff generator it persisted a
        missed mode at ``1e8`` that the pipeline path had just withheld).
        ``True`` is a caller's explicit assertion that the ``eigenvalues`` it
        passes are trustworthy; passing ``eigenvalues`` alone implies it, and
        ``True`` WITHOUT ``eigenvalues`` is refused with ``ValueError`` (PR
        #154 review, round 3): there is no spectrum the assertion could be
        about, and silently re-solving would be the bare eigensolve this
        contract exists to prevent.

    ``trajectory_backend`` selects how ``rho(t)`` is propagated (issue #162):
    ``"auto"`` (default, deterministic size/cost rule), ``"dense_expm"``
    (reference) or ``"expm_action"``. The backend actually used is recorded in
    :attr:`RelaxationResult.trajectory_backend`, together with the measured
    trace error, Hermiticity defect and minimum eigenvalue of the propagated
    states (``trajectory_max_trace_error``, ``trajectory_max_hermiticity_defect``,
    ``trajectory_min_eigenvalue``).
    """
    L_super = np.asarray(L_super)
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    if rho_steady_state is None:
        rho_steady_state = steady_state(L_super)
    if rho_initial is None:
        rho_initial = np.eye(d, dtype=complex) / d

    # One spectral-layer call serves everything this layer needs from the
    # spectrum: the gap for the default window, the eigenvalues for the fast
    # scale and the resolution guard, and the certificate verdict that says
    # whether any of it may be used. It runs when the caller supplied neither a
    # spectrum nor a verdict (PR #154 review, round 2: a caller-supplied
    # ``t_grid`` used to skip it and the resolution guard then launched a bare
    # eigensolve on a spectrum nobody had certified), or when the default
    # window needs a gap. Recomputing D1 here rather than re-deriving it keeps a
    # direct call as unit-invariant as the full pipeline: D1 carries the
    # certified zero-mode tolerance and the #113 ambiguity rule (an
    # unresolvable slow spectrum yields NaN, not a fast mode), and duplicating
    # any of that would let the two entry points drift apart. The
    # already-computed steady state is passed through so only the spectrum is
    # recomputed.
    if spectrum_resolved is True and eigenvalues is None:
        raise ValueError(
            "spectrum_resolved=True asserts that the supplied eigenvalues are "
            "trustworthy, but no eigenvalues were supplied; pass the certified "
            "spectrum with it, or leave spectrum_resolved=None so the layer "
            "derives the verdict from the spectral layer itself"
        )
    needs_verdict = eigenvalues is None and spectrum_resolved is None
    needs_gap = t_grid is None and gap is None
    if needs_verdict or needs_gap:
        spectral = compute_spectral_layer(L_super, rho_steady_state)
        if gap is None:
            gap = spectral.gap
        # Same predicate ``diagnose`` applies (ROUND-23 review, PR #121): an
        # applicable-but-unresolved certificate makes the candidate spectrum
        # untrustworthy; no certificate is the pre-#112 state.
        cert = spectral.zero_mode_certificate
        resolved_here = cert is None or not (
            bool(cert["applicable"]) and not bool(cert["resolved"])
        )
        if spectrum_resolved is None:
            spectrum_resolved = resolved_here
        if eigenvalues is None and spectrum_resolved:
            eigenvalues = spectral.eigenvalues
    if spectrum_resolved is None:
        # The caller passed eigenvalues without a verdict: their spectrum,
        # their assertion.
        spectrum_resolved = True

    if t_grid is not None:
        t_grid_source = "caller"
    else:
        assert gap is not None  # set above whenever t_grid is None
        # The FAST scale comes from the spectrum, never from a guess: it is
        # ``max(-Re lambda)``, the same eigenvalues the resolution guard reads.
        # ``fastest_decay_rate`` returns NaN when no finite positive rate
        # exists, and ``default_relaxation_grid`` then keeps the uniform window
        # -- fail-closed, so an unusable spectrum degrades to the historical
        # behaviour plus its warning rather than to an invented timescale.
        # An UNRESOLVED spectrum is not consulted at all (PR #154 review): the
        # NaN is the same "no usable timescale" the withheld gap carries.
        fast_rate = (
            fastest_decay_rate(L_super, eigenvalues=eigenvalues)
            if spectrum_resolved
            else float("nan")
        )
        t_grid = default_relaxation_grid(gap, fast_rate=fast_rate)
        if not (np.isfinite(gap) and gap > 0.0):
            t_grid_source = "legacy_fixed"
        elif is_uniform_grid(t_grid):
            t_grid_source = "gap_scaled"
        else:
            t_grid_source = "gap_scaled_multiscale"
    t_grid = np.asarray(t_grid, dtype=float)
    # The grid selects which residual model every M0..M3b fit below ATTEMPTS;
    # what each fit actually whitened with is decided per fit and reported
    # after the hierarchy has run (see ``residual_model`` below).
    grid_residual_model = "ar1" if is_uniform_grid(t_grid) else "car1"

    traj, used_backend = _propagate(
        L_super, rho_initial, np.asarray(t_grid, dtype=float), backend=trajectory_backend
    )
    trace_error, hermiticity_defect, min_eigenvalue = _trajectory_invariants(traj)
    final_rho = traj[-1]
    rel_entropy = np.array(
        [relative_entropy(traj[k], rho_steady_state) for k in range(traj.shape[0])]
    )
    fidelity_curve = np.array(
        [fidelity(traj[k], rho_steady_state) for k in range(traj.shape[0])]
    )
    trace_distance_curve = np.array(
        [trace_distance(traj[k], rho_steady_state) for k in range(traj.shape[0])]
    )

    # Under-resolution disclosure. Applies to the window actually used, so a
    # caller-supplied grid is checked on the same terms as the default.
    # PR #154 review: on an unresolved spectrum the guard has no measured
    # rates to name a missed mode with, so it measures nothing -- NaN for the
    # ratio and for the mode identity, and no warning that would attribute a
    # specific rate and interval to a spectrum the certificate has rejected.
    if spectrum_resolved:
        fast_resolution, worst_rate, blind, blind_start = _resolution_detail(
            L_super, t_grid, eigenvalues=eigenvalues
        )
    else:
        fast_resolution = worst_rate = blind = blind_start = float("nan")
    if np.isfinite(fast_resolution) and fast_resolution < MIN_SAMPLES_PER_FAST_EFOLD:
        # Name the interval from WHERE it is, not from a size comparison: on a
        # two-scale grid the largest late step exceeds the first step while no
        # lead-in exists at all, and the size test mislabelled it as one.
        span_kind = (
            "unsampled lead-in from t=0"
            if blind_start == 0.0 and float(t_grid[0]) > 0.0
            else f"sample interval starting at t={blind_start:.4g}"
        )
        decay_fraction = _decay_fraction_from_resolution(fast_resolution)
        warnings.warn(
            f"Relaxation layer: the mode at rate {worst_rate:.4g} decays by "
            f"{100.0 * decay_fraction:.1f}% across the "
            f"largest gap the grid leaves before it ({span_kind} = "
            f"{blind:.4g}, {fast_resolution:.3g} samples per e-folding), so it "
            "is stepped over rather than measured. The reported rates describe "
            "the dynamics this window DOES resolve; that component is not "
            "identifiable from this grid and the M0..M3b comparison cannot see "
            "it. The default grid resolves the fastest and the slowest mode "
            "(see default_relaxation_grid); supply an explicit t_grid if you "
            "need an intermediate one -- note the rates then describe that "
            "window instead.",
            UnderResolvedTransientWarning,
            stacklevel=2,
        )

    # Fit hierarchy on relative entropy decay (ensures positivity).
    fits: dict[str, FitResult] = {}
    for name in ("M0", "M1", "M2", "M3a", "M3b"):
        try:
            fit_result, _ = _fit_with_model(name, t_grid, rel_entropy)
            fits[name] = fit_result
        except (ValueError, RuntimeError):
            continue

    # Seventeenth review round (external, PR #127). This field used to be
    # derived from the grid GEOMETRY alone, which made it a statement about
    # what SHOULD have happened rather than about what did. ``fit_gls_ar1``
    # falls back to the discrete AR(1) treatment whenever
    # ``estimate_car1_theta`` returns NaN -- degenerate residuals on a
    # non-uniform grid, e.g. an exactly stationary trajectory with
    # ``rho_initial == rho_steady_state`` -- and the exported metadata then
    # claimed a CAR(1) whitening that never took place. Individual fits can
    # also differ from one another, which one label cannot express at all.
    # Derived from the fits themselves, fail-closed: no fit, no claim.
    #
    # ROUND-21 REVIEW (external, PR #127): "no fit" has to mean no SUCCESSFUL
    # fit. ``fit_gls_ar1`` returns ``success=False`` from its flat-curve guard
    # before it selects or applies any residual model, and the entry it returns
    # then carries ``residual_theta_car1 = NaN`` like a genuine CAR(1)
    # fallback would. Measured on ``rho_initial == rho_steady_state`` over the
    # two-scale grid: five unsuccessful entries, ``whitened_car1`` non-empty
    # and all-False, label ``"car1_fallback_ar1"`` -- persisted audit metadata
    # asserting an AR(1) whitening that never ran. Availability is therefore
    # read off ``success`` FIRST, and the family is classified only among the
    # fits that actually whitened something. The uniform-grid label gets the
    # same treatment: ``"ar1"`` is a statement about what the fits did, not
    # about the grid.
    successful = [fr for fr in fits.values() if fr.success]
    if grid_residual_model == "ar1":
        residual_model = "ar1" if successful else "ar1_unavailable"
    else:
        whitened_car1 = [np.isfinite(fr.residual_theta_car1) for fr in successful]
        if not whitened_car1:
            residual_model = "car1_unavailable"
        elif all(whitened_car1):
            residual_model = "car1"
        elif any(whitened_car1):
            residual_model = "car1_mixed"
        else:
            residual_model = "car1_fallback_ar1"

    # Round-17 review: only fits that did NOT end on the magnitude plateau
    # are candidates. With none left the winner is the "none" sentinel that
    # ``_dominant_rate`` and ``linear_fit_model`` already use -- beta_D then
    # falls through to NaN below instead of being read off a failed fit, and
    # the A-class rungs that compare ``aicc_model`` simply do not fire.
    # Round-17 review, CORRECTED after the full suite. The predicate is
    # ``success``, NOT ``isfinite(aicc)``: a non-finite AICc is not the same
    # event as a failed fit. Measured on validation system V4 (thermal
    # two-level), trace-distance curve: all five models converge, yet every
    # AICc is ``inf`` because the Geyer-corrected ``n_eff`` of that smooth,
    # strongly autocorrelated residual series is too small for the
    # small-sample correction. Keying on finiteness withheld D17 on a system
    # where nothing had failed. ``choose_model`` keeps its documented "all
    # entries inf -> M0" fallback; the only thing enforced here is that the
    # winner must come from the SUCCEEDED set.
    #
    # Cross-family review (PR #127, LIO-02/LIO-03): the ranking must also be
    # resolved above the rounding level of the curve, else ``_select_model``
    # falls back to the nested M0 or reports INDETERMINATE (rate NaN,
    # interval "unavailable"). The two curves get different levels:
    #
    # * trace distance -- formed from O(1) state entries (trace-norm scale 1),
    #   so the level is absolute: gamma_{d^2} * 1, raised to the measured
    #   trace / Hermiticity defect of the same trajectory;
    # * relative entropy -- gamma_{d^2} * max|S| only. This is a LOWER
    #   estimate (the logarithm's amplification near a rank-deficient pi is
    #   not included), so on this curve the policy can miss a tie but never
    #   invents one. Measured on V1-V5, the reference and phantom systems and
    #   ``_chain(1)``: every relative-entropy residual is >= 1e10 times this
    #   level, so no relative-entropy selection there changes.
    d_state = int(traj.shape[1])
    rel_entropy_error = _curve_data_error(
        rel_entropy, d=d_state, formation_scale=_finite_abs_max(rel_entropy)
    )
    trace_distance_error = _curve_data_error(
        trace_distance_curve,
        d=d_state,
        formation_scale=1.0,
        measured_defects=(float(trace_error), float(hermiticity_defect)),
    )
    winner, _tied = _select_model(fits, t_grid, rel_entropy, rel_entropy_error)

    # Bootstrap on the winning model for beta_D
    beta_D = _beta_from_params(winner, fits[winner].params) if winner in fits else float("nan")
    bca_lo, bca_hi = beta_D, beta_D
    jk = None
    if winner in fits and np.isfinite(beta_D):
        winner_fn = {"M0": M0, "M1": M1, "M2": M2, "M3a": M3a, "M3b": M3b}[winner]
        try:
            samples, theta_hat = parametric_bootstrap(
                winner_fn, t_grid, rel_entropy, fits[winner].params,
                B=bootstrap_B, rng=np.random.default_rng(seed),
            )
            # Issue #116: on grids longer than JACKKNIFE_MAX_GRID_POINTS (the
            # default 80-point grid included) no acceleration is estimated and
            # bca_ci returns the BC interval; ``interval_method`` says so.
            if t_grid.size <= JACKKNIFE_MAX_GRID_POINTS:
                # The refits keep the family of the fit they estimate the
                # acceleration for, also where a deletion changes the grid's
                # uniformity (PR #127 review).
                jk = _jackknife(
                    winner_fn, t_grid, rel_entropy, theta_hat, None,
                    residual_family=(
                        "car1"
                        if np.isfinite(fits[winner].residual_theta_car1)
                        else "ar1"
                    ),
                )
            cis = bca_ci(samples, theta_hat, jackknife_estimates=jk)
            beta_idx = _beta_index(winner, fits[winner].params)
            bca_lo, bca_hi = float(cis[beta_idx, 0]), float(cis[beta_idx, 1])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # Fail loud, not certain: a swallowed bootstrap failure used to
            # leave the degenerate CI (beta_D, beta_D), which the uncertainty
            # layer reads as fit_uncertainty == 0.0 — *perfect* confidence as
            # the failure mode of an uncertainty pipeline. NaN propagates as
            # "unknown" instead.
            warnings.warn(
                f"parametric bootstrap for beta_D failed ({exc!r}); "
                "bca_ci_beta set to (nan, nan) — fit uncertainty is UNKNOWN, "
                "not zero.",
                RuntimeWarning,
            )
            bca_lo, bca_hi = float("nan"), float("nan")

    # Issue #116: name the estimator that actually produced the interval. A
    # non-finite endpoint covers every no-interval outcome (bootstrap failure,
    # non-finite point estimate, degenerate bootstrap distribution, #125), so
    # an interval is never credited to an estimator that produced no interval.
    if not (np.isfinite(bca_lo) and np.isfinite(bca_hi)):
        interval_method = INTERVAL_METHOD_UNAVAILABLE
    elif jk is not None:
        interval_method = INTERVAL_METHOD_BCA_DELETE1
    else:
        interval_method = INTERVAL_METHOD_BC

    try:
        ent_asym = entanglement_asymmetry(final_rho)
    except Exception:
        ent_asym = float("nan")

    # LINEAR-metric relaxation rate for D17 (LIOU-#69). ``beta_D`` above is fit
    # on the RELATIVE-ENTROPY curve, which decays at a metric multiplier m times
    # the bare mode rate (m=2 for a faithful/full-rank steady state where D is
    # quadratic near pi; m=1 for a rank-deficient pi where the null-space-leakage
    # term is linear). That multiplier makes beta_D dimensionally incomparable to
    # the spectral gap Delta. The trace-distance curve is a linear distance
    # metric, so its dominant rate decays at the bare mode rate and IS
    # dimension-coherent with Delta -- this is what the gap-rate consistency
    # check (D17) must use. (_dominant_rate is fail-closed: it returns nan if
    # every model fit fails, so D17 then reads "unknown" rather than a spurious
    # match.)
    beta_D_linear, linear_fit_model = _dominant_rate(
        t_grid, trace_distance_curve, data_error=trace_distance_error
    )

    return RelaxationResult(
        von_neumann_entropy=von_neumann_entropy(final_rho),
        relative_entropy_curve=rel_entropy,
        fidelity_curve=fidelity_curve,
        trace_distance_curve=trace_distance_curve,
        entanglement_asymmetry=None if np.isnan(ent_asym) else ent_asym,
        fits=fits,
        aicc_model=winner,
        beta_D=float(beta_D),
        bca_ci_beta=(bca_lo, bca_hi),
        beta_D_linear=float(beta_D_linear),
        linear_fit_model=linear_fit_model,
        t_grid_source=t_grid_source,
        t_grid_span=float(t_grid[-1] - t_grid[0]),
        t_grid=t_grid.copy(),
        samples_per_fast_efolding=fast_resolution,
        # ROUND-21 REVIEW (external, PR #127): the scalar above is the MINIMUM
        # over modes, so on a grid with three or more separated timescales it
        # can describe an intermediate mode, not the fastest one. Which mode and
        # which interval it means used to live only in the warning text; once
        # the warning stream is gone the persisted report could not say. The
        # three values the warning is built from now travel with the number.
        worst_resolved_rate=float(worst_rate),
        worst_resolved_blind_interval=float(blind),
        worst_resolved_blind_start=float(blind_start),
        residual_model=residual_model,
        interval_method=interval_method,
        trajectory_backend=used_backend,
        trajectory_max_trace_error=trace_error,
        trajectory_max_hermiticity_defect=hermiticity_defect,
        trajectory_min_eigenvalue=min_eigenvalue,
    )
