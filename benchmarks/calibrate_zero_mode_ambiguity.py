"""Reproducible calibration artefact for the issue #113 ambiguity split.

Why this file exists
--------------------
``ZERO_MODE_AMBIGUITY_FACTOR`` (``src/liouscope/_consts.py``) is the only
constant in this package whose comment quotes distributions of BOTH
populations it has to separate ("83 healthy generators ... 2.38 max, 0.39
median; unresolved 4.87 and 4.87e2"). Those numbers were prose: the
population behind them was never committed, so nobody could re-derive them,
check them on another BLAS, or notice when they drift. A threshold justified
by an unreproducible sample is exactly the failure this repository criticises
elsewhere (issue #112 review: "calibrated from six fixtures").

This script ENUMERATES the population, measures it, and writes the result to
JSON. It sets no threshold and reads none for its verdict -- it produces the
evidence at which a threshold can be set.

The measured quantity
---------------------
For each generator ``L`` (a ``d^2 x d^2`` GKSL superoperator)::

    level = eps * ||L||_2                 # eigensolver backward error (#108)
    band  = { |lambda| : |lambda| <= certificate.bound }
    ratio = max(band) / level

``ratio`` is dimensionless and invariant under a rate rescale ``L -> cL`` in
exact arithmetic, which is why the sweep repeats every family at several
scales: any scale dependence in the measured value is round-off, and that is
precisely what the calibration must bound.

Population definition (declared BY CONSTRUCTION, not by outcome)
----------------------------------------------------------------
Sorting generators by their measured ``ratio`` and then calling the small ones
"healthy" would be circular. Membership is therefore fixed by how the
generator is BUILT:

* **healthy** -- every Lindblad rate lies within ``HEALTHY_RATE_SPREAD_MAX``
  (1e4) of every other one, and the Hamiltonian norm is of the same order.
  No engineered stiffness. Steady state unique or degenerate alike.
* **stiff** -- the measured issue #112 repro: a four-level classical jump
  network carrying slow rates ``~1e-5`` next to one deliberately fast rate.
  The rate spread is ``>= 1e9`` by construction, i.e. these generators are
  members because of how they were made, before anything is measured.

Usage
-----
``python benchmarks/calibrate_zero_mode_ambiguity.py``
    writes ``benchmarks/output/zero_mode_calibration.json``

``python benchmarks/calibrate_zero_mode_ambiguity.py --out other.json --seed 7``
    same sweep with a different RNG seed for the random-GKSL family.

To check that the committed artefact is still the one this code produces::

    python benchmarks/calibrate_zero_mode_ambiguity.py
    git diff --stat benchmarks/calibration/zero_mode_calibration.json

The structural part of that agreement (population sizes, family list, defect
verdicts) is asserted in ``tests/test_threshold_calibration.py``; the round-off
digits deliberately are NOT, see below.

The JSON records the environment (Python / NumPy / SciPy / platform / BLAS)
because the quantity being measured IS round-off; a number from another BLAS
is a different measurement, not a contradiction. That is also why a test must
not pin the digits: it would turn a legitimate platform difference into a red
build and teach the next reader to widen the tolerance until it means nothing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.nonnormality import henrici_eta_n, henrici_relative
from liouscope.diagnostics.spectral import liouvillian_gap
from liouscope.diagnostics.transient import kappa_trans
from liouscope.numerics.linalg import certified_eigvals

#: A family is "healthy" only if its rates stay inside this spread. Declared
#: here so the population definition is a constructor argument, never a
#: post-hoc reading of the measurement.
HEALTHY_RATE_SPREAD_MAX: float = 1.0e4

#: A stiff generator counts as DEFECTIVE when its reported gap misses the
#: closed-form gap by more than this, i.e. by an oracle independent of the
#: ambiguity split being calibrated.
GAP_DEFECT_TOL: float = 1.0e-6

#: Rate rescales applied to every healthy family. ``ratio`` is scale invariant
#: in exact arithmetic, so these probe round-off, not physics (issue #108 used
#: the same three).
SCALES: tuple[float, ...] = (1.0, 1.0e6, 1.0e12)

#: The issue #112 repro. Slow rates fixed; the fast rate is swept.
STIFF_PAIRS: tuple[tuple[int, int], ...] = ((0, 3), (0, 2), (1, 0), (3, 2), (2, 1))
STIFF_SLOW_RATES: tuple[float, ...] = (7.28e-6, 3.67e-5, 1.53e-5, 1.42e-5)
STIFF_FAST_RATES: tuple[float, ...] = (2.7e5, 1e6, 1e7, 1e8, 1e9, 1e10)

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_ID2 = np.eye(2, dtype=complex)


@dataclass(frozen=True, slots=True)
class Generator:
    """One population member: the matrix plus how it was constructed."""

    family: str
    label: str
    population: str
    dim: int
    scale: float
    rate_spread: float
    L: np.ndarray = field(repr=False)
    #: Independently known gap, or NaN when no closed form is available.
    true_gap: float = float("nan")


@dataclass(frozen=True, slots=True)
class Measurement:
    """Per-generator record. Every field is measured, none is a verdict."""

    family: str
    label: str
    population: str
    dim: int
    scale: float
    rate_spread: float
    solver: str
    certified: bool
    applicable: bool
    zero_mode_count: int
    ambiguous_count: int
    resolved: bool
    backward_error: float
    bound: float
    max_in_band: float
    ratio: float
    gap: float
    true_gap: float
    #: |gap - true_gap| / true_gap. This is the INDEPENDENT defect oracle: a
    #: stiff generator is defective because its reported gap is wrong, never
    #: because its ratio landed on one side of the split under test.
    gap_rel_error: float
    henrici_eta: float
    henrici_relative: float
    kappa_trans: float


# --------------------------------------------------------------------------
# Families. Each returns generators whose rates are inside
# HEALTHY_RATE_SPREAD_MAX by construction; the spread is returned, not assumed.
# --------------------------------------------------------------------------


def _spread(rates: list[float]) -> float:
    positive = [r for r in rates if r > 0.0]
    if not positive:
        return 1.0
    return float(max(positive) / min(positive))


def _amplitude_damping() -> Iterator[tuple[str, str, np.ndarray, float]]:
    for gamma in (0.1, 0.5, 1.0, 2.0):
        L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_SM], [gamma])
        yield "amp_damping", f"gamma={gamma:g}", L, _spread([gamma])


def _dephasing() -> Iterator[tuple[str, str, np.ndarray, float]]:
    for gamma in (0.05, 0.5, 3.0):
        L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_SZ], [gamma])
        yield "dephasing", f"gamma={gamma:g}", L, _spread([gamma])


def _rabi_damping() -> Iterator[tuple[str, str, np.ndarray, float]]:
    for omega in (0.5, 2.0):
        for gamma in (0.1, 1.0):
            L = build_liouvillian(0.5 * omega * _SX, [_SM], [gamma])
            yield (
                "rabi_damping",
                f"omega={omega:g},gamma={gamma:g}",
                L,
                _spread([gamma]),
            )


def _thermal_two_level() -> Iterator[tuple[str, str, np.ndarray, float]]:
    for n_bar in (0.1, 1.0, 3.0):
        down, up = 1.0 + n_bar, n_bar
        L = build_liouvillian(0.5 * _SZ, [_SM, _SM.conj().T], [down, up])
        yield "thermal", f"n_bar={n_bar:g}", L, _spread([down, up])


def _two_qubit_dephasing_degenerate() -> Iterator[tuple[str, str, np.ndarray, float]]:
    """Conserved magnetisation: a genuinely DEGENERATE stationary manifold."""
    for g in (0.25, 0.5, 2.0):
        jumps = [np.kron(_SZ, _ID2), np.kron(_ID2, _SZ)]
        L = build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, [g, g])
        yield "two_qubit_dephasing_degenerate", f"g={g:g}", L, _spread([g, g])


def _two_qubit_damped_chain() -> Iterator[tuple[str, str, np.ndarray, float]]:
    """Unique steady state, exchange coupling, both sites damped."""
    hop = np.kron(_SM, _SM.conj().T)
    for j_coupling in (0.5, 1.5):
        for gamma in (0.2, 1.0):
            H = j_coupling * (hop + hop.conj().T)
            jumps = [np.kron(_SM, _ID2), np.kron(_ID2, _SM)]
            L = build_liouvillian(H, jumps, [gamma, 0.5 * gamma])
            yield (
                "two_qubit_damped_chain",
                f"J={j_coupling:g},gamma={gamma:g}",
                L,
                _spread([gamma, 0.5 * gamma]),
            )


def _qutrit_ladder() -> Iterator[tuple[str, str, np.ndarray, float]]:
    for gamma in (0.3, 1.0):
        j21 = np.zeros((3, 3), dtype=complex)
        j21[0, 1] = 1.0
        j32 = np.zeros((3, 3), dtype=complex)
        j32[1, 2] = 1.0
        H = np.diag([0.0, 1.0, 2.2]).astype(complex)
        L = build_liouvillian(H, [j21, j32], [gamma, 2.0 * gamma])
        yield "qutrit_ladder", f"gamma={gamma:g}", L, _spread([gamma, 2.0 * gamma])


def _random_gksl(
    rng: np.random.Generator, draws: int
) -> Iterator[tuple[str, str, np.ndarray, float]]:
    """Random but SEEDED GKSL generators; rates bounded to keep the family healthy."""
    for dim in (2, 3, 4):
        for draw in range(draws):
            H_raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            H = 0.5 * (H_raw + H_raw.conj().T)
            n_jumps = 2
            jumps = [
                rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
                for _ in range(n_jumps)
            ]
            # Rates inside one decade -> spread 10 << HEALTHY_RATE_SPREAD_MAX.
            rates = [float(r) for r in rng.uniform(0.1, 1.0, size=n_jumps)]
            L = build_liouvillian(H, jumps, rates)
            yield "random_gksl", f"d={dim},draw={draw}", L, _spread(rates)


def _healthy_generators(seed: int, draws: int = 3) -> list[Generator]:
    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    rng = np.random.default_rng(seed)
    base: list[tuple[str, str, np.ndarray, float]] = []
    base.extend(_amplitude_damping())
    base.extend(_dephasing())
    base.extend(_rabi_damping())
    base.extend(_thermal_two_level())
    base.extend(_two_qubit_dephasing_degenerate())
    base.extend(_two_qubit_damped_chain())
    base.extend(_qutrit_ladder())
    base.extend(_random_gksl(rng, draws))

    out: list[Generator] = []
    for family, label, L, spread in base:
        if spread > HEALTHY_RATE_SPREAD_MAX:
            raise AssertionError(
                f"{family}/{label} declares rate spread {spread:g}, above the "
                f"healthy ceiling {HEALTHY_RATE_SPREAD_MAX:g}: it does not "
                "belong in the healthy population"
            )
        for scale in SCALES:
            dim = int(round(np.sqrt(L.shape[0])))
            out.append(
                Generator(
                    family=family,
                    label=f"{label}@c={scale:g}",
                    population="healthy",
                    dim=dim,
                    scale=scale,
                    rate_spread=spread,
                    L=scale * L,
                )
            )
    return out


def _classical_network(rates: list[float]) -> np.ndarray:
    jumps = []
    for to, frm in STIFF_PAIRS:
        j = np.zeros((4, 4), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, rates)


def _analytic_gap(rates: list[float]) -> float:
    """Closed-form gap of the classical jump network -- the INDEPENDENT oracle.

    With ``H = 0`` the generator block-decouples: populations follow the ``4x4``
    Markov generator ``K``, coherences are exactly diagonal with rates
    ``-0.5*(Gamma_i + Gamma_j)``, ``i != j``. The gap is therefore known without
    asking the eigensolver under test, which is the whole point -- a defect
    verdict must not be derived from the quantity being calibrated.
    """
    k = np.zeros((4, 4))
    for (to, frm), g in zip(STIFF_PAIRS, rates):
        k[to, frm] += g
        k[frm, frm] -= g
    gamma = -np.diag(k)
    coherences = [
        -0.5 * (gamma[i] + gamma[j]) for i in range(4) for j in range(4) if i != j
    ]
    # The zero mode is removed STRUCTURALLY, not by a relative magnitude filter.
    # STIFF_PAIRS is irreducible, so K has exactly one zero eigenvalue; dropping
    # the single smallest-modulus one is exact. A relative cutoff would be the
    # very failure this artefact exists to expose -- at fast=1e8 the cutoff
    # ``1e-12 * radius`` is 5e-5 and would swallow the true 1.07e-5 gap,
    # producing a fake "no defect" verdict (observed while building this file).
    populations = np.linalg.eigvals(k)
    order = np.argsort(np.abs(populations))
    populations = populations[order[1:]]
    spectrum = np.concatenate([populations, np.array(coherences, dtype=complex)])
    return float(min(-e.real for e in spectrum))


def _stiff_generators() -> list[Generator]:
    """The counter-population: engineered rate spread, slow modes at risk."""
    out: list[Generator] = []
    slow = list(STIFF_SLOW_RATES)
    for fast in STIFF_FAST_RATES:
        rates = [slow[0], slow[1], slow[2], fast, slow[3]]
        spread = _spread(rates)
        if spread < 1.0e9:
            raise AssertionError(
                f"stiff fast={fast:g} has spread {spread:g}: not stiff by "
                "construction, so it must not enter the stiff population"
            )
        out.append(
            Generator(
                family="classical_jump_network",
                label=f"fast={fast:g}",
                population="stiff",
                dim=4,
                scale=1.0,
                rate_spread=spread,
                L=_classical_network(rates),
                true_gap=_analytic_gap(rates),
            )
        )
    return out


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def measure(gen: Generator) -> Measurement:
    """Measure one generator. No thresholds are applied to the result."""
    ev, cert = certified_eigvals(gen.L)
    eps = float(np.finfo(float).eps)
    level = eps * float(np.linalg.norm(gen.L, 2))
    magnitudes = np.abs(ev)
    band = magnitudes[magnitudes <= cert.bound]
    max_in_band = float(band.max()) if band.size else float("nan")
    ratio = max_in_band / level if level > 0.0 else float("nan")
    gap = liouvillian_gap(ev)
    if np.isfinite(gen.true_gap) and gen.true_gap > 0.0 and np.isfinite(gap):
        gap_rel_error = abs(gap - gen.true_gap) / gen.true_gap
    else:
        gap_rel_error = float("nan")
    hermitian_part = 0.5 * (gen.L + gen.L.conj().T)
    abscissa = float(np.max(np.linalg.eigvalsh(hermitian_part)))
    return Measurement(
        family=gen.family,
        label=gen.label,
        population=gen.population,
        dim=gen.dim,
        scale=gen.scale,
        rate_spread=gen.rate_spread,
        solver=cert.solver,
        certified=bool(cert.certified),
        applicable=bool(cert.applicable),
        zero_mode_count=int(cert.zero_mode_count),
        ambiguous_count=int(cert.ambiguous_count),
        resolved=bool(cert.resolved),
        backward_error=level,
        bound=float(cert.bound),
        max_in_band=max_in_band,
        ratio=float(ratio),
        gap=float(gap),
        true_gap=float(gen.true_gap),
        gap_rel_error=float(gap_rel_error),
        henrici_eta=float(henrici_eta_n(gen.L)),
        henrici_relative=float(henrici_relative(gen.L)),
        kappa_trans=float(kappa_trans(abscissa, gap)),
    )


def summarise(values: list[float]) -> dict[str, Any]:
    """Percentile summary. Refuses an empty sample instead of emitting NaN."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        raise ValueError(
            "summarise: no finite values -- an empty or all-NaN sample is not a "
            "distribution and must not be summarised into a threshold"
        )
    arr = np.asarray(finite, dtype=float)
    qs = np.percentile(arr, [50, 75, 90, 95, 99])
    return {
        "n": int(arr.size),
        "n_non_finite_dropped": int(len(values) - arr.size),
        "min": float(arr.min()),
        "median": float(qs[0]),
        "p75": float(qs[1]),
        "p90": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
        "max": float(arr.max()),
    }


def _jsonable(value: float) -> float | None:
    """RFC 8259 has no NaN/Infinity literal (same rule as ZeroModeCertificate)."""
    return float(value) if np.isfinite(value) else None


def _tail_stability(seed: int) -> list[dict[str, Any]]:
    """How much do the summary statistics move with sample size and seed?

    A MAXIMUM over a finite sample is an order statistic, not a population
    bound: it grows with n and swings with the seed. Reporting it without this
    block is how "2.38" became a fact. The percentiles are reported alongside
    so a reader can see which statistic is actually stable.
    """
    out: list[dict[str, Any]] = []
    for draws in (3, 30):
        for offset in (0, 1, 2):
            rows = [measure(g) for g in _healthy_generators(seed + offset, draws)]
            stats = summarise([r.ratio for r in rows])
            out.append({"seed": seed + offset, "draws": draws, **stats})
    return out


def _metric_summary(rows: list[Measurement]) -> dict[str, Any]:
    """Distributions of the other P0-gated metrics on the SAME population.

    ``henrici_eta`` (classification.py, F5 leg: ``> 1.0``) is rate-dimensioned;
    ``henrici_relative`` and ``kappa_trans`` (F2 leg: ``> 2.0``) are
    dimensionless. Splitting by rate scale therefore shows directly whether a
    gate measures physics or the caller's choice of time unit.
    """
    by_scale: dict[str, Any] = {}
    for scale in SCALES:
        sel = [r for r in rows if r.scale == scale]
        if not sel:
            continue
        by_scale[f"c={scale:g}"] = {
            "n": len(sel),
            "henrici_eta": summarise([r.henrici_eta for r in sel]),
            "henrici_relative": summarise([r.henrici_relative for r in sel]),
            "kappa_trans": summarise([r.kappa_trans for r in sel]),
            "henrici_eta_above_1": sum(1 for r in sel if r.henrici_eta > 1.0),
            "kappa_trans_above_2": sum(1 for r in sel if r.kappa_trans > 2.0),
        }
    return by_scale


def run(seed: int, draws: int = 3) -> dict[str, Any]:
    """Enumerate, measure and summarise both populations."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    healthy = _healthy_generators(seed, draws)
    stiff = _stiff_generators()
    if not healthy or not stiff:
        raise AssertionError("both populations must be non-empty to calibrate a split")

    rows = [measure(g) for g in healthy] + [measure(g) for g in stiff]
    healthy_rows = [r for r in rows if r.population == "healthy"]
    stiff_rows = [r for r in rows if r.population == "stiff"]

    uncertified = [r.label for r in healthy_rows if not r.certified]
    if uncertified:
        raise AssertionError(
            "healthy population contains uncertified solves, so the sample is "
            f"not a healthy-generator sample: {uncertified}"
        )

    healthy_ratios = [r.ratio for r in healthy_rows]
    stiff_ratios = [r.ratio for r in stiff_rows]

    # Defect is decided by the ANALYTIC gap, never by the split under test.
    defective = [
        r for r in stiff_rows
        if np.isfinite(r.gap_rel_error) and r.gap_rel_error > GAP_DEFECT_TOL
    ]
    tail = _tail_stability(seed)
    # Compare against the LARGEST healthy ratio anywhere in the tail sweep, not
    # just this sample's: the max is an order statistic and grows with n, so
    # using the local one would restate the very over-claim being corrected.
    healthy_max = max([max(healthy_ratios), *(t["max"] for t in tail)])
    separation: dict[str, Any] = {
        "gap_defect_tolerance": GAP_DEFECT_TOL,
        "healthy_max_ratio": healthy_max,
        "healthy_max_ratio_this_sample": max(healthy_ratios),
        "healthy_max_ratio_source": (
            "largest value over this sample AND the tail-stability sweep"
        ),
        "n_stiff_by_construction": len(stiff_rows),
        "n_stiff_actually_defective": len(defective),
        "defective_cases": [
            {
                "label": r.label,
                "ratio": _jsonable(r.ratio),
                "rate_spread": r.rate_spread,
                "gap_rel_error": _jsonable(r.gap_rel_error),
                "flagged_by_ambiguity_check": not r.resolved,
            }
            for r in sorted(defective, key=lambda r: r.ratio)
        ],
    }
    if defective:
        lowest = min(r.ratio for r in defective)
        separation["lowest_defective_ratio"] = lowest
        separation["separation_factor_healthy_max_to_lowest_defect"] = (
            lowest / healthy_max if healthy_max > 0.0 else float("inf")
        )
        separation["defects_missed_by_current_split"] = [
            r.label for r in defective if r.resolved
        ]
    return {
        "artefact": "zero_mode_ambiguity_calibration",
        "artefact_version": 1,
        "issue": "#113",
        "quantity": "max{|lambda| : |lambda| <= certificate.bound} / (eps * ||L||_2)",
        "seed": seed,
        "random_gksl_draws_per_dim": draws,
        "population_rule": {
            "healthy": (
                "constructed with all Lindblad rates within "
                f"{HEALTHY_RATE_SPREAD_MAX:g}x of each other; membership is a "
                "property of the construction, never of the measured ratio"
            ),
            "stiff": (
                "issue #112 four-level classical jump network, slow rates ~1e-5 "
                "next to one swept fast rate; rate spread >= 1e9 by construction"
            ),
            "scales": list(SCALES),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "eps": float(np.finfo(float).eps),
        },
        "summary": {
            "healthy": summarise(healthy_ratios),
            "stiff": summarise(stiff_ratios),
        },
        "separation": separation,
        "tail_stability": tail,
        "other_p0_metrics_by_rate_scale": _metric_summary(healthy_rows),
        "families": sorted({r.family for r in healthy_rows}),
        "measurements": [
            {
                k: (_jsonable(v) if isinstance(v, float) else v)
                for k, v in asdict(r).items()
            }
            for r in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "calibration"
            / "zero_mode_calibration.json"
        ),
        help="destination JSON path (the default is the COMMITTED reference "
        "artefact; benchmarks/output/ is gitignored and would not be checkable)",
    )
    parser.add_argument("--seed", type=int, default=20260816, help="RNG seed (>= 0)")
    parser.add_argument(
        "--draws",
        type=int,
        default=3,
        help="random-GKSL draws per dimension (>= 1). The healthy MAX is a tail "
        "estimate and grows with this number -- see the artefact's tail note.",
    )
    parser.add_argument(
        "--print-only", action="store_true", help="summarise to stdout, write nothing"
    )
    args = parser.parse_args(argv)

    if args.seed < 0:
        parser.error(f"--seed must be non-negative, got {args.seed}")
    if args.draws < 1:
        parser.error(f"--draws must be >= 1, got {args.draws}")

    payload = run(args.seed, args.draws)
    h, s = payload["summary"]["healthy"], payload["summary"]["stiff"]
    print(f"healthy n={h['n']}  median={h['median']:.3g}  p95={h['p95']:.3g}  max={h['max']:.3g}")
    print(f"stiff   n={s['n']}  median={s['median']:.3g}  p95={s['p95']:.3g}  max={s['max']:.3g}")
    sep = payload["separation"]
    print(
        f"defective (analytic-gap oracle): {sep['n_stiff_actually_defective']}"
        f" of {sep['n_stiff_by_construction']} stiff generators"
    )
    if "lowest_defective_ratio" in sep:
        print(
            f"nearest pair: healthy max {sep['healthy_max_ratio']:.3g}  vs  "
            f"lowest defect {sep['lowest_defective_ratio']:.3g}  "
            f"(factor {sep['separation_factor_healthy_max_to_lowest_defect']:.2f}x)"
        )
        print(f"defects missed by the current split: {sep['defects_missed_by_current_split']}")

    if args.print_only:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicitly: the default translates to CRLF on Windows, so the
    # SAME sweep would produce different BYTES per platform and "regenerate, then
    # check `git diff` is empty" -- the check that keeps this artefact honest --
    # would fail on a Windows clone for a reason that has nothing to do with the
    # measurement. .gitattributes normalises the stored blob; this keeps the
    # working tree agreeing with it.
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
