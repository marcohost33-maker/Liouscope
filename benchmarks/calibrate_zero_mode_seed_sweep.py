"""Does the issue #113 calibration depend on the RNG seed?

Why this file exists
--------------------
``benchmarks/calibrate_zero_mode_ambiguity.py`` enumerates the population
behind ``ZERO_MODE_AMBIGUITY_FACTOR`` at ONE seed. Its ``tail_stability`` block
already shows that the healthy MAXIMUM is an order statistic that grows with
the sample size, and it varies the seed by +0/+1/+2 to hint that the maximum
also moves sideways. Three neighbouring seeds are a hint, not a measurement.

The ``_consts.py`` comment nevertheless quotes single numbers -- "max 3.36",
"4.23 over the larger n=339 tail sweep", "the nearest pair is therefore 4.23
against 4.87 -- a factor 1.15". Every one of those is a point estimate of a
random variable, and the largest healthy ratio is produced ENTIRELY by the
``random_gksl`` family, i.e. by exactly the part of the population the seed
controls (checked in the committed artefact: the eight largest healthy ratios
are all ``random_gksl``). So the question is not academic: if the healthy
maximum swings by a factor of two across seeds, then "factor 1.15" is one draw
from a distribution and must be written as one.

This script answers two questions with numbers instead of prose:

1. **Spread.** How far do ``max`` and ``p95`` of the healthy population move
   across independent seeds, at several sample sizes?
2. **Exceedance.** How often does a HEALTHY generator reach the shipped split
   of 30 at all -- and how often does it reach the lowest genuinely defective
   stiff ratio, which is where the two populations would actually overlap?

Question 2 is the one the ``_consts.py`` comment leaves open. A conservative
one-sided threshold is only conservative if the healthy side stays below it;
that is a claim about a tail, and a tail needs a sample.

What is measured, and what is NOT decided here
----------------------------------------------
The measured quantity is unchanged -- ``max{|lambda| : |lambda| <= bound} /
(eps * ||L||_2)`` -- and the population rule is unchanged, because both are
imported from the calibration script rather than restated. This file adds only
the seed axis.

The shipped constant IS read here, but only to COUNT exceedances; no verdict is
derived from it, and no threshold is proposed. Counting how often a healthy
generator crosses the shipped value is a measurement of the shipped value's
adequacy, not a use of it as a classifier.

The stiff population is seed-free BY CONSTRUCTION -- ``_stiff_generators()``
takes no seed and draws no random numbers -- so it is measured once and
recorded under ``stiff_reference``. Re-measuring it per seed would be theatre:
it would call the identical function with identical arguments. The reference it
supplies is ``lowest_defective_ratio``, the value a healthy generator would
have to reach for the two populations to overlap.

Usage
-----
``python benchmarks/calibrate_zero_mode_seed_sweep.py``
    writes ``benchmarks/calibration/zero_mode_seed_sweep.json``

``python benchmarks/calibrate_zero_mode_seed_sweep.py --print-only``
    summarise to stdout, write nothing.

The sweep is fully deterministic: the seed list and the draw counts are module
constants, and NumPy's ``default_rng`` is reproducible, so running it twice must
produce identical bytes. That is checked the same way as the single-seed
artefact (regenerate, then ``git diff --stat``).

On the choice of seeds
----------------------
The values in ``SWEEP_SEEDS`` are arbitrary and declared, which is the point:
``numpy.random.default_rng`` routes any integer entropy through
``SeedSequence``, whose stated purpose is to produce independent streams from
arbitrary (including neighbouring) seeds. So the sweep does not need "good"
seeds, it needs FIXED and DISCLOSED ones -- otherwise a later reader cannot
tell a re-measurement from a re-roll.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from liouscope._consts import ZERO_MODE_AMBIGUITY_FACTOR

#: Fixed, disclosed seeds. See "On the choice of seeds" above. 20260816 is the
#: seed of the committed single-seed artefact and is included so the two
#: artefacts can be cross-read.
SWEEP_SEEDS: tuple[int, ...] = (
    0, 1, 2, 7, 42, 1234, 65535, 20260816, 20260817, 987654321,
)

#: Sample sizes. The healthy population is ``69 + 9 * draws`` generators
#: (23 fixed fixtures x 3 rate scales, plus 3 dimensions x ``draws`` random
#: draws x 3 rate scales). Several levels are swept because the maximum is an
#: order statistic: separating "moved because of the seed" from "moved because
#: of n" needs both axes, not one.
SWEEP_DRAWS: tuple[int, ...] = (3, 30, 100)


def _calibration_module() -> Any:
    """Load the single-seed calibration script (``benchmarks`` is not a package).

    Imported rather than copied on purpose: the families, the healthy/stiff
    membership rule and the measurement must be THE SAME objects the committed
    artefact was produced from, or this sweep would be measuring a fork.
    """
    path = Path(__file__).resolve().parent / "calibrate_zero_mode_ambiguity.py"
    spec = importlib.util.spec_from_file_location("calibrate_zero_mode_ambiguity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the calibration module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: float) -> float | None:
    """RFC 8259 has no NaN/Infinity literal -- same rule as the sibling script.

    ``json.dumps`` happily writes the bare word ``Infinity``, which Python reads
    back without complaint and every standards-compliant parser rejects. That is
    a silent failure: the tests here go green while the artefact is unreadable
    to jq, to a JS consumer, or to any strict re-parse. Found exactly that way
    -- the across-seed spread of the MEDIAN divides by zero at the small sample
    size, where every median is exactly 0.0.

    A ratio with a zero denominator is genuinely undefined, so ``null`` is also
    the honest encoding, not merely the portable one.
    """
    return float(value) if np.isfinite(value) else None


def _percentiles(values: list[float]) -> dict[str, Any]:
    """Percentile summary; refuses an empty sample instead of emitting NaN."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        raise ValueError(
            "no finite ratios -- an empty sample is not a distribution and must "
            "not be summarised into a statement about a threshold"
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


def _stiff_reference(mod: Any) -> dict[str, Any]:
    """Measure the seed-free stiff population once, for the overlap question.

    ``lowest_defective_ratio`` is the number that matters: a healthy generator
    at or above it would mean the two populations OVERLAP, at which point no
    magnitude split -- conservative or not -- can separate them.
    """
    rows = [mod.measure(g) for g in mod._stiff_generators()]
    defective = [
        r for r in rows
        if np.isfinite(r.gap_rel_error) and r.gap_rel_error > mod.GAP_DEFECT_TOL
    ]
    if not defective:
        raise AssertionError(
            "the stiff population no longer contains a generator whose reported "
            "gap is wrong by the analytic oracle: there is nothing to separate "
            "the healthy population FROM, so the sweep has no reference"
        )
    return {
        "ratios": {r.label: _jsonable(r.ratio) for r in rows},
        "defective_labels": sorted(r.label for r in defective),
        "unresolved_slow_mode_ratios": sorted(
            (float(r.ratio) for r in defective if np.isfinite(r.ratio)), reverse=True
        ),
        "lowest_defective_ratio": float(
            min(r.ratio for r in defective if np.isfinite(r.ratio))
        ),
        "flagged_by_shipped_split": sorted(r.label for r in defective if not r.resolved),
        "missed_by_shipped_split": sorted(r.label for r in defective if r.resolved),
    }


def _one_cell(mod: Any, seed: int, draws: int, overlap_level: float) -> dict[str, Any]:
    """One (seed, draws) cell of the sweep."""
    rows = [mod.measure(g) for g in mod._healthy_generators(seed, draws)]
    uncertified = [r.label for r in rows if not r.certified]
    if uncertified:
        raise AssertionError(
            f"seed {seed}, draws {draws}: healthy population contains uncertified "
            f"solves, so it is not a healthy-generator sample: {uncertified}"
        )
    ratios = [r.ratio for r in rows]
    # The seed only moves the random family; splitting it out keeps a
    # seed-independent fixture from masquerading as evidence of seed stability.
    random_rows = [r for r in rows if r.family == "random_gksl"]
    fixed_rows = [r for r in rows if r.family != "random_gksl"]
    random_ratios = [r.ratio for r in random_rows]
    fixed_ratios = [r.ratio for r in fixed_rows]
    finite = [v for v in ratios if np.isfinite(v)]
    # DENOMINATOR HONESTY. Every base generator is measured at all three rate
    # scales, so the measurement count is 3x the number of distinct matrices,
    # and the fixed fixtures are the SAME matrices at every seed. Reporting
    # "N measurements" as if it were "N generators" is the exact over-count this
    # whole calibration exists to stop, so both denominators are recorded.
    def _base(subset: list[Any]) -> int:
        # (family, label) because labels are only unique WITHIN a family --
        # "gamma=0.1" occurs in amp_damping and inside a rabi label alike.
        return len({(r.family, r.label.split("@c=")[0]) for r in subset})

    return {
        "seed": seed,
        "draws": draws,
        "all": _percentiles(ratios),
        "random_gksl_only": _percentiles(random_ratios),
        "fixed_fixtures_only": _percentiles(fixed_ratios),
        "n_base_random_generators": _base(random_rows),
        "n_base_fixed_generators": _base(fixed_rows),
        "n_at_or_above_shipped_split": sum(
            1 for v in finite if v >= ZERO_MODE_AMBIGUITY_FACTOR
        ),
        "n_at_or_above_lowest_defect": sum(1 for v in finite if v >= overlap_level),
        "argmax_family": max(
            (r for r in rows if np.isfinite(r.ratio)), key=lambda r: r.ratio
        ).family,
        "argmax_label": max(
            (r for r in rows if np.isfinite(r.ratio)), key=lambda r: r.ratio
        ).label,
    }


def _across_seeds(cells: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Spread of a per-seed statistic ACROSS seeds -- the quantity in question."""
    vals = [float(c["all"][key]) for c in cells]
    lo, hi = min(vals), max(vals)
    return {
        "n_seeds": len(vals),
        "min": lo,
        "max": hi,
        "mean": float(np.mean(vals)),
        # null, not Infinity: see _jsonable. A zero denominator means the
        # statistic is identically zero at every seed (the median at the small
        # sample size), i.e. the spread is undefined rather than infinite.
        "spread_factor_max_over_min": _jsonable(hi / lo) if lo > 0.0 else None,
        "values_by_seed": {str(c["seed"]): float(c["all"][key]) for c in cells},
    }


def run(
    seeds: tuple[int, ...] = SWEEP_SEEDS, draws_levels: tuple[int, ...] = SWEEP_DRAWS
) -> dict[str, Any]:
    """Sweep the calibration over seeds and sample sizes."""
    if len(seeds) < 2:
        raise ValueError(
            f"a seed sweep needs at least 2 seeds to show a spread, got {len(seeds)}"
        )
    if any(s < 0 for s in seeds):
        raise ValueError(f"seeds must be non-negative, got {seeds}")
    if any(d < 1 for d in draws_levels):
        raise ValueError(f"draws must be >= 1, got {draws_levels}")

    mod = _calibration_module()
    stiff = _stiff_reference(mod)
    overlap_level = stiff["lowest_defective_ratio"]

    by_draws: dict[str, Any] = {}
    for draws in draws_levels:
        cells = [_one_cell(mod, s, draws, overlap_level) for s in seeds]
        by_draws[str(draws)] = {
            "cells": cells,
            "across_seeds_max": _across_seeds(cells, "max"),
            "across_seeds_p95": _across_seeds(cells, "p95"),
            "across_seeds_median": _across_seeds(cells, "median"),
            "total_n": sum(int(c["all"]["n"]) for c in cells),
            "total_at_or_above_shipped_split": sum(
                int(c["n_at_or_above_shipped_split"]) for c in cells
            ),
            "total_at_or_above_lowest_defect": sum(
                int(c["n_at_or_above_lowest_defect"]) for c in cells
            ),
        }

    # Pool every healthy measurement of the whole sweep for the exceedance
    # answer: the question "does a healthy generator ever reach 30" is about the
    # union of all samples, not about any one of them.
    blocks = [by_draws[str(d)] for d in draws_levels]
    cells = [c for b in blocks for c in b["cells"]]
    total_n = sum(int(b["total_n"]) for b in blocks)
    total_ge_split = sum(int(b["total_at_or_above_shipped_split"]) for b in blocks)
    total_ge_defect = sum(int(b["total_at_or_above_lowest_defect"]) for b in blocks)
    pooled_max = max(float(b["across_seeds_max"]["max"]) for b in blocks)
    argmax_cell = max(cells, key=lambda c: float(c["all"]["max"]))
    # Distinct MATRICES, not measurements: every random draw is fresh, but the
    # fixed fixtures are literally the same 23 matrices in every one of the
    # cells, so they are counted once.
    n_distinct = sum(int(c["n_base_random_generators"]) for c in cells) + max(
        int(c["n_base_fixed_generators"]) for c in cells
    )

    return {
        "artefact": "zero_mode_ambiguity_seed_sweep",
        "artefact_version": 1,
        "issue": "#113",
        "quantity": "max{|lambda| : |lambda| <= certificate.bound} / (eps * ||L||_2)",
        "companion_artefact": "benchmarks/calibration/zero_mode_calibration.json",
        "seeds": list(seeds),
        "draws_levels": list(draws_levels),
        "shipped_split": float(ZERO_MODE_AMBIGUITY_FACTOR),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "eps": float(np.finfo(float).eps),
        },
        "stiff_reference": stiff,
        "by_draws": by_draws,
        "pooled": {
            "n_healthy_measurements": total_n,
            "n_distinct_base_generators": n_distinct,
            "denominator_note": (
                "n_healthy_measurements counts every (generator, rate scale) "
                "pair; each base generator is measured at 3 scales and the fixed "
                "fixtures repeat in every cell. n_distinct_base_generators is "
                "the count of distinct matrices."
            ),
            "healthy_max_over_whole_sweep": pooled_max,
            "healthy_max_seed": argmax_cell["seed"],
            "healthy_max_draws": argmax_cell["draws"],
            "healthy_max_family": argmax_cell["argmax_family"],
            "healthy_max_label": argmax_cell["argmax_label"],
            "n_at_or_above_shipped_split": total_ge_split,
            "n_at_or_above_lowest_defect": total_ge_defect,
            "lowest_defective_stiff_ratio": overlap_level,
            "margin_shipped_split_over_pooled_healthy_max": (
                _jsonable(float(ZERO_MODE_AMBIGUITY_FACTOR) / pooled_max)
                if pooled_max > 0.0
                else None
            ),
            # The number the _consts.py comment quotes as "4.23 against 4.87,
            # a factor 1.15" -- recomputed against the sweep's healthy max
            # instead of one seed's.
            "nearest_pair_factor_over_whole_sweep": (
                _jsonable(overlap_level / pooled_max) if pooled_max > 0.0 else None
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "calibration"
            / "zero_mode_seed_sweep.json"
        ),
        help="destination JSON path (default: the COMMITTED reference artefact)",
    )
    parser.add_argument(
        "--print-only", action="store_true", help="summarise to stdout, write nothing"
    )
    args = parser.parse_args(argv)

    payload = run()
    print(
        f"seeds={len(payload['seeds'])}  draws levels={payload['draws_levels']}  "
        f"shipped split={payload['shipped_split']:g}"
    )
    def _fmt(value: float | None) -> str:
        # The undefined spread (zero denominator) must print as "n/a", not
        # crash the summary and not silently read as a number.
        return "n/a" if value is None else f"{value:.2f}x"

    for draws in payload["draws_levels"]:
        block = payload["by_draws"][str(draws)]
        mx, p95 = block["across_seeds_max"], block["across_seeds_p95"]
        n_per_seed = block["cells"][0]["all"]["n"]
        print(
            f"  draws={draws:<4d} n/seed={n_per_seed:<5d} "
            f"max {mx['min']:.3g}..{mx['max']:.3g} ({_fmt(mx['spread_factor_max_over_min'])})  "
            f"p95 {p95['min']:.3g}..{p95['max']:.3g} ({_fmt(p95['spread_factor_max_over_min'])})"
        )
    pooled = payload["pooled"]
    print(
        f"pooled: n={pooled['n_healthy_measurements']} measurements from "
        f"{pooled['n_distinct_base_generators']} distinct generators  "
        f"healthy max={pooled['healthy_max_over_whole_sweep']:.3g} "
        f"(seed {pooled['healthy_max_seed']}, draws {pooled['healthy_max_draws']}, "
        f"{pooled['healthy_max_family']})  "
        f">= shipped split ({payload['shipped_split']:g}): "
        f"{pooled['n_at_or_above_shipped_split']}  "
        f">= lowest defect ({pooled['lowest_defective_stiff_ratio']:.3g}): "
        f"{pooled['n_at_or_above_lowest_defect']}"
    )
    print(
        f"margin of the shipped split over the pooled healthy max: "
        f"{_fmt(pooled['margin_shipped_split_over_pooled_healthy_max'])}  |  "
        f"nearest pair (lowest defect / healthy max): "
        f"{_fmt(pooled['nearest_pair_factor_over_whole_sweep'])}"
    )

    if args.print_only:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": see the sibling script -- CRLF translation would make the
    # SAME sweep produce different bytes on Windows and break the regenerate-and
    # -diff check that keeps the artefact honest.
    args.out.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
