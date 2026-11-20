#!/usr/bin/env python3
"""Issue #101 slice C, phase 0: rate-unit census of the classifier.

``claim_status: pending`` (KANON ``RESEARCH-LIOUSCOPE``).

A Liouvillian carries a rate dimension. A change of the rate unit, ``L -> cL``
with ``c > 0`` and every time ``t -> t / c``, is the same physics, so a
mechanism class must not depend on ``c``. This script measures how often it
does, on declared control families, with the thresholds as they are. It is the
baseline that the preregistered calibration study of issue #101 slice C has
to improve on (``docs/explanation/f5-calibration-preregistration.md``).

Nothing here is a gate, and nothing here selects a threshold. The script only
prints and stores measurements.

    python benchmarks/issue101_rate_unit_census.py
    python benchmarks/issue101_rate_unit_census.py --per-family 4 --json out.json

The metamorphic relation
------------------------
For each system the pipeline runs once per rate factor ``c`` on ``c * L`` with
the relaxation time grid ``t_ref / c``. With the grid rescaled, every
trajectory is the same curve in rescaled time, so a difference between two
rate factors comes from an estimator grid or a decision threshold that is
absolute in the rate, not from the relaxation fit seeing a different window.
``--default-grid`` runs the library default instead (an absolute
``linspace(0, 10, 80)``), which is what a caller gets who passes no grid.

Families
--------
Design labels follow the control list of the #101 re-audit addendum. They are
labels of the DESIGN, declared before any threshold is fitted, and they are
pending independent physics review like the rest of slice C.

N1  detailed balance     thermal two-level system               F5 negative
N2  pure dephasing       conserved populations, normal          F5 negative
N3  coherent spectator   damped qubit + undamped qubit, gapless F5 negative
N4  driven damped        ordinary non-normal, gapped            F5 negative
P1  directional ladder   strongly non-normal d=4 ladder         F5 positive

Split
-----
Every system has an index. Even indices are the calibration split and odd
indices are the holdout split. The holdout split is evaluated only with
``--split holdout``; the preregistration states when that is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import scipy
from scipy.linalg import expm

import liouscope as ls

RATE_FACTORS: tuple[float, ...] = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3)
REFERENCE_GRID: tuple[float, float, int] = (0.0, 10.0, 80)
BOOTSTRAP_B: int = 20
PIPELINE_SEED: int = 1
FAMILY_SEED: int = 101

# Decision keys of the F5 path as it is on main, then the scale-relative
# candidates of slice A. The census reports the relative range
# ``(max - min) / max`` of each key over the rate factors
# (:func:`relative_range`). A key that a threshold consumes as a pure number
# has to be invariant, i.e. show a range of 0. ``henrici_eta`` is a rate, so
# its range is that of the rate factors themselves, about 1: that is the
# defect issue #101 opened on, not a property of the census.
DECISION_KEYS: tuple[str, ...] = (
    "henrici_eta",
    "pseudospectral_reach_legacy",  # pseudospectral_radius / gap
    "henrici_relative",
    "kreiss_scaled",
    "pseudospectral_radius_rel",
    "pseudospectral_abscissa_rel",
)

_I2 = np.eye(2, dtype=complex)
_SM = np.array([[0, 1], [0, 0]], dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SZ = np.diag([1.0, -1.0]).astype(complex)


@dataclass(frozen=True)
class CensusSystem:
    """One control system at unit rate factor."""

    family: str
    index: int
    L: np.ndarray
    rho_initial: np.ndarray
    rho_steady: np.ndarray | None
    f5_label: str  # "negative" or "positive": the design label


def _rng(family: str, index: int) -> np.random.Generator:
    tag = int.from_bytes(hashlib.sha256(family.encode()).digest()[:4], "big")
    return np.random.default_rng([FAMILY_SEED, tag, index])


def _hermitian(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a + a.conj().T) / 2


def _random_state(rng: np.random.Generator, n: int) -> np.ndarray:
    r = _hermitian(rng, n)
    rho = r @ r.conj().T
    return rho / np.trace(rho).real


def _long_time_state(L: np.ndarray, rho0: np.ndarray, t: float) -> np.ndarray:
    d = rho0.shape[0]
    v = expm(L * t) @ rho0.reshape(-1, order="F")
    rho = v.reshape(d, d, order="F")
    rho = (rho + rho.conj().T) / 2
    return rho / np.trace(rho).real


def detailed_balance(index: int) -> CensusSystem:
    """N1: thermal two-level system at a drawn inverse temperature."""
    rng = _rng("N1", index)
    beta = float(rng.uniform(0.2, 3.0))
    omega = float(rng.uniform(0.5, 2.0))
    gamma = float(rng.uniform(0.2, 2.0))
    n_th = 1.0 / np.expm1(beta * omega)
    H = 0.5 * omega * _SZ
    L = ls.build_liouvillian(
        H, [_SM, _SM.conj().T], [gamma * (n_th + 1.0), gamma * n_th]
    )
    return CensusSystem("N1", index, L, _random_state(rng, 2), None, "negative")


def pure_dephasing(index: int) -> CensusSystem:
    """N2: dephasing with conserved populations (normal).

    The stationary manifold is degenerate, one zero mode per population. The
    gap above it, the slowest coherence decay, is resolved.
    """
    rng = _rng("N2", index)
    d = int(rng.integers(2, 5))
    energies = np.sort(rng.uniform(-1.0, 1.0, size=d))
    H = np.diag(energies).astype(complex)
    jump = np.diag(rng.normal(size=d)).astype(complex)
    L = ls.build_liouvillian(H, [jump], [float(rng.uniform(0.2, 2.0))])
    rho0 = _random_state(rng, d)
    # Populations are conserved, so the steady state depends on rho0: the
    # fully dephased rho0.
    return CensusSystem(
        "N2", index, L, rho0, np.diag(np.diag(rho0)).astype(complex), "negative"
    )


def coherent_spectator(index: int) -> CensusSystem:
    """N3: damped qubit A next to an undamped, coherently rotating qubit B."""
    rng = _rng("N3", index)
    h_a, h_b = _hermitian(rng, 2), _hermitian(rng, 2)
    jump = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    H = np.kron(h_a, _I2) + np.kron(_I2, h_b)
    L = ls.build_liouvillian(
        H, [np.kron(jump, _I2)], [abs(float(rng.normal())) + 0.1]
    )
    rho0 = _random_state(rng, 4)
    return CensusSystem(
        "N3", index, L, rho0, _long_time_state(L, rho0, 300.0), "negative"
    )


def driven_damped(index: int) -> CensusSystem:
    """N4: resonantly driven, amplitude-damped qubit (gapped, non-normal)."""
    rng = _rng("N4", index)
    drive = float(rng.uniform(0.1, 3.0))
    gamma = float(rng.uniform(0.3, 2.0))
    L = ls.build_liouvillian(0.5 * drive * _SX, [_SM], [gamma])
    return CensusSystem("N4", index, L, _random_state(rng, 2), None, "negative")


def directional_ladder(index: int) -> CensusSystem:
    """P1: d=4 directional ladder, the F5 positive control of the anchor suite.

    ``index = 0`` is exactly the operator of
    ``tests/test_validation_systems/test_d17_gap_coherence.py::_phantom_ladder``
    (rates 1.5, 0.1, 0.15). Other indices draw the three rates around it.
    """
    rng = _rng("P1", index)
    d = 4
    low = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        low[i, i + 1] = 1.0
    zdiag = np.diag(np.arange(d, dtype=complex))
    if index == 0:
        rates = [1.5, 0.1, 0.15]
    else:
        rates = [
            float(rng.uniform(1.2, 2.5)),
            float(rng.uniform(0.05, 0.15)),
            float(rng.uniform(0.1, 0.2)),
        ]
    L = ls.build_liouvillian(
        np.zeros((d, d), dtype=complex), [low, low.conj().T, zdiag], rates
    )
    return CensusSystem("P1", index, L, _random_state(rng, d), None, "positive")


FAMILIES: dict[str, Callable[[int], CensusSystem]] = {
    "N1": detailed_balance,
    "N2": pure_dephasing,
    "N3": coherent_spectator,
    "N4": driven_damped,
    "P1": directional_ladder,
}


def split_indices(per_family: int, split: str) -> list[int]:
    """Indices of ``per_family`` systems in the named split.

    Calibration is the even indices, holdout the odd ones, so the two never
    share a system and enlarging one never moves a system into the other.
    """
    if split not in ("calibration", "holdout"):
        raise ValueError(f"unknown split {split!r}")
    offset = 0 if split == "calibration" else 1
    return [2 * k + offset for k in range(per_family)]


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def run_one(
    system: CensusSystem, c: float, *, default_grid: bool = False
) -> dict[str, Any]:
    """The classification and the F5 decision keys of ``system`` at factor ``c``."""
    t_grid = None if default_grid else np.linspace(*REFERENCE_GRID) / c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = ls.diagnose(
            c * system.L,
            rho_initial=system.rho_initial,
            rho_steady_state=system.rho_steady,
            t_grid=t_grid,
            bootstrap_B=BOOTSTRAP_B,
            seed=PIPELINE_SEED,
        )
    cls = rep.classification
    ev = cls.evidence
    gap = ev.get("gap", float("nan"))
    radius = ev.get("pseudospectral_radius", float("nan"))
    resolution = ev.get("gap_resolution", float("nan"))
    floor = resolution if resolution > 0.0 else 0.0
    reach = radius / gap if np.isfinite(gap) and gap > floor else float("nan")
    fired = [h["rule_id"] for h in cls.triggered_hypotheses]
    f5 = next(
        e for e in cls.hypothesis_matrix if e["rule_id"] == "F5_PSEUDOSPECTRAL"
    )
    keys = {
        "henrici_eta": ev.get("henrici_eta", float("nan")),
        "pseudospectral_reach_legacy": reach,
        "henrici_relative": ev.get("henrici_relative", float("nan")),
        "kreiss_scaled": ev.get("kreiss_scaled", float("nan")),
        "pseudospectral_radius_rel": ev.get("pseudospectral_radius_rel", float("nan")),
        "pseudospectral_abscissa_rel": ev.get(
            "pseudospectral_abscissa_rel", float("nan")
        ),
    }
    return {
        "c": c,
        "a_class": cls.a_class,
        "f_family": cls.f_family,
        "verdict": cls.verdict,
        "fired": fired,
        "winner_rule": fired[0] if fired else "A12_FALLBACK",
        "f5_status": f5["status"],
        "f5_fired": "F5_PSEUDOSPECTRAL" in fired,
        "keys": {k: _finite(v) for k, v in keys.items()},
    }


# A dimensionless key at or below this is zero for the invariance measure.
# Round-off zeros of about 1e-16 otherwise show a large max / min although
# nothing varies: measured, a key whose seven values lay between 2e-17 and
# 2e-16 reported a ratio of 13.
ZERO_FLOOR: float = 1.0e-12


def relative_range(values: list[float | None]) -> float | None:
    """``(max - min) / max`` of the available values, 0 for an invariant key.

    Values with ``|v| <= ZERO_FLOOR`` count as exactly 0. Returns None when
    fewer than two values are available (the key was withheld). A key that is
    exactly invariant gives 0; one that vanishes at some rate factors and not
    at others gives 1.
    """
    finite = [0.0 if abs(v) <= ZERO_FLOOR else abs(v) for v in values if v is not None]
    if len(finite) < 2:
        return None
    top = max(finite)
    if top == 0.0:
        return 0.0
    return float((top - min(finite)) / top)


def run_system(
    system: CensusSystem,
    rate_factors: tuple[float, ...] = RATE_FACTORS,
    *,
    default_grid: bool = False,
) -> dict[str, Any]:
    runs = [run_one(system, c, default_grid=default_grid) for c in rate_factors]
    classes = [(r["a_class"], r["f_family"]) for r in runs]
    verdicts = [r["verdict"] for r in runs]
    return {
        "family": system.family,
        "index": system.index,
        "f5_label": system.f5_label,
        "runs": runs,
        "class_stable": len(set(classes)) == 1,
        "verdict_stable": len(set(verdicts)) == 1,
        "n_distinct_classes": len(set(classes)),
        "key_range": {
            k: relative_range([r["keys"][k] for r in runs]) for k in DECISION_KEYS
        },
        "key_available": {
            k: sum(1 for r in runs if r["keys"][k] is not None) for k in DECISION_KEYS
        },
    }


def summarise(systems: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-family counts. Nothing here is a pass or fail."""
    out: dict[str, Any] = {}
    for family in sorted({s["family"] for s in systems}):
        rows = [s for s in systems if s["family"] == family]
        n = len(rows)
        factors = [r["c"] for r in rows[0]["runs"]]
        f5_fired_by_c = {
            repr(c): sum(1 for s in rows if s["runs"][i]["f5_fired"])
            for i, c in enumerate(factors)
        }
        flips: Counter[str] = Counter()
        for s in rows:
            winners = [r["winner_rule"] for r in s["runs"]]
            for a, b in pairwise(winners):
                if a != b:
                    flips[f"{a} -> {b}"] += 1
        ranges: dict[str, float | None] = {}
        for k in DECISION_KEYS:
            vals = [s["key_range"][k] for s in rows if s["key_range"][k] is not None]
            ranges[k] = float(max(vals)) if vals else None
        out[family] = {
            "n": n,
            "f5_label": rows[0]["f5_label"],
            "class_stable": sum(1 for s in rows if s["class_stable"]),
            "verdict_stable": sum(1 for s in rows if s["verdict_stable"]),
            "f5_fired_by_rate_factor": f5_fired_by_c,
            "winner_flips": dict(sorted(flips.items())),
            "max_key_range": ranges,
        }
    return out


def run_census(
    per_family: int = 12,
    *,
    split: str = "calibration",
    rate_factors: tuple[float, ...] = RATE_FACTORS,
    families: tuple[str, ...] | None = None,
    default_grid: bool = False,
) -> dict[str, Any]:
    names = tuple(FAMILIES) if families is None else families
    systems = [
        run_system(FAMILIES[name](index), rate_factors, default_grid=default_grid)
        for name in names
        for index in split_indices(per_family, split)
    ]
    payload: dict[str, Any] = {
        "census": "issue101_rate_unit_census",
        "claim_status": "pending",
        "split": split,
        "per_family": per_family,
        "rate_factors": list(rate_factors),
        "time_grid": "library default" if default_grid else "t_ref / c",
        "reference_grid": list(REFERENCE_GRID),
        "bootstrap_B": BOOTSTRAP_B,
        "pipeline_seed": PIPELINE_SEED,
        "family_seed": FAMILY_SEED,
        "summary": summarise(systems),
        "systems": systems,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    # Provenance is outside the digest: the digest identifies the RESULT, and
    # two platforms that agree should produce the same one.
    payload["provenance"] = {
        "liouscope": ls.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    return payload


def _format(payload: dict[str, Any]) -> str:
    lines = [
        f"rate-unit census, split = {payload['split']}, "
        f"time grid = {payload['time_grid']}, "
        f"rate factors = {payload['rate_factors']}",
        "",
        f"{'family':<7}{'label':<10}{'n':>4}{'class stable':>14}"
        f"{'verdict stable':>16}  F5 fired per rate factor",
    ]
    for family, row in payload["summary"].items():
        fired = " ".join(str(v) for v in row["f5_fired_by_rate_factor"].values())
        lines.append(
            f"{family:<7}{row['f5_label']:<10}{row['n']:>4}"
            f"{row['class_stable']:>14}{row['verdict_stable']:>16}  {fired}"
        )
    lines += ["", "largest (max - min) / max of a decision key over the rate factors:"]
    for family, row in payload["summary"].items():
        parts = [
            f"{k}={v:.3g}" for k, v in row["max_key_range"].items() if v is not None
        ]
        lines.append(f"  {family}: " + ", ".join(parts))
    lines += ["", "winner changes between neighbouring rate factors:"]
    for family, row in payload["summary"].items():
        for flip, count in row["winner_flips"].items():
            lines.append(f"  {family}: {count} x {flip}")
    lines += ["", f"result sha256: {payload['result_sha256']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--split", choices=("calibration", "holdout"),
                        default="calibration")
    parser.add_argument("--default-grid", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)
    payload = run_census(
        args.per_family, split=args.split, default_grid=args.default_grid
    )
    print(_format(payload))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
