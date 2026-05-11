"""Command-line entry point: ``python -m liouscope``.

Three sub-commands:

* ``version``   -- print package, taxonomy and schema versions.
* ``info``      -- print the constants and gate status from
                   :mod:`liouscope._consts`.
* ``diagnose``  -- load a Liouvillian from a ``.npy`` file and emit a JSON
                   :class:`DiagnosticReport`.

Example::

    python -m liouscope version
    python -m liouscope info
    python -m liouscope diagnose path/to/L.npy --seed 42 --output report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import (
    DIAGNOSTIC_SCHEMA_VERSION,
    TAXONOMY_VERSION,
    __version__,
    diagnose,
)
from ._consts import A_CLASS_DESCRIPTIONS, F_FAMILY_DESCRIPTIONS
from .io.export import dump_report
from .io.seed import seed_everything


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"liouscope        {__version__}")
    print(f"taxonomy         {TAXONOMY_VERSION}")
    print(f"diagnostic_schema {DIAGNOSTIC_SCHEMA_VERSION}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"liouscope        {__version__}")
    print(f"taxonomy         {TAXONOMY_VERSION}")
    print(f"diagnostic_schema {DIAGNOSTIC_SCHEMA_VERSION}")
    print()
    print("Mechanism classes (A1-A12):")
    for code, descr in A_CLASS_DESCRIPTIONS.items():
        print(f"  {code:>3s}  {descr}")
    print()
    print("Gap-failure families (F1-F5):")
    for code, descr in F_FAMILY_DESCRIPTIONS.items():
        print(f"  {code:>4s}  {descr}")
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    path = Path(args.liouvillian)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    L = np.load(path)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        print(f"error: expected a square 2-D array, got shape {L.shape}", file=sys.stderr)
        return 2

    seed_everything(args.seed)
    report = diagnose(
        L,
        bootstrap_B=args.bootstrap_B,
        seed=args.seed,
        include_mpemba=not args.no_mpemba,
    )

    print(f"Run ID                  {report.governance.run_id[:32]}...")
    print(f"D1 gap                  {report.spectral.gap:.6f}")
    print(f"D2 GNS gap              {report.spectral.gns_gap:.6f}")
    print(f"D2b KMS gap             {report.spectral.kms_gap:.6f}")
    print(f"D9 Petermann max        {report.nonnorm.petermann_max:.3e}")
    print(f"D10 Kreiss              {report.nonnorm.kreiss:.3e}")
    print(f"Best fit                {report.relaxation.aicc_model}")
    print(f"beta_D                  {report.relaxation.beta_D:.6f}")
    lo, hi = report.relaxation.bca_ci_beta
    print(f"BCa CI                  [{lo:.6f}, {hi:.6f}]")
    print(f"A-class / F-family      {report.classification.a_class} / {report.classification.f_family}")
    print(f"Verdict / Tier          {report.classification.verdict} / {report.classification.tier}")
    print(f"Confidence              {report.classification.confidence:.2f}")

    if args.output:
        dump_report(report, args.output)
        print(f"\nReport JSON written to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liouscope",
        description="Multi-diagnostic relaxation analysis for open quantum lattice systems.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_version = sub.add_parser("version", help="print package and schema versions")
    p_version.set_defaults(func=_cmd_version)

    p_info = sub.add_parser("info", help="print constants and gate metadata")
    p_info.set_defaults(func=_cmd_info)

    p_diag = sub.add_parser("diagnose", help="diagnose a Liouvillian from a .npy file")
    p_diag.add_argument(
        "liouvillian",
        help="path to a .npy file containing a (d^2, d^2) Liouvillian superoperator",
    )
    p_diag.add_argument("--seed", type=int, default=42, help="PRNG seed (default 42)")
    p_diag.add_argument("--bootstrap-B", dest="bootstrap_B", type=int, default=200,
                        help="bootstrap resample count (default 200)")
    p_diag.add_argument("--no-mpemba", action="store_true",
                        help="skip D19/D20 Mpemba diagnostics")
    p_diag.add_argument("--output", "-o", help="optional JSON output path")
    p_diag.set_defaults(func=_cmd_diagnose)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
