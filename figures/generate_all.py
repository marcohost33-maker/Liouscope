"""Master script generating all three paper figures.

Run::

    python figures/generate_all.py

Requires matplotlib. Falls back to printing the data tables when matplotlib
is unavailable.
"""

from __future__ import annotations

from pathlib import Path

from .fig1_spectrum import generate as gen_fig1
from .fig2_petermann import generate as gen_fig2
from .fig3_classification import generate as gen_fig3


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    gen_fig1(out_dir / "fig1_spectrum.pdf")
    gen_fig2(out_dir / "fig2_petermann.pdf")
    gen_fig3(out_dir / "fig3_classification.pdf")
    print(f"All figures emitted to {out_dir}")


if __name__ == "__main__":
    main()
