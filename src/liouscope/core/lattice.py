"""Lattice geometries: 1D chain, 2D square, honeycomb, triangular.

Each constructor returns a :class:`Lattice` carrying the site count and a
neighbour-pair list. Pairs are unordered ``(i, j)`` tuples with ``i < j``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Lattice:
    """Immutable lattice description.

    Parameters
    ----------
    n_sites
        Total site count.
    pairs
        Neighbour pairs as a tuple of ``(i, j)`` with ``i < j``.
    name
        Human-readable geometry label.
    """

    n_sites: int
    pairs: tuple[tuple[int, int], ...]
    name: str = "unnamed"

    def __post_init__(self) -> None:
        for i, j in self.pairs:
            if not (0 <= i < self.n_sites and 0 <= j < self.n_sites):
                raise ValueError(f"Pair ({i}, {j}) out of range for n_sites={self.n_sites}")
            if i >= j:
                raise ValueError(f"Pair ({i}, {j}) must satisfy i < j")


def one_d_chain(n: int, *, periodic: bool = False) -> Lattice:
    """1D chain of ``n`` sites."""
    if n < 2:
        raise ValueError("Chain needs at least two sites")
    pairs = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        pairs.append((0, n - 1))
    return Lattice(n_sites=n, pairs=tuple(pairs), name="1D-chain")


def square_lattice(rows: int, cols: int, *, periodic: bool = False) -> Lattice:
    """Square lattice of shape ``rows x cols`` with nearest-neighbour pairs."""
    if rows < 1 or cols < 1:
        raise ValueError("Lattice dimensions must be positive")

    def idx(r: int, c: int) -> int:
        return r * cols + c

    pairs: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                pairs.append((idx(r, c), idx(r, c + 1)))
            elif periodic and cols > 1:
                pairs.append((idx(r, 0), idx(r, c)))
            if r + 1 < rows:
                pairs.append((idx(r, c), idx(r + 1, c)))
            elif periodic and rows > 1:
                pairs.append((idx(0, c), idx(r, c)))
    return Lattice(
        n_sites=rows * cols,
        pairs=tuple(sorted(set(pairs))),
        name=f"square-{rows}x{cols}",
    )


def honeycomb_lattice(n_units: int) -> Lattice:
    """Honeycomb (brick-wall) lattice with ``2 * n_units`` sites.

    Constructs a finite brick-wall realisation: pairs sites
    ``(2k, 2k+1)`` (intra-cell) and ``(2k+1, 2k+2)`` (inter-cell).
    """
    if n_units < 1:
        raise ValueError("n_units must be at least 1")
    n_sites = 2 * n_units
    pairs: list[tuple[int, int]] = []
    for k in range(n_units):
        pairs.append((2 * k, 2 * k + 1))
        if 2 * k + 2 < n_sites:
            pairs.append((2 * k + 1, 2 * k + 2))
    return Lattice(n_sites=n_sites, pairs=tuple(pairs), name="honeycomb")


def triangular_lattice(rows: int, cols: int) -> Lattice:
    """Triangular lattice via square lattice plus the (r, c)-(r+1, c+1) diagonals."""
    base = square_lattice(rows, cols)
    pairs = list(base.pairs)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows - 1):
        for c in range(cols - 1):
            pairs.append((idx(r, c), idx(r + 1, c + 1)))
    return Lattice(
        n_sites=rows * cols,
        pairs=tuple(sorted(set(pairs))),
        name=f"triangular-{rows}x{cols}",
    )
