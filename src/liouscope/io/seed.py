"""Deterministic seed handling and the SPEC 7 ``rng`` bridge (phase a).

SPEC 7 (scientific-python.org/specs/spec-0007/) standardises an ``rng``
keyword over legacy ``seed`` arguments. LiouScope adopts the *additive* first
phase: every seeded entry point accepts ``rng`` alongside ``seed``, raises if
both are supplied, and keeps ``seed``-only calls byte-identical to before.

The manifest contract (``GovernanceMetadata.seed``, ``make_run_id``) requires a
plain non-negative integer, so ``rng`` inputs are normalised to a
*derived integer seed* via :func:`derive_seed` and that derived value is what
the run manifest records — a manifest alone therefore still reproduces the run.
"""

from __future__ import annotations

import os
import random
from typing import Final, TypeAlias

import numpy as np

DEFAULT_SEED: Final[int] = 42

# SPEC 7 vocabulary: a SeedLike seeds a fresh Generator, an RNGLike *is* the
# random state. Phase (a) accepts both through the single ``rng`` keyword.
SeedLike: TypeAlias = int | np.integer | np.random.SeedSequence
RNGLike: TypeAlias = np.random.Generator | np.random.BitGenerator


def _validate_seed_int(seed: object, *, name: str = "seed") -> int:
    """Reject bools/negatives; return the value as a plain ``int``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"{name} must be a non-negative int, got {seed!r}")
    return int(seed)


def derive_seed(
    rng: RNGLike | SeedLike | None = None,
    seed: int | None = None,
    *,
    default: int = DEFAULT_SEED,
) -> int:
    """Normalise the SPEC 7 ``rng`` keyword to the manifest's integer seed.

    Exactly one of ``rng`` / ``seed`` may be supplied (SPEC 7 phase a); with
    neither, ``default`` preserves the legacy behaviour of the call site.

    The legacy ``seed`` path is returned unchanged (beyond bool/negative
    rejection) so existing runs keep their run_id. ``rng`` inputs map as:

    - ``int``: used directly — ``np.random.default_rng(rng)`` and the derived
      seed produce the same stream, so this is a pure spelling change.
    - ``np.random.SeedSequence``: first ``uint32`` of ``generate_state`` — a
      deterministic function of the sequence's entropy, recorded in the
      manifest in its place.
    - ``Generator`` / ``BitGenerator``: one ``uint32`` draw. This advances the
      caller's generator (standard SPEC 7 consumption semantics) and the drawn
      value becomes the manifest seed, keeping the run reproducible from the
      manifest even though the caller's generator state itself is not
      serialisable.
    """
    if rng is not None and seed is not None:
        raise ValueError(
            "pass either rng (SPEC 7) or the legacy seed, not both"
        )
    if rng is None:
        return default if seed is None else _validate_seed_int(seed)
    if isinstance(rng, bool):
        raise ValueError(f"rng must not be a bool, got {rng!r}")
    if isinstance(rng, (int, np.integer)):
        return _validate_seed_int(rng, name="rng")
    if isinstance(rng, np.random.SeedSequence):
        return int(rng.generate_state(1, dtype=np.uint32)[0])
    if isinstance(rng, (np.random.Generator, np.random.BitGenerator)):
        gen = np.random.default_rng(rng)
        return int(gen.integers(0, 2**32))
    raise TypeError(
        "rng must be an int, numpy integer, SeedSequence, Generator or "
        f"BitGenerator, got {type(rng).__name__}"
    )


def seed_everything(
    seed: int | None = None,
    *,
    rng: RNGLike | SeedLike | None = None,
) -> np.random.Generator:
    """Pin Python, NumPy and the ``PYTHONHASHSEED`` for reproducibility.

    Accepts the legacy positional ``seed`` (default 42) or the SPEC 7 ``rng``
    keyword — but not both. Returns the dedicated ``np.random.Generator``
    callers should use for subsequent random draws (preferred over the legacy
    global state).
    """
    resolved = derive_seed(rng, seed, default=DEFAULT_SEED)
    # Legacy ``np.random.seed`` only accepts 0 <= seed < 2**32 (MT19937 uint32
    # seeding). Reject out-of-range seeds *before* mutating any global state so
    # a bad seed fails closed instead of leaving PYTHONHASHSEED / random.seed
    # partially applied while np.random.seed raises mid-function.
    if resolved >= 2**32:
        raise ValueError(
            f"seed must be < 2**32 for legacy np.random.seed, got {resolved}"
        )
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)
    return np.random.default_rng(resolved)
