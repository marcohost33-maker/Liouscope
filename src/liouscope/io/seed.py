"""Deterministic seed handling."""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

DEFAULT_SEED: Final[int] = 42


def seed_everything(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Pin Python, NumPy and the ``PYTHONHASHSEED`` for reproducibility.

    Returns the dedicated ``np.random.Generator`` callers should use for
    subsequent random draws (preferred over the legacy global state).
    """
    # Validate through an `object`-typed alias: the annotation promises `int`,
    # but callers outside the typed surface can pass anything, and `bool` is an
    # `int` subclass that would silently seed with 0/1.
    seed_obj: object = seed
    if isinstance(seed_obj, bool) or not isinstance(seed_obj, int) or seed_obj < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
