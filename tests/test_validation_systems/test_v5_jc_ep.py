"""V5 Jaynes-Cummings exceptional-point reference.

Expected: Petermann factor diverges by many orders of magnitude as
``g`` approaches the EP at ``g = kappa / 4``.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import diagnose
from liouscope.examples import v5_jaynes_cummings


# V5 (Jaynes-Cummings near the exceptional point) is genuinely multiscale:
# measured r_max / Delta = 396 at g = 0.05, while a uniform 80-point window over
# ten e-foldings of the slowest mode resolves at most ~8x separation. The
# relaxation layer therefore discloses that the fast mode is stepped over
# (UnderResolvedTransientWarning) -- correct signal, not a false positive, and
# orthogonal to what these tests assert. Pinned positively in
# tests/test_relaxation_grid_scale.py rather than merely silenced here. Filtered
# BY MESSAGE, matching the convention above (the prefix stops before the
# colon: pytest splits filter strings on ':' before regex-matching).
@pytest.mark.filterwarnings("ignore:Relaxation layer")
def test_v5_runs_far_from_ep():
    sys = v5_jaynes_cummings(g=0.05, kappa=1.0, n_max=2)
    report = diagnose(
        sys.L, rho_initial=sys.rho_initial, bootstrap_B=10,
        include_mpemba=False, seed=42,
    )
    assert report.spectral.gap > 0
    assert np.isfinite(report.nonnorm.petermann_max)


# V5 (Jaynes-Cummings near the exceptional point) is genuinely multiscale:
# measured r_max / Delta = 396 at g = 0.05, while a uniform 80-point window over
# ten e-foldings of the slowest mode resolves at most ~8x separation. The
# relaxation layer therefore discloses that the fast mode is stepped over
# (UnderResolvedTransientWarning) -- correct signal, not a false positive, and
# orthogonal to what these tests assert. Pinned positively in
# tests/test_relaxation_grid_scale.py rather than merely silenced here. Filtered
# BY MESSAGE, matching the convention above (the prefix stops before the
# colon: pytest splits filter strings on ':' before regex-matching).
@pytest.mark.filterwarnings("ignore:Relaxation layer")
def test_v5_petermann_grows_near_ep():
    """Petermann factor grows as g approaches the EP at kappa/4."""
    kappa = 1.0
    K_far = None
    K_near = None
    for g in (0.05, 0.24):
        sys = v5_jaynes_cummings(g=g, kappa=kappa, n_max=2)
        report = diagnose(
            sys.L, rho_initial=sys.rho_initial, bootstrap_B=10,
            include_mpemba=False, seed=42,
        )
        if g == 0.05:
            K_far = report.nonnorm.petermann_max
        else:
            K_near = report.nonnorm.petermann_max
    assert K_far is not None and K_near is not None
    # Expect K_near to be substantially larger; loose lower bound.
    assert K_near > K_far
