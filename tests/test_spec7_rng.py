"""SPEC 7 phase (a) contract: additive ``rng`` keyword alongside legacy ``seed``.

Issue #72 item 1. The bridge is :func:`liouscope.io.seed.derive_seed`; the
contracts pinned here are:

1. seed-only and no-arg calls are byte-identical to the pre-SPEC-7 behaviour;
2. ``rng`` and ``seed`` together raise (no silent precedence);
3. every accepted ``rng`` type maps deterministically to a manifest-recordable
   integer seed, so a run manifest alone still reproduces the run.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import derive_seed, diagnose, seed_everything
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.lep import compute_lep_layer, initial_state_sensitivity


def _small_liouvillian() -> np.ndarray:
    H = np.array([[1.0, 0.2], [0.2, -1.0]], dtype=complex)
    L_op = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return build_liouvillian(H, [L_op])


# ---------------------------------------------------------------- derive_seed


def test_derive_seed_defaults_to_call_site_default():
    assert derive_seed(None, None, default=42) == 42
    assert derive_seed(None, None, default=7) == 7


def test_derive_seed_legacy_seed_passes_through_unchanged():
    assert derive_seed(None, 123) == 123
    # Legacy path must not clamp: manifest only requires a non-negative int.
    assert derive_seed(None, 2**40) == 2**40


def test_derive_seed_rejects_both():
    with pytest.raises(ValueError, match="not both"):
        derive_seed(np.random.default_rng(0), 1)
    with pytest.raises(ValueError, match="not both"):
        derive_seed(5, 5)


def test_derive_seed_int_rng_is_identity():
    assert derive_seed(17, None) == 17
    assert derive_seed(np.int64(17), None) == 17


@pytest.mark.parametrize("bad", [True, False])
def test_derive_seed_rejects_bool(bad):
    with pytest.raises(ValueError):
        derive_seed(bad, None)
    with pytest.raises(ValueError):
        derive_seed(None, bad)


def test_derive_seed_rejects_negative_and_wrong_type():
    with pytest.raises(ValueError):
        derive_seed(-1, None)
    with pytest.raises(ValueError):
        derive_seed(None, -1)
    with pytest.raises(TypeError):
        derive_seed("not-an-rng", None)  # type: ignore[arg-type]


def test_derive_seed_seedsequence_is_deterministic_and_uint32():
    ss = np.random.SeedSequence(99)
    a = derive_seed(np.random.SeedSequence(99), None)
    b = derive_seed(ss, None)
    assert a == b
    assert 0 <= a < 2**32
    assert derive_seed(np.random.SeedSequence(100), None) != a


def test_derive_seed_generator_is_state_deterministic_and_consuming():
    a = derive_seed(np.random.default_rng(5), None)
    b = derive_seed(np.random.default_rng(5), None)
    assert a == b
    assert 0 <= a < 2**32
    # SPEC 7 consumption semantics: the caller's generator advances, so two
    # derivations from the SAME generator object differ.
    gen = np.random.default_rng(5)
    first = derive_seed(gen, None)
    second = derive_seed(gen, None)
    assert first == a
    assert first != second


def test_derive_seed_bitgenerator_matches_wrapping_generator():
    from_bitgen = derive_seed(np.random.PCG64(11), None)
    from_gen = derive_seed(np.random.default_rng(np.random.PCG64(11)), None)
    assert from_bitgen == from_gen


# ------------------------------------------------------------ seed_everything


def test_seed_everything_rng_equals_seed():
    r1 = seed_everything(0)
    r2 = seed_everything(rng=0)
    assert r1.random() == r2.random()


def test_seed_everything_rejects_both():
    with pytest.raises(ValueError, match="not both"):
        seed_everything(0, rng=0)


def test_seed_everything_default_unchanged():
    r1 = seed_everything()
    r2 = seed_everything(42)
    assert r1.random() == r2.random()


def test_seed_everything_still_rejects_out_of_range():
    with pytest.raises(ValueError, match="2\\*\\*32"):
        seed_everything(2**32)


# ------------------------------------------------------------------ diagnose


def test_diagnose_rng_int_equals_seed():
    L = _small_liouvillian()
    r_seed = diagnose(L, bootstrap_B=20, seed=42, include_mpemba=False)
    r_rng = diagnose(L, bootstrap_B=20, rng=42, include_mpemba=False)
    assert r_seed.governance.seed == r_rng.governance.seed == 42
    assert r_seed.governance.run_id == r_rng.governance.run_id
    assert r_seed.lep.initial_state_sensitivity == r_rng.lep.initial_state_sensitivity


def test_diagnose_default_seed_still_42():
    L = _small_liouvillian()
    r_default = diagnose(L, bootstrap_B=20, include_mpemba=False)
    r_explicit = diagnose(L, bootstrap_B=20, seed=42, include_mpemba=False)
    assert r_default.governance.seed == 42
    assert r_default.governance.run_id == r_explicit.governance.run_id


def test_diagnose_rejects_both_rng_and_seed():
    L = _small_liouvillian()
    with pytest.raises(ValueError, match="not both"):
        diagnose(L, seed=1, rng=np.random.default_rng(1))


def test_diagnose_generator_rng_records_derived_seed_in_manifest():
    L = _small_liouvillian()
    expected = derive_seed(np.random.default_rng(5), None)
    report = diagnose(
        L, bootstrap_B=20, rng=np.random.default_rng(5), include_mpemba=False
    )
    assert report.governance.seed == expected
    # Manifest-based reproduction: rerunning from the recorded integer seed
    # yields the identical run identity.
    replay = diagnose(L, bootstrap_B=20, seed=expected, include_mpemba=False)
    assert replay.governance.run_id == report.governance.run_id


# ----------------------------------------------------------------- D18 / LEP


def test_initial_state_sensitivity_rng_equals_seed():
    L = _small_liouvillian()
    rho_ss = np.eye(2, dtype=complex) / 2
    s_seed = initial_state_sensitivity(L, rho_ss, seed=7)
    s_rng = initial_state_sensitivity(L, rho_ss, rng=7)
    s_default = initial_state_sensitivity(L, rho_ss)
    assert s_seed == s_rng == s_default
    with pytest.raises(ValueError, match="not both"):
        initial_state_sensitivity(L, rho_ss, seed=7, rng=7)


def test_compute_lep_layer_rng_passthrough():
    L = _small_liouvillian()
    eigs = np.linalg.eigvals(L)
    kwargs = {"beta_D_linear": 1.0, "gap": 1.0}
    r_seed = compute_lep_layer(L, eigs, seed=7, **kwargs)
    r_rng = compute_lep_layer(L, eigs, rng=np.random.SeedSequence(3), **kwargs)
    r_rng_repeat = compute_lep_layer(
        L, eigs, rng=np.random.SeedSequence(3), **kwargs
    )
    assert r_seed.initial_state_sensitivity > 0
    assert r_rng.initial_state_sensitivity == r_rng_repeat.initial_state_sensitivity
