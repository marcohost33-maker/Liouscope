from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one occurrence, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str, start_marker: str, end_marker: str, replacement: str, *, label: str
) -> str:
    if text.count(start_marker) != 1:
        raise SystemExit(f"{label}: start marker count is {text.count(start_marker)}, expected 1")
    start = text.index(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"{label}: end marker not found")
    return text[:start] + replacement + text[end:]


# ---------------------------------------------------------------------------
# Production contract: H is a Hamiltonian input, so its structural
# Hermiticity tolerance is normalized only by the canonical coherent scale.
# Dissipation remains diagnostic information but cannot relax that contract.
# ---------------------------------------------------------------------------

dense_path = "src/liouscope/core/lindblad.py"
dense = read(dense_path)
dense = replace_between(
    dense,
    "    # PR #127 -- WHAT THE TOLERANCE IS RELATIVE TO",
    "    canonical = _canonical_generator_scales(H, jump_ops, rates, d_h)\n",
    """    # PR #127 E3 RESOLUTION (cross-family review, 2026-09-12): this API\n    # accepts a Hamiltonian H, not an arbitrary effective non-Hermitian\n    # generator decomposition. Hermiticity is therefore a structural contract\n    # of the coherent component. Canonicalizing the Lindblad gauge is still\n    # useful because L -> L + c I carries a Hermitian compensation into H; the\n    # coherent reference must be measured after that compensation. Physical\n    # dissipation is reported below for diagnostics, but it MUST NOT enlarge\n    # the tolerance for a non-Hermitian Hamiltonian. If a future API accepts a\n    # general generator/effective non-Hermitian Hamiltonian, it needs a separate\n    # validator and contract rather than weakening build_liouvillian(H, ...).\n""",
    label="dense rationale",
)
dense = replace_once(
    dense,
    "        reference = max(coherent_scale, dissipation_scale)\n",
    "        reference = coherent_scale\n",
    label="dense reference",
)
dense = replace_once(
    dense,
    "of the generator scale (max|H - H^dag| = {defect:.3e}, ",
    "of the canonical coherent Hamiltonian scale (max|H - H^dag| = {defect:.3e}, ",
    label="dense error contract",
)
write(dense_path, dense)

sparse_path = "src/liouscope/sparse/build.py"
sparse = read(sparse_path)
sparse = replace_between(
    sparse,
    "    # Generator-relative tolerance in the canonical Lindblad gauge, in parity\n",
    "    canonical = _sparse_canonical_generator_scales(H_sp, jump_ops, rates, d)\n",
    """    # PR #127 E3 RESOLUTION: mirror the dense component contract. The\n    # Hamiltonian must be Hermitian relative to its canonical coherent scale;\n    # physical dissipation is diagnostic and cannot excuse a structural H\n    # defect. Canonical Lindblad-gauge compensation remains load-bearing.\n""",
    label="sparse rationale",
)
sparse = replace_once(
    sparse,
    "        reference = max(coherent_scale, dissipation_scale)\n",
    "        reference = coherent_scale\n",
    label="sparse reference",
)
sparse = replace_once(
    sparse,
    "of the generator scale (max|H - H^dag| = {defect:.3e}, ",
    "of the canonical coherent Hamiltonian scale (max|H - H^dag| = {defect:.3e}, ",
    label="sparse error contract",
)
write(sparse_path, sparse)


# ---------------------------------------------------------------------------
# Migrate the two historical test files that encoded the superseded E3 answer.
# Keep the gauge-, unit-, overflow- and dense/sparse-parity tests intact.
# ---------------------------------------------------------------------------

test_path = "tests/test_pr127_generator_relative_hermiticity.py"
test = read(test_path)
first = test.find('"""')
second = test.find('"""', first + 3)
if first != 0 or second < 0:
    raise SystemExit("generator-relative tests: module docstring markers not found")
test = (
    '"""PR #127 Hermiticity contract after E3 cross-family resolution.\n\n'
    "The builder API accepts H as a Hamiltonian. Canonical Lindblad-gauge\n"
    "compensation remains part of the measurement, but physical dissipation\n"
    "does not relax the structural Hermiticity requirement of H. The tests\n"
    "below preserve the earlier gauge/unit/overflow discrimination while\n"
    "reversing the two fixtures that had encoded dissipation as an excuse.\n"
    '"""'
    + test[second + 3 :]
)

test = replace_between(
    test,
    "def test_the_pure_gauge_fixture_is_accepted_because_of_its_dissipator() -> None:\n",
    "# ---------------------------------------------------------------------------\n# Negative controls: the excuse is a MAGNITUDE, not the presence of a jump op\n",
    """def test_nonhermitian_pure_gauge_fixture_is_rejected_despite_dissipation() -> None:\n    \"\"\"F1 reversed by E3: dissipation cannot make non-Hermitian H valid.\"\"\"\n    H = np.eye(2, dtype=complex)\n    H[0, 1] = 0.5 * _EPS\n    _one_orbit_premise(H)\n    dense, sparse = _verdicts(H, [_SIGMA_MINUS])\n    assert dense is not None and \"Hermitian\" in dense, dense\n    assert sparse is not None and \"Hermitian\" in sparse, sparse\n\n\n@pytest.mark.parametrize(\"rate\", [0.0, 1.0e-12, 1.0, 1.0e12])\ndef test_dissipation_strength_cannot_move_hamiltonian_structure_verdict(\n    rate: float,\n) -> None:\n    H = np.eye(2, dtype=complex)\n    H[0, 1] = 0.5 * _EPS\n    dense, sparse = _verdicts(H, [_SIGMA_MINUS], [rate])\n    assert dense is not None and \"Hermitian\" in dense, (rate, dense)\n    assert sparse is not None and \"Hermitian\" in sparse, (rate, sparse)\n\n\n@pytest.mark.parametrize(\"rate\", [0.0, 1.0e-12, 1.0, 1.0e12])\ndef test_exactly_hermitian_hamiltonian_survives_dissipation_sweep(rate: float) -> None:\n    H = np.array([[1.0, 0.25], [0.25, -1.0]], dtype=complex)\n    assert _verdicts(H, [_SIGMA_MINUS], [rate]) == (None, None)\n\n\n# ---------------------------------------------------------------------------\n# Negative controls: dissipation never excuses a Hamiltonian defect\n""",
    label="F1 migration",
)

test = replace_between(
    test,
    '@pytest.mark.parametrize("unit", [1.0e-6, 1.0, 1.0e6])\n',
    "# ---------------------------------------------------------------------------\n# Round 2 of the review: the scale must be a function of the GENERATOR.\n",
    """@pytest.mark.parametrize(\"unit\", [1.0e-6, 1.0, 1.0e6])\ndef test_the_verdict_does_not_move_with_the_units(unit: float) -> None:\n    \"\"\"Rescaling all rate/energy quantities cannot change H-structure validity.\"\"\"\n    H = unit * np.eye(2, dtype=complex)\n    H[0, 1] = unit * 0.5 * _EPS\n    dense, sparse = _verdicts(H, [_SIGMA_MINUS], [unit])\n    assert dense is not None and \"Hermitian\" in dense, (unit, dense)\n    assert sparse is not None and \"Hermitian\" in sparse, (unit, sparse)\n    dense, sparse = _verdicts(H, [])\n    assert dense is not None and sparse is not None\n\n\n# ---------------------------------------------------------------------------\n# Round 2 of the review: canonical Lindblad-gauge compensation remains load-bearing.\n""",
    label="unit migration",
)

test = replace_between(
    test,
    "@pytest.mark.parametrize(\n    (\"fraction\", \"accepted\"), [(0.75, False), (0.25, True)], ids=[\"above\", \"below\"]\n)\n",
    "def test_the_lindblad_gauge_with_unequal_rates_keeps_verdict_and_parity() -> None:\n",
    """@pytest.mark.parametrize(\"fraction\", [0.25, 0.75])\ndef test_dissipation_scale_is_diagnostic_not_an_acceptance_reference(\n    fraction: float,\n) -> None:\n    \"\"\"Both sides of the former K/2 boundary are now structural H failures.\"\"\"\n    from liouscope._consts import EPS_HERMITICITY\n\n    H = np.eye(2, dtype=complex)\n    H[0, 1] = fraction * EPS_HERMITICITY\n    dense, sparse = _verdicts(H, [_SIGMA_MINUS])\n    assert dense is not None and \"Hermitian\" in dense, (fraction, dense)\n    assert sparse is not None and \"Hermitian\" in sparse, (fraction, sparse)\n\n\ndef test_the_lindblad_gauge_with_unequal_rates_keeps_verdict_and_parity() -> None:\n""",
    label="K/2 migration",
)
write(test_path, test)

round19_path = "tests/test_pr127_review_round19.py"
round19 = read(round19_path)
round19 = replace_between(
    round19,
    "def test_pure_gauge_hamiltonian_is_accepted_by_both_builders() -> None:\n",
    "def test_a_real_hermiticity_defect_is_still_rejected() -> None:\n",
    """def test_numerically_nonhermitian_pure_gauge_fixture_is_rejected() -> None:\n    \"\"\"E3: an identity gauge term cannot hide the remaining H defect.\"\"\"\n    h = _numerically_pure_gauge_hamiltonian()\n    defect = float(np.max(np.abs(h - h.conj().T)))\n    gauge = h - (np.trace(h).real / 2.0) * np.eye(2, dtype=complex)\n    gauge_scale = float(np.max(np.abs(gauge)))\n    assert defect == 0.5 * float(np.finfo(float).eps)\n    assert defect > 1.0e-9 * gauge_scale\n    for builder in (build_liouvillian, build_sparse_liouvillian):\n        with pytest.raises(ValueError, match=\"Hermitian\"):\n            builder(h, [_SIGMA_MINUS])\n\n\ndef test_exact_identity_hamiltonian_is_still_accepted_by_both_builders() -> None:\n    h = np.eye(2, dtype=complex)\n    assert build_liouvillian(h, [_SIGMA_MINUS]).shape == (4, 4)\n    assert build_sparse_liouvillian(h, [_SIGMA_MINUS]).shape == (4, 4)\n\n\ndef test_a_real_hermiticity_defect_is_still_rejected() -> None:\n""",
    label="round19 pure-gauge migration",
)
write(round19_path, round19)

print("PR #127 E3 migration applied with all guards satisfied")
