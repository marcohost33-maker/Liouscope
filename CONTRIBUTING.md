# Contributing to LiouScope

Thanks for your interest in LiouScope. This document describes how to
develop, test and submit changes to the project.

## Development setup

```bash
git clone https://github.com/coworker-research/liouscope
cd liouscope
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,qutip]
```

## Running the test suite

```bash
pytest --cov=liouscope --cov-fail-under=90
pytest tests/test_anchors.py            # correctness-anchor regression gate
```

## Lint and type-check

```bash
ruff check src/ tests/
mypy src/liouscope
```

## Reproducibility

Every numerical change must keep the `reproduce_paper.py` SHA-256 hash stable
(or update the recorded hash in the same commit with a clear explanation).

```bash
python benchmarks/reproduce_paper.py
```

## Correctness anchors

`tests/test_anchors.py` encodes the fourteen historical correctness anchors
A--N. **Any anchor failure blocks release.** When in doubt, add a new anchor
test rather than removing one.

## Style

- Frozen `@dataclass(frozen=True, slots=True, kw_only=True)` for result containers.
- `order='F'` (column-stacking) everywhere -- never `order='C'`.
- `scipy.linalg.eig` (zgeev) for non-Hermitian eigenproblems -- never `eigh`.
- Type hints on public functions.

## Spec reference

The canonical specification is the v2.0 consolidated report (2026-05-10).
`TAXONOMY_VERSION = "A1-A12-v3.1"` and `DIAGNOSTIC_SCHEMA_VERSION =
"D1-D24-Übersicht-v3-2026-04-24"` must remain stamped on every result.
