# LiouScope developer Makefile.
#
# Common targets:
#   make install      Install package in editable mode with dev extras.
#   make test         Run the full test suite with coverage gate.
#   make anchors      Run only the anchor regression gate.
#   make lint         Lint with ruff.
#   make typecheck    mypy on src/liouscope.
#   make reproduce    Run benchmarks/reproduce_paper.py (deterministic SHA-256).
#   make figures      Generate the three paper figures.
#   make build        Build sdist + wheel.
#   make check-dist   twine check on built artefacts.
#   make validate-pyproject  Schema-validate pyproject.toml.
#   make precommit    Run pre-commit on all files.
#   make clean        Remove build / caches.

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

.PHONY: install test anchors lint typecheck reproduce figures build check-dist \
        validate-pyproject precommit clean help

help:
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk -F'##' '{printf "  %-20s %s\n", $$1, $$2}'

install: ## Install in editable mode with dev,qutip,figures extras
	$(PIP) install -e ".[dev,qutip,figures]"

test: ## Run pytest with coverage gate
	$(PYTHON) -m pytest --cov=liouscope --cov-fail-under=80 -v

anchors: ## Run only the correctness-anchor regression gate
	$(PYTHON) -m pytest tests/test_anchors.py -v

lint: ## ruff check src/ tests/
	$(PYTHON) -m ruff check src/ tests/

typecheck: ## mypy src/liouscope
	$(PYTHON) -m mypy src/liouscope

reproduce: ## Run reproduce_paper benchmark
	$(PYTHON) benchmarks/reproduce_paper.py

figures: ## Generate the three paper figures
	$(PYTHON) -m figures.generate_all

build: ## Build sdist + wheel into dist/
	$(PYTHON) -m build

check-dist: build ## Run twine check on built artefacts
	$(PYTHON) -m twine check dist/*

validate-pyproject: ## Schema-validate pyproject.toml
	$(PYTHON) -m validate_pyproject pyproject.toml

precommit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Remove build artefacts and caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache \
	       benchmarks/output figures/output coverage.xml htmlcov
	find . -name __pycache__ -type d -exec rm -rf {} +
