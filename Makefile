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

.PHONY: install test anchors lint typecheck reproduce figures benchmarks golden \
        manifest-hashes docs docs-strict build check-dist validate-pyproject \
        precommit clean help

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

benchmarks: ## Run every benchmark entry from LIOUSCOPE_BENCHMARK_MANIFEST.yaml
	@mkdir -p benchmarks/output
	@for bm in BM-001 BM-002 BM-003 BM-003b; do \
		echo ">>> $$bm"; \
		$(PYTHON) benchmarks/run.py $$bm --output benchmarks/output/$$bm.json || exit 1; \
	done

golden: ## Regenerate benchmarks/golden/*.json (commit alongside manifest hash updates)
	@mkdir -p benchmarks/golden
	@for bm in BM-001 BM-003 BM-003b; do \
		echo ">>> golden $$bm"; \
		$(PYTHON) benchmarks/run.py $$bm --output benchmarks/golden/$$bm.json || exit 1; \
	done
	@echo "Updated golden files. Now run 'make manifest-hashes' to verify SHA-256 in the manifest."

manifest-hashes: ## Verify manifest output_hash matches the SHA-256 of each golden file
	$(PYTHON) -m pytest tests/test_benchmark_manifest_integrity.py -v

docs: ## Build the Sphinx HTML documentation
	$(PYTHON) -m sphinx -b html docs docs/_build/html

docs-strict: ## Build docs with warnings-as-errors (CI gate)
	$(PYTHON) -m sphinx -W --keep-going -b html docs docs/_build/html

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
