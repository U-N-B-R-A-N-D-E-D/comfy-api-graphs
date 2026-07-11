# Makefile for Comfy API Graphs
#
# Honest local workflow. No live PyPI/GitHub publish until configured.

.PHONY: help install install-dev test lint format clean build publish docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package (editable / local)
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,examples]"
	pre-commit install

test:  ## Run test suite
	pytest -v

test-cov:  ## Run tests with coverage
	pytest --cov=comfy_api_graphs --cov-report=html --cov-report=term

lint:  ## Run all linters
	ruff check src tests
	black --check src tests
	mypy src

format:  ## Format code with black and ruff
	black src tests
	ruff check --fix src tests

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean  ## Build package for distribution (local artifacts only)
	python -m build

publish-test:  ## Not configured — refuse silent TestPyPI upload
	@echo "ERROR: TestPyPI publish is not configured for this package yet."
	@echo "Use: make install  (pip install -e .)  until a real release exists."
	@exit 1

publish:  ## Not configured — refuse silent PyPI upload
	@echo "ERROR: PyPI publish is not configured for this package yet."
	@echo "Use: make install  (pip install -e .)  until a real release exists."
	@exit 1

example-basic:  ## Run basic example
	python examples/basic_txt2img.py

example-custom:  ## Run custom workflow example
	python examples/custom_workflow.py

example-video:  ## Run video example (quarantined stubs — expects NotImplementedError)
	python examples/video_generation.py

docs:  ## Documentation lives in README.md / docs/ (no build step)
	@echo "See README.md and docs/ — no separate docs build configured."

version:  ## Show current version
	@python -c "import comfy_api_graphs; print(comfy_api_graphs.__version__)"
