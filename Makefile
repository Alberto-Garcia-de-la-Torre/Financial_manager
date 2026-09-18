# finmgr — developer entry points.
#
# Everything runs out of the project's own .venv, never the system Python, and
# every target bootstraps that venv first. A cold clone therefore needs nothing
# but `make test`.

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
# Touched after a successful install; its timestamp against pyproject.toml is
# what decides whether the dependencies need reinstalling.
STAMP := $(VENV)/.install-stamp
# Not at the conventional .pre-commit-config.yaml, so every entry point that
# touches the hooks has to name it explicitly.
HOOKS := tools/pre-commit-config.yaml

.DEFAULT_GOAL := help

.PHONY: help install hooks hooks-all test lint fmt check clean

help: ## List the available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

$(STAMP): pyproject.toml | $(BIN)/python
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet --editable ".[dev]"
	@touch $(STAMP)

install: $(STAMP) ## Create .venv and install the package with its dev extras

hooks: $(STAMP) ## Install the git pre-commit hooks (run once per clone)
	$(BIN)/pre-commit install --config $(HOOKS)

hooks-all: $(STAMP) ## Run every hook over the whole tree, not just staged files
	$(BIN)/pre-commit run --config $(HOOKS) --all-files

test: $(STAMP) ## Run the test suite
	$(BIN)/pytest

lint: $(STAMP) ## Check style and formatting without changing anything
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

fmt: $(STAMP) ## Reformat and apply the safe lint fixes
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

check: lint test ## What CI would run: lint, then tests

clean: ## Remove caches and build artefacts (leaves .venv and data/ alone)
	rm -rf .pytest_cache .ruff_cache build dist
	find . -path ./$(VENV) -prune -o -name '__pycache__' -type d -print0 \
		| xargs -0 rm -rf
