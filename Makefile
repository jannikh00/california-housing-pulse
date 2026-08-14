# Thin wrapper around the `chp` CLI.
#
# This file deliberately contains no pipeline logic: it sets the environment and
# composes targets, nothing more. `make all` and `uv run chp all` do the same
# work, so Windows users can invoke the CLI directly without losing anything.
#
# UV_NO_EDITABLE is required because uv's editable install writes a .pth file
# without a trailing newline, which Python's site machinery ignores; the package
# then silently fails to import. --reinstall-package keeps the non-editable copy
# in step with the working tree.

UV ?= uv
PACKAGE := california-housing-pulse
export UV_NO_EDITABLE := 1

RUN := $(UV) run --reinstall-package $(PACKAGE)

.PHONY: all setup fetch build features eda baselines test verify lint clean help

help:
	@echo "make setup   - create .venv and install dependencies"
	@echo "make all     - fetch, build, features, eda, baselines, test (full rebuild)"
	@echo "make fetch   - download raw sources and record provenance"
	@echo "make build   - rebuild staged tables, panel, report, and dictionary"
	@echo "make features- build the leakage-safe feature matrix and availability table"
	@echo "make eda     - render the exploratory analysis report and figures"
	@echo "make baselines - fit baselines and models, evaluate, render results"
	@echo "make test    - run the test suite"
	@echo "make verify  - check raw snapshots against recorded hashes"
	@echo "make lint    - run ruff"
	@echo "make clean   - remove rebuildable artifacts (raw snapshots are kept)"

setup:
	$(UV) sync --extra dev

all: setup
	$(RUN) chp all

fetch: setup
	$(RUN) chp fetch

build: setup
	$(RUN) chp build

features: setup
	$(RUN) chp features

eda: setup
	$(RUN) chp eda

baselines: setup
	$(RUN) chp baselines

test: setup
	$(RUN) chp test

verify: setup
	$(RUN) chp verify

lint: setup
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

clean:
	rm -rf data/interim/*.parquet data/processed/*.parquet
