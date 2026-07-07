.PHONY: install install-dev lint format typecheck test test-fast coverage openapi docker-build docker-run clean

PY ?= python3
VENVDIR ?= .venv
BIN := $(VENVDIR)/bin

install:
	$(PY) -m venv $(VENVDIR)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[serve]"

install-dev:
	$(PY) -m venv $(VENVDIR)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[serve,dev]"
	$(BIN)/pre-commit install || true

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest

test-fast:
	$(BIN)/pytest -m "not slow"

coverage:
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=xml

openapi:
	$(BIN)/python scripts/export_openapi.py docs/openapi.json

docker-build:
	docker build -t whisperlite:latest .

docker-run:
	docker compose up

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
