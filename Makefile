.PHONY: install test lint

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts
