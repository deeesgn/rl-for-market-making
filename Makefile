.PHONY: install test smoke lint

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python -m pytest

smoke:
	python scripts/smoke_test.py

lint:
	python -m ruff check src tests scripts
