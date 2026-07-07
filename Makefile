.PHONY: install test smoke run-mock baselines lint

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python -m pytest

smoke:
	python scripts/smoke_test.py

run-mock:
	python scripts/run_mock_env.py --config configs/env_mock.yaml

baselines:
	python scripts/run_baselines.py --episodes 20 --seed 42 --config configs/env_mock.yaml

lint:
	python -m ruff check src tests scripts
