.PHONY: install test smoke run-mock baselines train-mock eval-mock compare-mock lint

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

train-mock:
	python scripts/train_mock_ppo.py --timesteps 5000 --seed 42 --config configs/env_mock.yaml --model-path models/mock_ppo.zip

eval-mock:
	python scripts/evaluate_mock_ppo.py --episodes 20 --seed 100 --config configs/env_mock.yaml --model-path models/mock_ppo.zip

compare-mock:
	python scripts/compare_mock_strategies.py --episodes 20 --seed 100 --config configs/env_mock.yaml --model-path models/mock_ppo.zip --show-actions

lint:
	python -m ruff check src tests scripts
