.PHONY: install test smoke run-mock baselines train-mock eval-mock compare-mock stress-mock train-mock-randomized eval-mock-randomized compare-mock-regimes lint

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

stress-mock:
	python scripts/run_mock_regime_stress.py --episodes 20 --seed 200 --base-config configs/env_mock.yaml --model-path models/mock_ppo.zip

train-mock-randomized:
	python scripts/train_mock_ppo_randomized.py --timesteps 5000 --seed 42 --config configs/env_mock.yaml --model-path models/mock_ppo_randomized.zip

eval-mock-randomized:
	python scripts/evaluate_randomized_ppo.py --episodes 20 --seed 300 --base-config configs/env_mock.yaml --model-path models/mock_ppo_randomized.zip

compare-mock-regimes:
	python scripts/compare_mock_regime_models.py --episodes 20 --seed 300 --base-config configs/env_mock.yaml --static-model-path models/mock_ppo.zip --randomized-model-path models/mock_ppo_randomized.zip

lint:
	python -m ruff check src tests scripts
