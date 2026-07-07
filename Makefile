.PHONY: install test smoke run-mock baselines train-mock eval-mock compare-mock stress-mock train-mock-randomized eval-mock-randomized compare-mock-regimes bybit-dry-run bybit-download-dry-run bybit-download-sample discover-bybit-urls verify-bybit-archive-dry-run verify-bybit-archive-sample inspect-bybit-sample convert-bybit-sample convert-bybit-verified show-splits check-bybit-sample build-bybit-manifest lint

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
	python scripts/evaluate_randomized_ppo.py --episodes 20 --seed 200 --base-config configs/env_mock.yaml --model-path models/mock_ppo_randomized.zip

compare-mock-regimes:
	python scripts/compare_mock_regime_models.py --episodes 20 --seed 300 --base-config configs/env_mock.yaml --static-model-path models/mock_ppo.zip --randomized-model-path models/mock_ppo_randomized.zip

bybit-dry-run:
	python scripts/download_bybit_data.py --config configs/data_bybit.yaml --dry-run

bybit-download-dry-run:
	python scripts/download_bybit_data.py --config configs/data_bybit.yaml --dataset trades --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-03

bybit-download-sample:
	python scripts/download_bybit_data.py --config configs/data_bybit.yaml --dataset trades --symbol BTCUSDT --start-date 2024-01-01 --end-date 2024-01-02 --max-files 1 --execute

discover-bybit-urls:
	python scripts/discover_bybit_urls.py --config configs/data_bybit.yaml --dataset trades --symbol BTCUSDT --date 2024-01-01

verify-bybit-archive-dry-run:
	python scripts/verify_bybit_archive.py --config configs/data_bybit.yaml --dataset trades --symbol BTCUSDT --date 2024-01-01 --output-dir data/raw/bybit/verify

verify-bybit-archive-sample:
	python scripts/verify_bybit_archive.py --config configs/data_bybit.yaml --dataset trades --symbol BTCUSDT --date 2024-01-01 --output-dir data/raw/bybit/verify --execute

inspect-bybit-sample:
	python scripts/inspect_bybit_data.py --path data/sample/bybit_trades_sample.csv --dataset trades

convert-bybit-sample:
	python scripts/convert_bybit_data.py --input data/sample/bybit_trades_sample.csv --dataset trades --output data/processed/bybit/trades/BTCUSDT_sample.parquet

convert-bybit-verified:
	python scripts/convert_bybit_data.py --input data/raw/bybit/verify/trades/BTCUSDT/BTCUSDT2024-01-01.csv.gz --dataset trades --output data/processed/bybit/trades/BTCUSDT_2024-01-01.parquet

show-splits:
	python scripts/show_data_splits.py --protocol configs/experiment_protocol.yaml

check-bybit-sample:
	python scripts/check_processed_data.py --input data/processed/bybit/trades/BTCUSDT_sample.parquet --dataset trades

build-bybit-manifest:
	python scripts/build_dataset_manifest.py --input-dir data/processed/bybit --output data/processed/bybit/manifest.json

lint:
	python -m ruff check src tests scripts
