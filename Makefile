.PHONY: install test lint smoke run-mock baselines convert-orderbook-2025-listed-robust repair-orderbook-2025 check-orderbook-ready

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts

smoke:
	python scripts/smoke_test.py

run-mock:
	python scripts/run_mock_env.py --config configs/env_mock.yaml

baselines:
	python scripts/run_baselines.py --episodes 20 --seed 42 --config configs/env_mock.yaml

convert-orderbook-2025-listed-robust:
	python scripts/download_convert_orderbook_range.py --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-12-31 --output-dir data/processed/bybit/orderbook --raw-temp-dir data/raw/bybit/tmp/orderbook --depth 10 --frequency 1s --sleep-seconds 2 --retries 8 --connect-timeout 30 --read-timeout 300 --retry-backoff-seconds 30

repair-orderbook-2025:
	python scripts/download_convert_orderbook_range.py --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-12-31 --output-dir data/processed/bybit/orderbook --raw-temp-dir data/raw/bybit/tmp/orderbook --depth 10 --frequency 1s --sleep-seconds 2 --retries 10 --connect-timeout 30 --read-timeout 300 --retry-backoff-seconds 45 --only-missing

check-orderbook-ready:
	python scripts/check_orderbook_ready.py --input-dir data/processed/bybit/orderbook/BTCUSDT --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-12-31
