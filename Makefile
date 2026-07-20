.PHONY: install test lint smoke run-mock baselines real-baselines real-experiment-smoke real-trades-2025 train-real eval-real real-experiment-full train-real-zero-fee eval-real-zero-fee real-zero-fee optimize-real-zero-fee train-real-residual-sac train-real-residual-sac-early-stop diagnose-real-fills final-as-comparison convert-orderbook-2025-listed-robust repair-orderbook-2025 check-orderbook-ready

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

real-baselines:
	python scripts/run_real_orderbook_baselines.py --data-dir data/processed/bybit/orderbook/BTCUSDT --start-date 2025-01-01 --end-date 2025-01-07

real-experiment-smoke:
	python scripts/run_real_experiment.py --mode all --start-date 2025-01-01 --end-date 2025-01-07 --timesteps 20000 --seeds 42

real-trades-2025:
	python scripts/run_real_experiment.py --mode prepare --start-date 2025-01-01 --end-date 2025-12-31

train-real:
	python scripts/run_real_experiment.py --mode train --start-date 2025-01-01 --end-date 2025-12-31 --timesteps 500000 --seeds 42,100,200

eval-real:
	python scripts/run_real_experiment.py --mode evaluate --start-date 2025-01-01 --end-date 2025-12-31 --seeds 42,100,200

real-experiment-full:
	python scripts/run_real_experiment.py --mode all --start-date 2025-01-01 --end-date 2025-12-31 --timesteps 500000 --seeds 42,100,200

train-real-zero-fee:
	python scripts/run_real_experiment.py --mode train --start-date 2025-01-01 --end-date 2025-12-31 --timesteps 500000 --seeds 42,100,200 --maker-fee 0 --mandatory-quoting

eval-real-zero-fee:
	python scripts/run_real_experiment.py --mode evaluate --start-date 2025-01-01 --end-date 2025-12-31 --seeds 42,100,200 --maker-fee 0 --mandatory-quoting

real-zero-fee:
	python scripts/run_real_experiment.py --mode full --start-date 2025-01-01 --end-date 2025-12-31 --timesteps 500000 --seeds 42,100,200 --maker-fee 0 --mandatory-quoting

optimize-real-zero-fee:
	python scripts/run_real_experiment.py --mode optimize --start-date 2025-01-01 --end-date 2025-12-31 --maker-fee 0 --mandatory-quoting --max-hours 10

train-real-residual-sac:
	python scripts/run_real_experiment.py --mode residual --start-date 2025-01-01 --end-date 2025-12-31 --seeds 42,100,200 --maker-fee 0 --screening-timesteps 250000 --full-timesteps 700000

train-real-residual-sac-early-stop:
	python scripts/run_real_experiment.py --mode residual-early-stop --start-date 2025-01-01 --end-date 2025-12-31 --seeds 42,100,200 --maker-fee 0 --validation-interval 50000 --early-stop-patience 3 --min-score-improvement 0.01 --full-timesteps 700000

diagnose-real-fills:
	python scripts/run_real_experiment.py --mode diagnose-fills --start-date 2025-01-01 --end-date 2025-12-31 --queue-fraction 0.25

final-as-comparison:
	python scripts/run_real_experiment.py --mode final-as-comparison --start-date 2025-01-01 --end-date 2025-12-31 --maker-fee 0 --queue-fraction 0.25

convert-orderbook-2025-listed-robust:
	python scripts/download_convert_orderbook_range.py --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-12-31 --output-dir data/processed/bybit/orderbook --raw-temp-dir data/raw/bybit/tmp/orderbook --depth 10 --frequency 1s --sleep-seconds 2 --retries 8 --connect-timeout 30 --read-timeout 300 --retry-backoff-seconds 30

repair-orderbook-2025:
	python scripts/download_convert_orderbook_range.py --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-12-31 --output-dir data/processed/bybit/orderbook --raw-temp-dir data/raw/bybit/tmp/orderbook --depth 10 --frequency 1s --sleep-seconds 2 --retries 10 --connect-timeout 30 --read-timeout 300 --retry-backoff-seconds 45 --only-missing

check-orderbook-ready:
	python scripts/check_orderbook_ready.py --input-dir data/processed/bybit/orderbook/BTCUSDT --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-12-31
