# RL for Market Making

A compact quant research project for market-making experiments with inventory risk control.
The repository includes mock mechanics, a listing-driven Bybit BTCUSDT Top-10 orderbook pipeline,
one-second trade aggregation, a real-data replay environment, rule-based baselines, and a minimal
PPO experiment. `hftbacktest` integration is not part of the current working path.

## Project Layout

```text
.
├── configs/
│   └── env_mock.yaml
├── data/
│   └── processed/bybit/orderbook/   # ignored local parquet output
├── scripts/
│   ├── check_orderbook_ready.py
│   ├── download_convert_orderbook_range.py
│   ├── run_baselines.py
│   ├── run_mock_env.py
│   ├── run_real_orderbook_baselines.py
│   ├── run_real_experiment.py
│   └── smoke_test.py
├── src/rl_mm/
│   ├── backtest/
│   ├── data/
│   ├── env/
│   └── strategies/
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install
make smoke
```

Docker remains available for a clean smoke-test environment:

```bash
docker build -t rl-mm .
docker run --rm rl-mm
```

## Tests

Run the active test suite and lint checks:

```bash
make test
make lint
```

## Mock Market Making

Run one deterministic mock episode or compare the fixed-spread and inventory-skew baselines:

```bash
make run-mock
make baselines
```

Run the same rule-based baselines on processed BTCUSDT orderbook snapshots:

```bash
make real-baselines
```

Run the complete seven-day trade preparation, PPO training, and common-window evaluation smoke
experiment:

```bash
make real-experiment-smoke
```

The full-year workflow remains explicit and resumable:

```bash
make real-trades-2025
make train-real
make eval-real
```

Processed trades are stored under `data/processed/bybit/trades/BTCUSDT/`. PPO models are local,
ignored outputs under `models/` and are saved separately as `real_ppo_seed42.zip`,
`real_ppo_seed100.zip`, and `real_ppo_seed200.zip`. Existing completed models are reused; pass
`--force-train` directly to `run_real_experiment.py` only when deliberate retraining is needed.

## Convert 2025 Orderbook Data

The converter fetches the exact remote BTCUSDT archive listing, processes only listed files,
streams each ZIP through the Top-10 parser, writes parquet atomically, and resumes by skipping
completed dates:

```bash
make convert-orderbook-2025-listed-robust
```

To retry only dates whose parquet output is missing:

```bash
make repair-orderbook-2025
```

Raw archives are temporary unless explicitly retained. Processed daily parquet files are stored
under:

```text
data/processed/bybit/orderbook/BTCUSDT/
```

Raw and processed market data are excluded from Git.

## Dataset Readiness

Validate 2025 coverage, schema consistency, timestamp ordering, duplicate timestamps, best
bid/ask validity, spread, mid-price, and imbalance bounds:

```bash
make check-orderbook-ready
```

The command prints a terminal summary and writes the small local report:

```text
data/processed/bybit/orderbook/orderbook_ready_2025.json
```
