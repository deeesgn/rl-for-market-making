# RL for Market Making

A compact quant research project for market-making experiments with inventory risk control.
The current repository contains a deterministic mock Gymnasium environment, two rule-based
baselines, and a listing-driven pipeline for converting Bybit BTCUSDT orderbook archives into
Top-10 parquet data sampled at one second. RL training and `hftbacktest` integration are not part
of the current working path.

## Project Layout

```text
.
├── configs/
│   └── env_mock.yaml
├── data/
│   ├── raw/                         # ignored local downloads
│   └── processed/bybit/orderbook/   # ignored parquet output
├── models/                          # ignored local model output
├── scripts/
│   ├── check_orderbook_ready.py
│   ├── download_convert_orderbook_range.py
│   ├── run_baselines.py
│   ├── run_mock_env.py
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
