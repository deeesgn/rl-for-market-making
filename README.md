# RL for Market Making

Docker-ready skeleton for a quant research project on RL market making with inventory risk control.

The eventual goal is to train an RL agent that places bid and ask quotes on historical Bybit order book data with `hftbacktest`, compare it with fixed-spread and Avellaneda-Stoikov baselines, and report PnL/risk metrics.

This version intentionally contains no RL training, Bybit ingestion, `hftbacktest` integration, real order book data, or production trading logic. It does include a mock Gymnasium environment and two simple rule-based baselines for early mechanics checks.

## Project Layout

```text
.
├── configs/
├── data/
│   └── sample/
├── models/
├── reports/
│   └── figures/
├── scripts/
│   ├── run_baselines.py
│   ├── run_mock_env.py
│   └── smoke_test.py
├── src/
│   └── rl_mm/
├── tests/
│   ├── test_baselines.py
│   ├── test_import.py
│   └── test_mock_env.py
├── Dockerfile
├── Makefile
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Local Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
make smoke
make lint
```

## Docker Quickstart

```bash
docker build -t rl-mm .
docker run --rm rl-mm
```

Run the test suite in the image:

```bash
docker run --rm rl-mm python -m pytest
```

## Mock Market Environment

The first runnable market-making component is a tiny Gymnasium environment that simulates a random-walk mid price, probabilistic bid/ask fills, inventory, cash, PnL, and an inventory-aware reward.

```bash
make run-mock
```

The mock config lives at `configs/env_mock.yaml`. It is only for testing mechanics before adding Bybit data, `hftbacktest`, baselines, or RL training.

## Mock Baselines

Two simple rule-based strategies can be compared over multiple seeded mock episodes:

- `FixedSpreadStrategy`: always sends action `2`, the medium symmetric quote.
- `InventorySkewStrategy`: sends action `4` to reduce long inventory, action `5` to reduce short inventory, and action `2` otherwise.

```bash
make baselines
```

This runs `scripts/run_baselines.py` with 20 episodes and seed 42, then prints mean and standard deviation for total PnL, total reward, max absolute inventory, final inventory, and number of steps. You can also run it directly:

```bash
python scripts/run_baselines.py --episodes 20 --seed 42 --config configs/env_mock.yaml
```

## Notes

- `src/rl_mm/` is the importable Python package.
- `data/sample/` is for tiny checked-in sample fixtures only.
- `models/` and `reports/figures/` are placeholder output directories.
- Generated data, reports, model files, caches, and virtual environments are ignored by Git.
