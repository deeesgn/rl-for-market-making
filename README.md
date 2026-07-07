# RL for Market Making

Docker-ready skeleton for a quant research project on RL market making with inventory risk control.

The eventual goal is to train an RL agent that places bid and ask quotes on historical Bybit order book data with `hftbacktest`, compare it with fixed-spread and Avellaneda-Stoikov baselines, and report PnL/risk metrics.

This initial version intentionally contains no RL, Bybit, `hftbacktest`, baseline, backtest, or trading logic.

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
│   └── smoke_test.py
├── src/
│   └── rl_mm/
├── tests/
│   └── test_import.py
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

## Notes

- `src/rl_mm/` is the importable Python package.
- `data/sample/` is for tiny checked-in sample fixtures only.
- `models/` and `reports/figures/` are placeholder output directories.
- Generated data, reports, model files, caches, and virtual environments are ignored by Git.
