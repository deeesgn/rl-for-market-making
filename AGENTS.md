# AGENTS.md

## Project mission

This repository is a research project for reinforcement-learning market making on
Bybit BTCUSDT Perpetual.

The goal is to build a reproducible and honest market-making experiment using
real orderbook and trade data.

Optimize out-of-sample risk-adjusted trading performance. Never manufacture
positive results by changing accounting, execution assumptions, rewards, or test
data after seeing results.

## Active project paths

Primary active files:

- `src/rl_mm/env/real_orderbook_env.py`
- `src/rl_mm/strategies/fixed_spread.py`
- `src/rl_mm/strategies/inventory_skew.py`
- `scripts/run_real_experiment.py`
- `tests/test_real_orderbook_env.py`
- `tests/test_real_pipeline.py`
- `Makefile`

The mock environment is secondary. Do not modify mock code unless the task
explicitly requires it.

Local data and models are ignored by Git:

- `data/raw/`
- `data/processed/bybit/orderbook/`
- `data/processed/bybit/trades/`
- `models/`

Never delete, overwrite, move, or commit local parquet data or model files unless
the user explicitly requests it.

## Repository size and structure

Keep the repository small and easy to understand.

Prefer modifying an existing file over adding a new file.

Do not add any of the following unless the user explicitly approves it:

- additional scripts
- additional config files
- notebooks
- CSV reports
- generated reports
- tracked experiment outputs
- new folders
- duplicated utilities
- one-off debugging tools

Do not split a small implementation across many files.

Remove temporary code after it is no longer needed.

Do not change README for every internal implementation detail. Update it only
when the public workflow or commands change.

## Data and experimental integrity

Current dataset:

- symbol: BTCUSDT Perpetual
- year: 2025
- orderbook: Top-10 snapshots at 1-second frequency
- trades: aggregated at 1-second frequency
- one parquet file per day

Use chronological splitting only:

- train: 2025-01-01 through 2025-10-01
- validation: 2025-10-02 through 2025-11-06
- test: 2025-11-07 through 2025-12-31

Rules:

- Never shuffle dates across splits.
- Never use test data for training, tuning, early stopping, model selection, or
  threshold selection.
- Use identical deterministic evaluation windows for all compared strategies.
- Select models and parameters using validation data only.
- Access test data only after final validation selection.
- Never alter the test procedure after seeing test results.
- Avoid all future-data leakage.
- Observations at time `t` may contain only information available at or before
  time `t`.

## Trading simulation rules

Use real trade-driven passive fills and the existing queue-aware execution model.

Do not replace real fills with price-crossing fills unless explicitly requested.

Maintain:

- partial fills
- queue-ahead logic
- maximum inventory enforcement
- maker-fee accounting
- cash accounting
- mark-to-market equity
- atomic and reproducible evaluation

The maker fee must remain configurable.

Do not hardcode zero fees globally. Zero fees are allowed only as an explicitly
labelled research experiment.

PnL must always be based on:

`equity = cash + inventory * current_mid_price`

Evaluation PnL must never include artificial rewards.

Training reward may contain only clearly documented risk-control terms such as
inventory penalties.

Never add:

- fake profit
- spread-capture bonuses presented as PnL
- fill bonuses
- bonuses solely for trading
- hidden reward shaping
- double-counted fees

Do not guarantee that a strategy will be profitable.

## Current research direction

The current active research direction is baseline-guided residual SAC.

The agent should improve a strong adaptive market-making baseline rather than
learn the entire strategy from random actions.

Current selected architecture:

- residual continuous actions
- selective-narrow quoting
- behaviour-cloning warm start
- real trade-driven fills
- queue fraction fixed by the experiment
- configurable maker fee
- validation checkpointing
- early stopping
- baseline anchoring
- separate seeds 42, 100, and 200

For the current early-stopping experiment:

- validate every 50,000 timesteps
- maximum training budget: 700,000 timesteps
- early-stopping patience: 3 validation checks
- minimum score improvement: 0.01
- save the best validation checkpoint
- never assume the final checkpoint is the best checkpoint

Model ranking must use validation metrics and must include both profitability and
risk.

Do not tune execution assumptions to improve model performance.

## Development workflow

Before editing:

1. Read this file.
2. Inspect `git status`.
3. Read only the files relevant to the task.
4. Give a brief implementation plan.
5. Check whether the requested behavior already exists.

Implementation rules:

- Make complete, meaningful changes rather than many tiny incremental changes.
- Keep diffs focused.
- Preserve existing working behavior unless the task explicitly changes it.
- Do not rewrite unrelated code.
- Do not introduce unnecessary abstractions.
- Use readable Python with straightforward control flow.
- Add dependencies only when necessary and explain why.
- Reuse existing helpers instead of creating duplicates.
- Keep random seeds explicit.
- Keep long experiments resumable.
- Save models and state atomically.
- Never overwrite a better checkpoint with a worse checkpoint.

## Commands Codex may run automatically

Codex may run:

```bash
make test
make lint
make smoke
git diff --check