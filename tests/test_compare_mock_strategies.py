from pathlib import Path

import pytest

from rl_mm.backtest.metrics import aggregate_episode_metrics, compute_episode_metrics
from scripts.compare_mock_strategies import ensure_model_exists, print_action_distribution


def test_missing_ppo_model_message_tells_user_to_train_first(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_model.zip"

    with pytest.raises(FileNotFoundError, match="Run `make train-mock` first"):
        ensure_model_exists(missing_model)


def test_print_action_distribution(capsys) -> None:
    episode = compute_episode_metrics(
        rewards=[0.0, 0.0, 0.0],
        pnls=[0.0, 0.0, 0.0],
        inventories=[0.0, 0.0, 0.0, 0.0],
        quoted=[False, True, True],
        actions=[0, 2, 4],
        bid_fills=[False, True, False],
        ask_fills=[False, False, True],
    )

    print_action_distribution({"test_strategy": aggregate_episode_metrics([episode])})

    output = capsys.readouterr().out
    assert "Action distribution" in output
    assert "strategy" in output
    assert "action_0" in output
    assert "action_5" in output
    assert "test_strategy" in output
