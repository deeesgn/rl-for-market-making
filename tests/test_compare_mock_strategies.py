from pathlib import Path

import pytest

from scripts.compare_mock_strategies import ensure_model_exists


def test_missing_ppo_model_message_tells_user_to_train_first(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_model.zip"

    with pytest.raises(FileNotFoundError, match="Run `make train-mock` first"):
        ensure_model_exists(missing_model)
