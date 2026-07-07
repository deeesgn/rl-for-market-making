from pathlib import Path

import yaml

from scripts.compare_mock_regime_models import compare_all_regimes, warn_missing_model


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_warn_missing_model_skips_with_clear_warning(tmp_path: Path, capsys) -> None:
    assert warn_missing_model("static_ppo", tmp_path / "missing.zip") is False

    output = capsys.readouterr().out
    assert "Warning:" in output
    assert "static_ppo" in output
    assert "skipping static_ppo" in output


def test_compare_all_regimes_skips_missing_models(tmp_path: Path, capsys) -> None:
    base_config_path = tmp_path / "base.yaml"
    regime_path = tmp_path / "calm.yaml"
    write_yaml(
        base_config_path,
        {
            "max_steps": 2,
            "price_volatility": 0.0,
            "fill_probability": 0.0,
            "no_quote_penalty": 0.001,
        },
    )
    write_yaml(regime_path, {"name": "calm", "overrides": {"drift": 0.0}})

    results = compare_all_regimes(
        base_config_path=base_config_path,
        regime_paths=[regime_path],
        episodes=1,
        seed=1,
        static_model_path=tmp_path / "missing_static.zip",
        randomized_model_path=tmp_path / "missing_randomized.zip",
    )

    output = capsys.readouterr().out
    assert "skipping static_ppo" in output
    assert "skipping randomized_ppo" in output
    assert set(results["calm"]) == {"fixed_spread", "inventory_skew"}
