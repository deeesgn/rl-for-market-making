from pathlib import Path

import yaml

from scripts.run_mock_regime_stress import load_regime_config, run_stress


def test_load_regime_config_reads_name_and_overrides(tmp_path: Path) -> None:
    regime_path = tmp_path / "regime.yaml"
    regime_path.write_text(
        yaml.safe_dump(
            {
                "name": "test_regime",
                "overrides": {"price_volatility": 0.01, "drift": 0.001},
            }
        ),
        encoding="utf-8",
    )

    name, overrides = load_regime_config(regime_path)

    assert name == "test_regime"
    assert overrides == {"price_volatility": 0.01, "drift": 0.001}


def test_run_stress_skips_missing_ppo_model(tmp_path: Path, capsys) -> None:
    base_config_path = tmp_path / "base.yaml"
    regime_path = tmp_path / "calm.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "max_steps": 2,
                "price_volatility": 0.0,
                "fill_probability": 0.0,
                "no_quote_penalty": 0.001,
            }
        ),
        encoding="utf-8",
    )
    regime_path.write_text(
        yaml.safe_dump({"name": "calm", "overrides": {"drift": 0.0}}),
        encoding="utf-8",
    )

    results = run_stress(
        base_config_path=base_config_path,
        regime_paths=[regime_path],
        episodes=1,
        seed=1,
        model_path=tmp_path / "missing_ppo.zip",
    )

    output = capsys.readouterr().out
    assert "skipping PPO" in output
    assert set(results["calm"]) == {"fixed_spread", "inventory_skew"}
