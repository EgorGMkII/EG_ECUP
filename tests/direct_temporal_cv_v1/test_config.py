from pathlib import Path

from src.direct_temporal_cv_v1.config import load_experiment_config


def test_baseline_config_is_strict_and_has_four_folds() -> None:
    config = load_experiment_config(Path("configs/direct_temporal_cv_v1/baseline_catboost.yaml"))
    assert config.enabled_models == ("catboost_direct",)
    assert len(config.folds) == 4
