from pathlib import Path

import pytest

from src.direct_temporal_cv_v1.config import load_experiment_config
from src.direct_temporal_cv_v1.registry import build_adapters


@pytest.mark.parametrize("name,model_id", [("ett_direct.yaml", "ett_direct"), ("tcn_direct.yaml", "tcn_direct")])
def test_neural_extension_configs_are_contract_checkable(name: str, model_id: str) -> None:
    config = load_experiment_config(Path("configs/direct_temporal_cv_v1") / name)
    adapter = build_adapters(config.enabled_models)[0]
    assert adapter.model_id == model_id
    assert adapter.validate_config(config.raw["models"][model_id]).model_id == model_id


def test_neural_extensions_fail_fast_until_filled() -> None:
    for model_id in ("ett_direct", "tcn_direct"):
        adapter = build_adapters((model_id,))[0]
        assert adapter.model_id == model_id


def test_btyd_config_is_separate_from_baseline() -> None:
    config = load_experiment_config(Path("configs/direct_temporal_cv_v1/catboost_btyd_pending.yaml"))
    assert config.raw["features"]["btyd_v1"] is True
    assert config.output_root.name == "direct_cv_catboost_btyd_pending_v1"


def test_btyd_contract_uses_classifier_only_columns() -> None:
    from src.direct_temporal_cv_v1.btyd import DirectBTYDFeatureProvider

    assert DirectBTYDFeatureProvider.allowed_columns == (
        "btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive"
    )
