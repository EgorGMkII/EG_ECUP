from pathlib import Path

import pytest

from src.reference_framework_v1.config import load_experiment_config
from src.reference_framework_v1.registry import build_adapters, collect_required_stores


def test_registry_preserves_config_order_and_store_union() -> None:
    adapters = build_adapters(("ett", "tcn", "residual_mlp", "catboost"))
    assert [adapter.model_id for adapter in adapters] == ["ett", "tcn", "residual_mlp", "catboost"]
    assert collect_required_stores(adapters) == frozenset({"frames", "daily", "events"})


def test_baseline_config_is_strictly_loadable() -> None:
    config = load_experiment_config(Path("configs/reference_framework_v1/baselines/post_ny_full.yaml"))
    assert config.stage == "full"
    assert config.enabled_models == ("catboost", "s1", "s2", "ett")


def test_extended_candidate_config_is_strictly_loadable() -> None:
    config = load_experiment_config(Path("configs/reference_framework_v1/baselines/post_ny_tcn_mlp_btyd_full.yaml"))
    assert config.enabled_models == ("catboost", "s1", "s2", "ett", "tcn", "residual_mlp")


def test_candidate_adapters_accept_shared_loss_weights_injection() -> None:
    config = load_experiment_config(Path("configs/reference_framework_v1/experiments/post_ny_ssl_parity_tcn_mlp_btyd_selected_100k.yaml"))
    for adapter in build_adapters(("tcn", "residual_mlp")):
        values = dict(config.raw["models"][adapter.model_id])
        values["loss_weights"] = config.raw["loss_weights"]
        adapter.validate_config(values)
