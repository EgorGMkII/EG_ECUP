from pathlib import Path

import pytest

from src.reference_framework_v1.config import load_experiment_config
from src.reference_framework_v1.registry import build_adapters, collect_required_stores


def test_registry_preserves_config_order_and_store_union() -> None:
    adapters = build_adapters(("ett", "catboost"))
    assert [adapter.model_id for adapter in adapters] == ["ett", "catboost"]
    assert collect_required_stores(adapters) == frozenset({"frames", "events"})


def test_baseline_config_is_strictly_loadable() -> None:
    config = load_experiment_config(Path("configs/reference_framework_v1/baselines/post_ny_full.yaml"))
    assert config.stage == "full"
    assert config.enabled_models == ("catboost", "s1", "s2", "ett")
