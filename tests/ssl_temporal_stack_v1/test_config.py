from __future__ import annotations

from copy import deepcopy

import pytest

from src.ssl_temporal_stack_v1.config import LoadedConfig, load_config, resolved_contract, validate_loaded_config


def test_machine_readable_config_matches_frozen_contract() -> None:
    config = load_config()
    resolved = resolved_contract(config)
    assert resolved["experiment_id"] == "SSL_TEMPORAL_STACK_V1"
    assert len(resolved["run_a_anchors"]) == 11
    assert len(resolved["run_b_anchors"]) == 14
    total = sum(
        item["ssl_steps"]
        + item["base_steps"]
        + 3 * (item["specialist_head_steps"] + item["specialist_finetune_steps"])
        for item in resolved["budgets"].values()
    )
    assert total == 18_000


def test_anchor_drift_is_rejected() -> None:
    config = load_config()
    raw = deepcopy(config.raw)
    raw["run_a"]["anchors"] = raw["run_a"]["anchors"][:-1]
    changed = LoadedConfig(config.path, raw, config.sha256, config.train_path, config.cohort_path, config.output_root)
    with pytest.raises(ValueError, match="RUN A anchors"):
        validate_loaded_config(changed)


def test_meta_drift_is_rejected() -> None:
    config = load_config()
    raw = deepcopy(config.raw)
    raw["meta"]["alpha"] = 1.0
    changed = LoadedConfig(config.path, raw, config.sha256, config.train_path, config.cohort_path, config.output_root)
    with pytest.raises(ValueError, match="meta contract"):
        validate_loaded_config(changed)
