from __future__ import annotations

from datetime import date, timedelta

from src.ssl_temporal_stack_v1.contract import EXPERIMENT, load_feature_order, validate_experiment


def test_frozen_contract() -> None:
    validate_experiment()
    assert len(EXPERIMENT.run_a_anchors) == 11
    assert len(EXPERIMENT.run_b_anchors) == 14
    assert EXPERIMENT.run_b_anchors[:11] == EXPERIMENT.run_a_anchors
    assert sum(value.total_steps for value in EXPERIMENT.budgets.values()) == 18_000


def test_last_run_b_target_ends_on_validation_anchor() -> None:
    target_end = date.fromisoformat(EXPERIMENT.run_b_anchors[-1]) + timedelta(days=30)
    assert target_end == date.fromisoformat(EXPERIMENT.validation_anchor)


def test_catboost_feature_contract() -> None:
    features = load_feature_order()
    assert len(features) == 374
    assert len(set(features)) == 374
    assert not any(name.startswith("btyd_") for name in features)
