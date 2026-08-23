from __future__ import annotations

from datetime import date

import polars as pl

import numpy as np
import pytest

from src.ssl_temporal_stack_v1.stores import (
    build_state_labels,
    required_anchors,
    validate_catboost_feature_values,
)


def test_store_union_has_exactly_fifteen_unique_anchors() -> None:
    anchors = required_anchors()
    assert len(anchors) == 15
    assert len(set(anchors)) == 15
    assert anchors == tuple(sorted(anchors))
    assert anchors[-1] == "2026-01-14"


def test_state_labels_use_past_for_state_and_future_for_target() -> None:
    raw = pl.DataFrame({
        "user_id": [1, 1, 1, 2, 2],
        "event_date": [
            date(2025, 1, 1), date(2025, 3, 31), date(2025, 4, 1),
            date(2025, 3, 31), date(2025, 5, 1),
        ],
        "gmv": [100.0, 10.0, 20.0, 0.0, 999.0],
    })
    labels = build_state_labels(raw, [1, 2, 3], "2025-03-31")
    assert labels["was_active"].to_list() == [1, 0, 0]
    assert labels["will_buy"].to_list() == [1, 0, 0]
    assert labels["future_gmv_30d"].to_list() == [20.0, 0.0, 0.0]


def test_only_verified_shape_features_may_contain_nan() -> None:
    frame = pl.DataFrame({
        "ts_gmv_skewness_90d": [np.nan, 0.0],
        "ts_gmv_kurtosis_90d": [np.nan, 1.0],
        "other": [1.0, 2.0],
    })
    counts = validate_catboost_feature_values(
        frame, tuple(frame.columns), anchor="smoke"
    )
    assert counts == {"ts_gmv_skewness_90d": 1, "ts_gmv_kurtosis_90d": 1}


def test_unexpected_nan_is_fatal() -> None:
    frame = pl.DataFrame({"other": [np.nan]})
    with pytest.raises(RuntimeError, match="Unexpected NaN"):
        validate_catboost_feature_values(frame, ("other",), anchor="smoke")
