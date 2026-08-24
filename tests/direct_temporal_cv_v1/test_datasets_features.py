from datetime import date

import numpy as np
import polars as pl

from src.direct_temporal_cv_v1.datasets import build_target_z
from src.direct_temporal_cv_v1.features import WINDOWS, SparseAggregateFeatureProvider


def _raw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 11), date(2025, 1, 12)],
            "user_id": [10, 10, 20, 10],
            "gmv": [1.0, 2.0, 9.0, 1000.0],
            "gmv_search": [0.0, 1.0, 9.0, 1000.0],
            "gmv_cat": [1.0, 1.0, 0.0, 0.0],
            "searches": [1, 2, 1, 10],
            "to_ord": [0, 1, 1, 1],
            "to_cart": [1, 0, 1, 1],
        }
    )


def test_target_is_inclusive_and_template_aligned() -> None:
    values = build_target_z(_raw(), [10, 20, 30], date(2025, 1, 1), date(2025, 1, 11))
    np.testing.assert_allclose(values, [np.log1p(3.0), np.log1p(9.0), 0.0])


def test_features_are_causal_and_sparse() -> None:
    result = SparseAggregateFeatureProvider().build_pair(_raw(), [10, 20, 30], date(2025, 1, 2), date(2025, 1, 2))
    assert result.train.height == 3
    assert result.feature_order == tuple(c for c in result.train.columns if c != "user_id")
    assert len(WINDOWS) == 7
    assert result.train.filter(pl.col("user_id") == 10)["gmv_sum_7d"][0] == 3.0
    # The future 1,000 GMV row is not visible at the earlier anchor.
    assert result.train.filter(pl.col("user_id") == 10)["gmv_sum_7d"][0] < 1000.0
    assert result.train.filter(pl.col("user_id") == 30)["gmv_sum_7d"][0] == 0.0
    assert "target" not in result.feature_order
    assert "will_buy_30d" not in result.feature_order
