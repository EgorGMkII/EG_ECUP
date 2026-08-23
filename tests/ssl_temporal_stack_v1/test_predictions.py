from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.ssl_temporal_stack_v1.predictions import PREDICTION_COLUMNS, bank_arrays, make_prediction_bank


def _predictions(rows: int) -> dict[str, np.ndarray]:
    return {name: np.full(rows, index / 10, dtype=np.float64) for index, name in enumerate(PREDICTION_COLUMNS)}


def test_prediction_bank_has_frozen_order_and_alignment() -> None:
    frame = pl.DataFrame({
        "user_id": [3, 1], "was_active": [0, 1], "will_buy": [1, 0],
        "future_gmv_30d": [10.0, 0.0], "z_target": [np.log1p(10.0), 0.0],
    })
    bank = make_prediction_bank(frame, "2025-12-15", _predictions(2))
    assert bank["user_id"].to_list() == [3, 1]
    assert tuple(bank.columns[-12:]) == PREDICTION_COLUMNS
    arrays = bank_arrays(bank)
    assert arrays["react"].shape == (2, 4)
    assert arrays["churn"].shape == (2, 4)
    assert arrays["amount"].shape == (2, 4)


def test_prediction_bank_rejects_unknown_column() -> None:
    frame = pl.DataFrame({
        "user_id": [1], "was_active": [0], "will_buy": [0],
        "future_gmv_30d": [0.0], "z_target": [0.0],
    })
    predictions = _predictions(1)
    predictions["unknown"] = np.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        make_prediction_bank(frame, "2025-12-15", predictions)
