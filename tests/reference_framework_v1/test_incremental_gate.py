import polars as pl
import pytest

from src.reference_framework_v1.screening import _replace


def _bank(prediction_name: str, prediction: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [11, 12],
            "anchor": ["2026-01-14", "2026-01-14"],
            "was_active": [0, 1],
            "will_buy": [0, 1],
            "future_gmv_30d": [0.0, 10.0],
            "z_target": [0.0, 2.3978952728],
            prediction_name: [prediction, prediction],
        }
    )


def test_incremental_gate_appends_new_independent_direct_channel() -> None:
    base = _bank("cb_react_logit", 0.1)
    candidate = _bank("cb_direct_z", 0.2)

    merged = _replace(base, candidate, ("cb_direct_z",), "2026-01-14")

    assert merged.columns[-2:] == ["cb_react_logit", "cb_direct_z"]
    assert merged["cb_react_logit"].to_list() == [0.1, 0.1]
    assert merged["cb_direct_z"].to_list() == [0.2, 0.2]


def test_incremental_gate_rejects_partial_channel_replacement() -> None:
    base = _bank("cb_react_logit", 0.1)
    candidate = _bank("cb_react_logit", 0.2).with_columns(pl.lit(0.3).alias("cb_churn_logit"))

    with pytest.raises(ValueError, match="partially"):
        _replace(base, candidate, ("cb_react_logit", "cb_churn_logit"), "2026-01-14")
