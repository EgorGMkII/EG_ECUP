import numpy as np
import torch
from unittest.mock import patch
from pathlib import Path

from datetime import date

import polars as pl

from src.btyd_research_pipeline import extract_full_history_rfm_for_anchor
from src.reference_framework_v1.candidates.btyd import AuditedBTYDClassifierProvider
from src.reference_framework_v1.candidates.residual_mlp import ResidualMLPTransitionBase, StreamingFeatureScaler
from src.reference_framework_v1.candidates.tcn import TCNRecipe, TCNTransitionBase
from src.ssl_temporal_stack_v1.stores import DailyTensorStore


def test_tcn_is_causal_shape_safe_and_reaches_history() -> None:
    model = TCNTransitionBase()
    assert TCNRecipe().receptive_field >= 180
    result = model(torch.zeros(3, 180, 15))
    assert set(result) == {"reactivation_logit", "churn_logit"}
    assert result["reactivation_logit"].shape == (3,)


def test_tcn_accepts_a_right_aligned_shorter_causal_history_view() -> None:
    model = TCNTransitionBase(TCNRecipe(history_days=90))
    result = model(torch.zeros(3, 90, 15))
    assert result["churn_logit"].shape == (3,)


def test_tcn_receptive_field_covers_full_365_day_candidate() -> None:
    model = TCNTransitionBase(TCNRecipe(history_days=365))
    assert model.recipe.receptive_field >= 365
    result = model(torch.zeros(2, 365, 15))
    assert result["reactivation_logit"].shape == (2,)


def test_long_daily_store_preserves_legacy_180_right_aligned_view() -> None:
    values = np.zeros((2, 365, 15), dtype=np.float32)
    values[:, -1, 0] = 7.0
    store = DailyTensorStore(Path("unused"), ("2025-12-15",), 2, stored_history_days=365)
    with patch("src.ssl_temporal_stack_v1.stores.np.load", return_value=values):
        assert store.get("2025-12-15").shape == (2, 180, 15)
        assert store.get("2025-12-15", history_days=365).shape == (2, 365, 15)
        assert store.get("2025-12-15")[0, -1, 0] == 7.0


def test_residual_mlp_and_scaler_are_finite() -> None:
    scaler = StreamingFeatureScaler().partial_fit(np.ones((4, 374)))
    values = scaler.transform(np.full((2, 374), np.nan))
    assert np.isfinite(values).all()
    result = ResidualMLPTransitionBase()(torch.from_numpy(values))
    assert set(result) == {"reactivation_logit", "churn_logit", "amount_z"}


def test_btyd_contract_is_audited_classifier_only() -> None:
    provider = AuditedBTYDClassifierProvider()
    assert provider.feature_names == ("btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive")
    assert all("monetary" not in name and "gmv" not in name for name in provider.feature_names)


def test_btyd_rfm_does_not_read_events_after_anchor() -> None:
    raw = pl.DataFrame({
        "user_id": [1, 1, 1, 2],
        "event_date": [date(2025, 1, 1), date(2025, 1, 10), date(2025, 2, 1), date(2025, 2, 1)],
        "gmv": [10.0, 20.0, 9999.0, 9999.0],
    })
    rfm = extract_full_history_rfm_for_anchor(raw, [1, 2], date(2025, 1, 15))
    assert rfm["btyd_n_purchases"].to_list() == [2, 0]
    assert rfm["btyd_frequency"].to_list() == [1.0, 0.0]
