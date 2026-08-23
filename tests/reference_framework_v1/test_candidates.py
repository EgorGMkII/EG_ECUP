import numpy as np
import torch

import polars as pl

from src.reference_framework_v1.candidates.btyd import LifetimesBTYDFeatureProvider
from src.reference_framework_v1.candidates.residual_mlp import ResidualMLPTransitionBase, StreamingFeatureScaler
from src.reference_framework_v1.candidates.tcn import TCNRecipe, TCNTransitionBase


def test_tcn_is_causal_shape_safe_and_reaches_history() -> None:
    model = TCNTransitionBase()
    assert TCNRecipe().receptive_field >= 180
    result = model(torch.zeros(3, 180, 15))
    assert set(result) == {"reactivation_logit", "churn_logit"}
    assert result["reactivation_logit"].shape == (3,)


def test_residual_mlp_and_scaler_are_finite() -> None:
    scaler = StreamingFeatureScaler().partial_fit(np.ones((4, 374)))
    values = scaler.transform(np.full((2, 374), np.nan))
    assert np.isfinite(values).all()
    result = ResidualMLPTransitionBase()(torch.from_numpy(values))
    assert set(result) == {"reactivation_logit", "churn_logit", "amount_z"}


def test_btyd_features_are_causal_and_finite() -> None:
    frame = pl.DataFrame({"purchase_days_90d": [1, 2, 3, 4] * 4, "days_since_last_order": [90, 50, 20, 5] * 4, "available_history_days": [90] * 16, "gmv_sum_90d": [0.0, 100.0, 300.0, 1000.0] * 4, "will_buy": [1] * 16})
    provider = LifetimesBTYDFeatureProvider().fit([frame])
    transformed = provider.transform(frame)
    assert transformed.columns == list(provider.feature_names)
    assert np.isfinite(transformed.to_numpy()).all()
