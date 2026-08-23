import numpy as np
import torch

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
