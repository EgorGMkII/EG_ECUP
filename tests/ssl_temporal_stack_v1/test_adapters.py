from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.ssl_temporal_stack_v1.adapters import (
    expected_columns_for_adapter,
    predict_dense_specialist,
    resolved_catboost_params,
)
from src.ssl_temporal_stack_v1.config import load_config
from src.ssl_temporal_stack_v1.models import GRUBackbone, Specialist


class TinyDaily:
    def get(self, anchor: str) -> np.ndarray:
        return np.zeros((5, 180, 15), dtype=np.float32)


def test_adapter_prediction_columns_are_frozen() -> None:
    assert expected_columns_for_adapter("catboost") == (
        "cb_react_logit", "cb_churn_logit", "cb_amount_z"
    )
    assert expected_columns_for_adapter("ett") == (
        "ett_react_logit", "ett_churn_logit", "ett_amount_z"
    )


def test_catboost_parameters_are_explicit_gpu_contract() -> None:
    params = resolved_catboost_params(load_config(), seed=123)
    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"
    assert params["iterations"] == 1500
    assert params["random_seed"] == 123
    assert params["allow_writing_files"] is False


def test_catboost_smoke_override_removes_gpu_device_argument() -> None:
    params = resolved_catboost_params(
        load_config(), seed=123, overrides={"task_type": "CPU", "iterations": 2, "verbose": False}
    )
    assert params["task_type"] == "CPU"
    assert params["iterations"] == 2
    assert "devices" not in params


def test_dense_specialist_inference_preserves_row_count() -> None:
    model = Specialist(GRUBackbone(), "react", "s1")
    stores = SimpleNamespace(daily=TinyDaily())
    prediction = predict_dense_specialist(
        model, stores, "2025-12-15", device=torch.device("cpu"), batch_size=2
    )
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
