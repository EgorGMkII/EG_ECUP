from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.reference_framework_v1.base import RunContext
from src.reference_framework_v1.candidate_adapters import fit_predict_residual_mlp, fit_predict_tcn


class _Frames:
    feature_names = tuple(f"f{i}" for i in range(374))

    def __init__(self) -> None:
        labels = {
            "user_id": [1, 2, 3, 4], "was_active": [0, 0, 1, 1],
            "will_buy": [0, 1, 0, 1], "future_gmv_30d": [0.0, 1.0, 0.0, 2.0],
            "z_target": [0.0, np.log(2.0), 0.0, np.log(3.0)],
        }
        self.frame = pl.DataFrame({**{"user_id": labels.pop("user_id")}, **{name: [0.1, 0.2, 0.3, 0.4] for name in self.feature_names}, **labels})

    def get(self, anchor: str) -> pl.DataFrame:
        return self.frame


class _Daily:
    def get(self, anchor: str) -> np.ndarray:
        return np.zeros((4, 180, 15), dtype=np.float32)


class _Stores:
    frames = _Frames()
    daily = _Daily()


def _context() -> RunContext:
    return RunContext("TEST", ("2025-01-01",), "2025-01-15", (1, 2, 3, 4), _Stores(), torch.device("cpu"), 42, Path("."))


def _stage() -> dict[str, object]:
    return {"steps": 1, "learning_rate": 1e-3, "scheduler": "constant", "warmup_steps": 0, "weight_decay": 0.0}


def test_tcn_adapter_returns_two_finite_columns() -> None:
    stage = _stage()
    result = fit_predict_tcn(_context(), {"batch_size": 4, "channels": 8, "dropout": 0.0, "base": stage, "specialists": {"react": {"H": stage, "F": stage}, "churn": {"H": stage, "F": stage}}})
    assert set(result.predictions) == {"tcn_react_logit", "tcn_churn_logit"}
    assert all(values.shape == (4,) and np.isfinite(values).all() for values in result.predictions.values())


def test_mlp_adapter_returns_three_finite_columns() -> None:
    stage = _stage()
    result = fit_predict_residual_mlp(_context(), {"batch_size": 4, "hidden": 16, "blocks": 2, "dropout": 0.0, "base": stage, "specialists": {task: {"H": stage, "F": stage} for task in ("react", "churn", "amount")}})
    assert set(result.predictions) == {"mlp_react_logit", "mlp_churn_logit", "mlp_amount_z"}
    assert all(values.shape == (4,) and np.isfinite(values).all() for values in result.predictions.values())
