"""Literal CPU CatBoost adapter for the direct temporal protocol."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectCatBoostAdapter(DirectModelAdapter):
    model_id = "catboost_direct"
    requirements = ModelRequirements(tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function", "thread_count", "random_seed", "verbose"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown catboost_direct fields: {sorted(unknown)}")
        values = {
            "iterations": int(raw.get("iterations", 300)),
            "depth": int(raw.get("depth", 8)),
            "learning_rate": float(raw.get("learning_rate", 0.05)),
            "l2_leaf_reg": float(raw.get("l2_leaf_reg", 5.0)),
            "loss_function": str(raw.get("loss_function", "RMSE")),
            "thread_count": int(raw.get("thread_count", 8)),
            "random_seed": int(raw.get("random_seed", 42)),
            "verbose": raw.get("verbose", False),
        }
        if values["iterations"] <= 0 or values["depth"] <= 0 or values["learning_rate"] <= 0 or values["thread_count"] <= 0:
            raise ValueError("CatBoost iterations/depth/lr/thread_count must be positive")
        if values["loss_function"].upper() != "RMSE":
            raise ValueError("direct baseline is pinned to CatBoost RMSE")
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("catboost_direct requires train and validation tabular snapshots")
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("catboost is required for direct CatBoost experiments") from error
        train = context.train_tabular
        valid = context.validation_tabular
        feature_order = tuple(c for c in train.columns if c != "user_id")
        if feature_order != tuple(c for c in valid.columns if c != "user_id"):
            raise ValueError("CatBoost train/validation feature order mismatch")
        x_train = train.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_valid = valid.select(feature_order).to_numpy().astype(np.float32, copy=False)
        if x_train.shape[0] != context.train_target_z.shape[0] or x_valid.shape[0] != context.users.shape[0]:
            raise ValueError("CatBoost rows are not aligned with targets/template users")
        if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
            raise ValueError("CatBoost feature matrix contains non-finite values")
        values = dict(config.values)
        values.setdefault("allow_writing_files", False)
        started = time.perf_counter()
        model = CatBoostRegressor(**values)
        model.fit(x_train, context.train_target_z, verbose=False)
        prediction = np.asarray(model.predict(x_valid), dtype=np.float64)
        elapsed = time.perf_counter() - started
        if prediction.shape != (context.users.shape[0],) or not np.isfinite(prediction).all():
            raise ValueError("CatBoost prediction shape/non-finite check failed")
        report = {
            "model_id": self.model_id,
            "fold_id": context.fold.fold_id,
            "recipe": values,
            "feature_count": len(feature_order),
            "train_rows": int(x_train.shape[0]),
            "validation_rows": int(x_valid.shape[0]),
            "elapsed_seconds": elapsed,
            "fresh_model_per_fold": True,
        }
        return FoldPrediction(self.model_id, np.asarray(context.users), prediction, report)
