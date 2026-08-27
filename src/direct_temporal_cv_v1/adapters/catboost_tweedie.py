"""Direct CatBoost adapter using Tweedie loss function for four-fold temporal protocol."""
from __future__ import annotations

import time
from typing import Any, Mapping
import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectCatBoostTweedieAdapter(DirectModelAdapter):
    model_id = "catboost_tweedie"
    requirements = ModelRequirements(tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function", "thread_count", "random_seed", "verbose"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown catboost_tweedie fields: {sorted(unknown)}")
        values = {
            "iterations": int(raw.get("iterations", 500)),
            "depth": int(raw.get("depth", 7)),
            "learning_rate": float(raw.get("learning_rate", 0.03)),
            "l2_leaf_reg": float(raw.get("l2_leaf_reg", 5.0)),
            "loss_function": str(raw.get("loss_function", "Tweedie:variance_power=1.3")),
            "thread_count": int(raw.get("thread_count", 8)),
            "random_seed": int(raw.get("random_seed", 42)),
            "verbose": raw.get("verbose", False),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("catboost_tweedie requires train and validation tabular snapshots")
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("catboost is required for CatBoost experiments") from error
        train = context.train_tabular
        valid = context.validation_tabular
        feature_order = tuple(c for c in train.columns if c != "user_id")
        if feature_order != tuple(c for c in valid.columns if c != "user_id"):
            raise ValueError("CatBoost train/validation feature order mismatch")
        x_train = train.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_valid = valid.select(feature_order).to_numpy().astype(np.float32, copy=False)
        
        # Tweedie is trained on original Y = exp(Z) - 1 >= 0
        y_train = np.maximum(np.expm1(context.train_target_z), 0.0)
        
        values = dict(config.values)
        values.setdefault("allow_writing_files", False)
        started = time.perf_counter()
        model = CatBoostRegressor(**values)
        model.fit(x_train, y_train, verbose=False)
        
        y_pred = np.maximum(model.predict(x_valid), 0.0)
        prediction = np.asarray(np.log1p(y_pred), dtype=np.float64)
        elapsed = time.perf_counter() - started
        
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
