"""Direct LightGBM adapter with leaf-wise tree growth for four-fold temporal protocol."""
from __future__ import annotations

import time
from typing import Any, Mapping
import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectLightGBMAdapter(DirectModelAdapter):
    model_id = "lightgbm_direct"
    requirements = ModelRequirements(tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "n_estimators", "num_leaves", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_samples",
            "reg_alpha", "reg_lambda", "n_jobs", "random_state", "verbose"
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown lightgbm_direct fields: {sorted(unknown)}")
        values = {
            "n_estimators": int(raw.get("n_estimators", 400)),
            "num_leaves": int(raw.get("num_leaves", 127)),
            "max_depth": int(raw.get("max_depth", 12)),
            "learning_rate": float(raw.get("learning_rate", 0.04)),
            "subsample": float(raw.get("subsample", 0.8)),
            "colsample_bytree": float(raw.get("colsample_bytree", 0.7)),
            "min_child_samples": int(raw.get("min_child_samples", 50)),
            "reg_alpha": float(raw.get("reg_alpha", 1.0)),
            "reg_lambda": float(raw.get("reg_lambda", 5.0)),
            "n_jobs": int(raw.get("n_jobs", 8)),
            "random_state": int(raw.get("random_state", 42)),
            "verbose": int(raw.get("verbose", -1)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("lightgbm_direct requires train and validation tabular snapshots")
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise RuntimeError("lightgbm is required for direct LightGBM experiments") from error

        train = context.train_tabular
        valid = context.validation_tabular
        feature_order = tuple(c for c in train.columns if c != "user_id")
        if feature_order != tuple(c for c in valid.columns if c != "user_id"):
            raise ValueError("LightGBM train/validation feature order mismatch")

        x_train = train.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_valid = valid.select(feature_order).to_numpy().astype(np.float32, copy=False)

        values = dict(config.values)
        started = time.perf_counter()
        model = lgb.LGBMRegressor(**values)
        model.fit(x_train, context.train_target_z)
        prediction = np.asarray(model.predict(x_valid), dtype=np.float64)
        elapsed = time.perf_counter() - started

        if prediction.shape != (context.users.shape[0],) or not np.isfinite(prediction).all():
            raise ValueError("LightGBM prediction shape/non-finite check failed")

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
