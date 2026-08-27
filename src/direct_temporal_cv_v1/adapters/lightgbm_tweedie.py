"""Direct LightGBM adapter with Tweedie objective for four-fold temporal protocol."""
from __future__ import annotations

import time
from typing import Any, Mapping
import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectLightGBMTweedieAdapter(DirectModelAdapter):
    model_id = "lightgbm_tweedie"
    requirements = ModelRequirements(tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "n_estimators", "num_leaves", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_samples",
            "reg_alpha", "reg_lambda", "n_jobs", "random_state", "verbose",
            "objective", "tweedie_variance_power", "iterations"
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown lightgbm_tweedie fields: {sorted(unknown)}")
        values = {
            "objective": "tweedie",
            "tweedie_variance_power": float(raw.get("tweedie_variance_power", 1.2)),
            "n_estimators": int(raw.get("n_estimators", raw.get("iterations", 500))),
            "num_leaves": int(raw.get("num_leaves", 127)),
            "max_depth": int(raw.get("max_depth", 12)),
            "learning_rate": float(raw.get("learning_rate", 0.03)),
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
            raise ValueError("lightgbm_tweedie requires train and validation tabular snapshots")
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise RuntimeError("lightgbm is required for LightGBM experiments") from error
        train = context.train_tabular
        valid = context.validation_tabular
        feature_order = tuple(c for c in train.columns if c != "user_id")
        if feature_order != tuple(c for c in valid.columns if c != "user_id"):
            raise ValueError("LightGBM train/validation feature order mismatch")
        x_train = train.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_valid = valid.select(feature_order).to_numpy().astype(np.float32, copy=False)
        
        # Tweedie is trained on original Y = exp(Z) - 1 >= 0
        y_train = np.maximum(np.expm1(context.train_target_z), 0.0)
        
        values = dict(config.values)
        started = time.perf_counter()
        model = lgb.LGBMRegressor(**values)
        model.fit(x_train, y_train)
        
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
