"""Direct Delta / Residual Regressor Adapter.

Instead of directly predicting absolute z = log1p(GMV), fits a regressor to predict the
delta / residual against the user's historical 30-day baseline:
    z_base = log1p(gmv_sum_30d)
    delta_target = z_true - z_base
    z_pred = max(0, z_base + predicted_delta)
"""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectDeltaRegressorAdapter(DirectModelAdapter):
    model_id = "direct_delta_regressor"
    requirements = ModelRequirements(tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "baseline_feature",  # "gmv_sum_30d" (default) or "ewma_gmv_tau30"
            "backend",           # "catboost" or "lightgbm"
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "thread_count",
            "random_seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {self.model_id} fields: {sorted(unknown)}")

        values = {
            "baseline_feature": str(raw.get("baseline_feature", "gmv_sum_30d")),
            "backend": str(raw.get("backend", "catboost")),
            "iterations": int(raw.get("iterations", 450)),
            "depth": int(raw.get("depth", 8)),
            "learning_rate": float(raw.get("learning_rate", 0.04)),
            "l2_leaf_reg": float(raw.get("l2_leaf_reg", 5.0)),
            "thread_count": int(raw.get("thread_count", 8)),
            "random_seed": int(raw.get("random_seed", 42)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("DirectDeltaRegressor requires tabular snapshots")

        train_df = context.train_tabular
        val_df = context.validation_tabular
        feature_order = tuple(c for c in train_df.columns if c != "user_id")

        base_feat = config.values["baseline_feature"]
        if base_feat not in train_df.columns:
            raise ValueError(f"Baseline feature {base_feat} not found in train snapshot")

        # Compute z_base
        train_base_gmv = train_df[base_feat].to_numpy().astype(np.float64)
        val_base_gmv = val_df[base_feat].to_numpy().astype(np.float64)

        train_z_base = np.log1p(np.maximum(train_base_gmv, 0.0))
        val_z_base = np.log1p(np.maximum(val_base_gmv, 0.0))

        y_train_target = context.train_target_z.astype(np.float64)
        delta_train = y_train_target - train_z_base

        x_train = train_df.select(feature_order).to_numpy().astype(np.float32)
        x_val = val_df.select(feature_order).to_numpy().astype(np.float32)

        seed = config.values["random_seed"]
        threads = config.values["thread_count"]
        t0 = time.perf_counter()

        if config.values["backend"] == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                iterations=config.values["iterations"],
                depth=config.values["depth"],
                learning_rate=config.values["learning_rate"],
                l2_leaf_reg=config.values["l2_leaf_reg"],
                loss_function="RMSE",
                thread_count=threads,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(x_train, delta_train)
            pred_delta = model.predict(x_val)
        else:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=config.values["iterations"],
                max_depth=config.values["depth"],
                num_leaves=min(127, 2 ** config.values["depth"] - 1),
                learning_rate=config.values["learning_rate"],
                reg_lambda=config.values["l2_leaf_reg"],
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=50,
                n_jobs=threads,
                random_state=seed,
                verbose=-1,
            )
            model.fit(x_train, delta_train)
            pred_delta = model.predict(x_val)

        pred_z = np.clip(val_z_base + pred_delta, 0.0, 15.0)
        fit_dur = time.perf_counter() - t0

        report = {
            "model_id": self.model_id,
            "fold_id": context.fold.fold_id,
            "baseline_feature": base_feat,
            "backend": config.values["backend"],
            "delta_mean_train": float(delta_train.mean()),
            "delta_mean_pred": float(pred_delta.mean()),
            "elapsed_seconds": fit_dur,
        }

        return FoldPrediction(self.model_id, np.asarray(context.users), pred_z, report)
