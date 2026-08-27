"""Direct Frequency Specialists Adapter.

Splits the cohort into Frequent buyers (order_days_90d >= min_orders) and Dormant/Rare buyers,
and fits two independent direct GBDT regressors in log1p(GMV) space WITHOUT any hurdle zeroing.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectFrequencySpecialistAdapter(DirectModelAdapter):
    model_id = "direct_frequency_specialist"
    requirements = ModelRequirements(tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "min_orders_90d",
            "frequent_backend",  # "catboost" or "lightgbm"
            "dormant_backend",   # "catboost" or "lightgbm"
            "frequent_iterations",
            "frequent_depth",
            "frequent_lr",
            "frequent_l2",
            "dormant_iterations",
            "dormant_depth",
            "dormant_lr",
            "dormant_l2",
            "thread_count",
            "random_seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {self.model_id} fields: {sorted(unknown)}")

        values = {
            "min_orders_90d": int(raw.get("min_orders_90d", 2)),
            "frequent_backend": str(raw.get("frequent_backend", "catboost")),
            "dormant_backend": str(raw.get("dormant_backend", "lightgbm")),
            "frequent_iterations": int(raw.get("frequent_iterations", 450)),
            "frequent_depth": int(raw.get("frequent_depth", 8)),
            "frequent_lr": float(raw.get("frequent_lr", 0.04)),
            "frequent_l2": float(raw.get("frequent_l2", 5.0)),
            "dormant_iterations": int(raw.get("dormant_iterations", 450)),
            "dormant_depth": int(raw.get("dormant_depth", 7)),
            "dormant_lr": float(raw.get("dormant_lr", 0.035)),
            "dormant_l2": float(raw.get("dormant_l2", 5.0)),
            "thread_count": int(raw.get("thread_count", 8)),
            "random_seed": int(raw.get("random_seed", 42)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("DirectFrequencySpecialist requires tabular snapshots")

        train_df = context.train_tabular
        val_df = context.validation_tabular
        feature_order = tuple(c for c in train_df.columns if c != "user_id")

        split_col = "order_days_90d"
        if split_col not in train_df.columns:
            raise ValueError(f"Required split column {split_col} not found in features")

        min_orders = config.values["min_orders_90d"]
        seed = config.values["random_seed"]
        threads = config.values["thread_count"]

        train_orders = train_df[split_col].to_numpy()
        val_orders = val_df[split_col].to_numpy()

        train_frequent_mask = train_orders >= min_orders
        val_frequent_mask = val_orders >= min_orders

        x_train = train_df.select(feature_order).to_numpy().astype(np.float32)
        y_train = context.train_target_z.astype(np.float32)

        x_val = val_df.select(feature_order).to_numpy().astype(np.float32)

        pred_z = np.zeros(len(x_val), dtype=np.float64)
        t0 = time.perf_counter()

        # 1. Fit Specialist for Frequent buyers
        x_train_freq = x_train[train_frequent_mask]
        y_train_freq = y_train[train_frequent_mask]
        x_val_freq = x_val[val_frequent_mask]

        if config.values["frequent_backend"] == "catboost":
            from catboost import CatBoostRegressor
            cb_freq = CatBoostRegressor(
                iterations=config.values["frequent_iterations"],
                depth=config.values["frequent_depth"],
                learning_rate=config.values["frequent_lr"],
                l2_leaf_reg=config.values["frequent_l2"],
                loss_function="RMSE",
                thread_count=threads,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            cb_freq.fit(x_train_freq, y_train_freq)
            if len(x_val_freq) > 0:
                pred_z[val_frequent_mask] = cb_freq.predict(x_val_freq)
        else:
            import lightgbm as lgb
            lgb_freq = lgb.LGBMRegressor(
                n_estimators=config.values["frequent_iterations"],
                max_depth=config.values["frequent_depth"],
                num_leaves=2 ** config.values["frequent_depth"] - 1,
                learning_rate=config.values["frequent_lr"],
                reg_lambda=config.values["frequent_l2"],
                n_jobs=threads,
                random_state=seed,
                verbose=-1,
            )
            lgb_freq.fit(x_train_freq, y_train_freq)
            if len(x_val_freq) > 0:
                pred_z[val_frequent_mask] = lgb_freq.predict(x_val_freq)

        # 2. Fit Specialist for Dormant / Rare buyers
        train_dormant_mask = ~train_frequent_mask
        val_dormant_mask = ~val_frequent_mask

        x_train_dorm = x_train[train_dormant_mask]
        y_train_dorm = y_train[train_dormant_mask]
        x_val_dorm = x_val[val_dormant_mask]

        if config.values["dormant_backend"] == "catboost":
            from catboost import CatBoostRegressor
            cb_dorm = CatBoostRegressor(
                iterations=config.values["dormant_iterations"],
                depth=config.values["dormant_depth"],
                learning_rate=config.values["dormant_lr"],
                l2_leaf_reg=config.values["dormant_l2"],
                loss_function="RMSE",
                thread_count=threads,
                random_seed=seed + 1,
                verbose=False,
                allow_writing_files=False,
            )
            cb_dorm.fit(x_train_dorm, y_train_dorm)
            if len(x_val_dorm) > 0:
                pred_z[val_dormant_mask] = cb_dorm.predict(x_val_dorm)
        else:
            import lightgbm as lgb
            lgb_dorm = lgb.LGBMRegressor(
                n_estimators=config.values["dormant_iterations"],
                max_depth=config.values["dormant_depth"],
                num_leaves=min(127, 2 ** config.values["dormant_depth"] - 1),
                learning_rate=config.values["dormant_lr"],
                reg_lambda=config.values["dormant_l2"],
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=50,
                n_jobs=threads,
                random_state=seed + 1,
                verbose=-1,
            )
            lgb_dorm.fit(x_train_dorm, y_train_dorm)
            if len(x_val_dorm) > 0:
                pred_z[val_dormant_mask] = lgb_dorm.predict(x_val_dorm)

        pred_z = np.clip(pred_z, 0.0, 15.0)
        fit_dur = time.perf_counter() - t0

        report = {
            "model_id": self.model_id,
            "fold_id": context.fold.fold_id,
            "min_orders_90d": min_orders,
            "frequent_share_train": float(train_frequent_mask.mean()),
            "dormant_share_train": float(train_dormant_mask.mean()),
            "frequent_backend": config.values["frequent_backend"],
            "dormant_backend": config.values["dormant_backend"],
            "elapsed_seconds": fit_dur,
        }

        return FoldPrediction(self.model_id, np.asarray(context.users), pred_z, report)
