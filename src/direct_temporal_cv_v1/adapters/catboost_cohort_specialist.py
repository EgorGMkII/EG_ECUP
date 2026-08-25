"""Cohort-specialist CatBoost: active-user hurdle in z-space.

For was_active=1 (gmv_sum_{W}d > 0, W = activity_window_days):
    pred_z = P(will_buy | active) * E[z | active, will_buy=1]

For was_active=0:
    pred_z = direct CatBoost regressor (same recipe as baseline)

Both components stay in z = log1p(GMV) space.  The decomposition
E[z] = P(buy) * E[z|buy] is MSE-optimal for RMSLE.

activity_window_days controls what counts as "was_active":
  90 (default) — purchased in 90-day window before anchor
  60 — purchased in 60-day window
  30 — purchased in 30-day window (stricter: recent buyers only)
All three windows are already computed by SparseAggregateFeatureProvider.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class CatBoostCohortSpecialistAdapter(DirectModelAdapter):
    model_id = "catboost_cohort_specialist"
    requirements = ModelRequirements(tabular_features=True)

    _VALID_WINDOWS = (30, 60, 90)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "churn_iterations", "churn_depth", "churn_learning_rate", "churn_l2_leaf_reg",
            "amount_iterations", "amount_depth", "amount_learning_rate", "amount_l2_leaf_reg",
            "inactive_iterations", "inactive_depth", "inactive_learning_rate", "inactive_l2_leaf_reg",
            "thread_count", "random_seed", "activity_window_days",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {self.model_id} fields: {sorted(unknown)}")
        activity_window = int(raw.get("activity_window_days", 90))
        if activity_window not in self._VALID_WINDOWS:
            raise ValueError(
                f"activity_window_days must be one of {self._VALID_WINDOWS}, got {activity_window}"
            )
        values = {
            "activity_window_days": activity_window,
            "churn_iterations": int(raw.get("churn_iterations", 500)),
            "churn_depth": int(raw.get("churn_depth", 6)),
            "churn_learning_rate": float(raw.get("churn_learning_rate", 0.04)),
            "churn_l2_leaf_reg": float(raw.get("churn_l2_leaf_reg", 3.0)),
            "amount_iterations": int(raw.get("amount_iterations", 300)),
            "amount_depth": int(raw.get("amount_depth", 8)),
            "amount_learning_rate": float(raw.get("amount_learning_rate", 0.05)),
            "amount_l2_leaf_reg": float(raw.get("amount_l2_leaf_reg", 5.0)),
            "inactive_iterations": int(raw.get("inactive_iterations", 300)),
            "inactive_depth": int(raw.get("inactive_depth", 8)),
            "inactive_learning_rate": float(raw.get("inactive_learning_rate", 0.05)),
            "inactive_l2_leaf_reg": float(raw.get("inactive_l2_leaf_reg", 5.0)),
            "thread_count": int(raw.get("thread_count", 8)),
            "random_seed": int(raw.get("random_seed", 42)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("cohort specialist requires tabular snapshots")
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("catboost is required") from error

        train_tab = context.train_tabular
        val_tab = context.validation_tabular
        feature_order = tuple(c for c in train_tab.columns if c != "user_id")

        activity_window = config.values["activity_window_days"]
        activity_feature = f"gmv_sum_{activity_window}d"
        if activity_feature not in feature_order:
            raise ValueError(
                f"Feature '{activity_feature}' not in tabular columns. "
                f"Available gmv_sum columns: {[c for c in feature_order if 'gmv_sum' in c]}"
            )

        x_train = train_tab.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_val = val_tab.select(feature_order).to_numpy().astype(np.float32, copy=False)

        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
            raise ValueError("Feature matrix contains non-finite values")

        activity_idx = feature_order.index(activity_feature)
        train_active = x_train[:, activity_idx] > 0
        val_active = x_val[:, activity_idx] > 0
        print(f"  [COHORT] activity_feature={activity_feature} activity_window={activity_window}d", flush=True)

        train_z = context.train_target_z
        train_will_buy = (train_z > 0).astype(np.int32)

        v = config.values
        seed, threads = v["random_seed"], v["thread_count"]

        prediction_z = np.zeros(len(context.users), dtype=np.float64)
        reports: dict[str, Any] = {}
        started = time.perf_counter()

        # ── Active cohort: churn classifier ──────────────────────────────
        x_train_active = x_train[train_active]
        y_cls_target = train_will_buy[train_active]
        n_active_train = int(train_active.sum())
        n_inactive_train_prelim = int((~train_active).sum())
        buy_rate = float(y_cls_target.mean())

        print(
            f"  [COHORT] Active train N={n_active_train} ({n_active_train/(n_active_train+n_inactive_train_prelim):.1%}), "
            f"Inactive train N={n_inactive_train_prelim}, buy_rate(active)={buy_rate:.4f}",
            flush=True,
        )

        cb_churn = CatBoostClassifier(
            iterations=v["churn_iterations"],
            depth=v["churn_depth"],
            learning_rate=v["churn_learning_rate"],
            l2_leaf_reg=v["churn_l2_leaf_reg"],
            loss_function="Logloss",
            thread_count=threads,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        cb_churn.fit(x_train_active, y_cls_target, verbose=False)

        # In-sample AUC (diagnostic only)
        from sklearn.metrics import roc_auc_score
        train_proba = cb_churn.predict_proba(x_train_active)[:, 1]
        train_auc = float(roc_auc_score(y_cls_target, train_proba))
        print(f"  [COHORT] Churn classifier train AUC={train_auc:.6f}", flush=True)

        val_active_idx = np.where(val_active)[0]
        p_buy_active = cb_churn.predict_proba(x_val[val_active])[:, 1]

        reports["churn_classifier"] = {
            "iterations": v["churn_iterations"],
            "depth": v["churn_depth"],
            "lr": v["churn_learning_rate"],
            "l2_leaf_reg": v["churn_l2_leaf_reg"],
            "train_n": n_active_train,
            "train_buy_rate": buy_rate,
            "train_auc": train_auc,
            "val_active_n": int(val_active.sum()),
        }

        # ── Active cohort: conditional amount regressor ──────────────────
        active_buyers_mask = train_active & (train_will_buy == 1)
        x_train_buyers = x_train[active_buyers_mask]
        z_train_buyers = train_z[active_buyers_mask]
        n_buyers = int(active_buyers_mask.sum())

        print(f"  [COHORT] Active buyers train N={n_buyers}", flush=True)

        cb_amount = CatBoostRegressor(
            iterations=v["amount_iterations"],
            depth=v["amount_depth"],
            learning_rate=v["amount_learning_rate"],
            l2_leaf_reg=v["amount_l2_leaf_reg"],
            loss_function="RMSE",
            thread_count=threads,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        cb_amount.fit(x_train_buyers, z_train_buyers, verbose=False)

        cond_z_active = np.maximum(cb_amount.predict(x_val[val_active]), 0.0)

        # Combine in z-space: E[z] = P(buy) * E[z|buy]
        prediction_z[val_active_idx] = p_buy_active * cond_z_active

        reports["amount_regressor"] = {
            "iterations": v["amount_iterations"],
            "depth": v["amount_depth"],
            "lr": v["amount_learning_rate"],
            "train_n": n_buyers,
            "train_mean_z": float(z_train_buyers.mean()),
            "train_std_z": float(z_train_buyers.std()),
        }

        # ── Inactive cohort: direct regressor ────────────────────────────
        x_train_inactive = x_train[~train_active]
        z_train_inactive = train_z[~train_active]
        val_inactive_idx = np.where(~val_active)[0]
        n_inactive_train = int((~train_active).sum())

        print(f"  [COHORT] Inactive train N={n_inactive_train}", flush=True)

        cb_inactive = CatBoostRegressor(
            iterations=v["inactive_iterations"],
            depth=v["inactive_depth"],
            learning_rate=v["inactive_learning_rate"],
            l2_leaf_reg=v["inactive_l2_leaf_reg"],
            loss_function="RMSE",
            thread_count=threads,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        cb_inactive.fit(x_train_inactive, z_train_inactive, verbose=False)
        prediction_z[val_inactive_idx] = np.maximum(cb_inactive.predict(x_val[~val_active]), 0.0)

        elapsed = time.perf_counter() - started

        reports["inactive_regressor"] = {
            "iterations": v["inactive_iterations"],
            "train_n": n_inactive_train,
            "val_inactive_n": int(val_inactive_idx.shape[0]),
        }

        # ── Validation AUC on held-out fold ──────────────────────────────
        val_z = context.validation_target_z
        val_will_buy_active = (val_z[val_active_idx] > 0).astype(np.int32)
        if 0 < val_will_buy_active.sum() < len(val_will_buy_active):
            val_auc = float(roc_auc_score(val_will_buy_active, p_buy_active))
            reports["churn_classifier"]["val_auc"] = val_auc
            print(f"  [COHORT] Churn classifier val AUC={val_auc:.6f}", flush=True)

        if not np.isfinite(prediction_z).all():
            raise ValueError("Cohort specialist produced non-finite predictions")

        report = {
            "model_id": self.model_id,
            "fold_id": context.fold.fold_id,
            "elapsed_seconds": elapsed,
            "fresh_model_per_fold": True,
            "feature_count": len(feature_order),
            "cohort_reports": reports,
        }
        return FoldPrediction(self.model_id, np.asarray(context.users), prediction_z, report)
