"""Direct LightGBM Regressor with Tweedie objective for zero-inflated continuous GMV prediction."""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from src.direct_temporal_cv_v1.contracts import DirectModelAdapter, EvaluationResult, PredictionSummary, Recipe, ValidationMetrics
from src.direct_temporal_cv_v1.metrics import compute_metrics, compute_prediction_summary

class DirectLightGBMTweedieAdapter(DirectModelAdapter):
    adapter_id = "lightgbm_tweedie"

    def train_and_evaluate(
        self,
        x_train: np.ndarray,
        z_train: np.ndarray,
        x_val: np.ndarray,
        z_val: np.ndarray,
        feature_order: tuple[str, ...],
        recipe: Recipe,
        fold_id: str,
    ) -> EvaluationResult:
        # Tweedie optimizes on Y = exp(Z) - 1
        y_train = np.maximum(np.expm1(z_train), 0.0)
        
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.2,
            "n_estimators": recipe.iterations or 500,
            "num_leaves": 127,
            "max_depth": 12,
            "learning_rate": recipe.learning_rate or 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_samples": 50,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "n_jobs": recipe.thread_count,
            "random_state": recipe.random_seed,
            "verbose": -1,
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(x_train, y_train)
        
        y_pred = np.maximum(model.predict(x_val), 0.0)
        z_pred = np.log1p(y_pred)
        
        metrics = compute_metrics(z_val, z_pred)
        summary = compute_prediction_summary(z_pred)
        
        return EvaluationResult(
            model_id=self.adapter_id,
            fold_id=fold_id,
            recipe=recipe,
            feature_order=feature_order,
            metrics=metrics,
            summary=summary,
            z_pred=z_pred,
            fitted_model=model,
        )
