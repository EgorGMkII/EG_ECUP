"""Direct CatBoost Regressor using Tweedie loss function for zero-inflated distributions."""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from src.direct_temporal_cv_v1.contracts import DirectModelAdapter, EvaluationResult, PredictionSummary, Recipe, ValidationMetrics
from src.direct_temporal_cv_v1.metrics import compute_metrics, compute_prediction_summary

class DirectCatBoostTweedieAdapter(DirectModelAdapter):
    adapter_id = "catboost_tweedie"

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
        # Tweedie optimizes directly on continuous non-negative Y = exp(Z) - 1
        y_train = np.maximum(np.expm1(z_train), 0.0)
        
        params = {
            "iterations": recipe.iterations or 500,
            "depth": recipe.depth or 8,
            "learning_rate": recipe.learning_rate or 0.03,
            "l2_leaf_reg": recipe.l2_leaf_reg or 5.0,
            "loss_function": "Tweedie:variance_power=1.3",
            "thread_count": recipe.thread_count,
            "random_seed": recipe.random_seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        
        model = CatBoostRegressor(**params)
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
