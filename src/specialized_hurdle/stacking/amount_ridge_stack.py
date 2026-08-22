"""Positive-Only Amount Ridge Stack with Interaction Features."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class AmountRidgeStackResult:
    ridge_model: Ridge
    alpha: float
    best_mse: float
    model_names: List[str]
    max_train_target_z: float


def fit_amount_ridge_stack(
    cond_z_matrix: np.ndarray,
    was_active_arr: np.ndarray,
    y_amount_true: np.ndarray,
    model_names: List[str],
    alphas: List[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> AmountRidgeStackResult:
    """Fits positive-only Ridge regression on conditional magnitude targets.

    Features: [cond_z_1, ..., cond_z_M, was_active, was_active * cond_z_1, ...]
    """
    n_samples, n_models = cond_z_matrix.shape
    act_col = was_active_arr.reshape(-1, 1).astype(float)
    interaction_cols = cond_z_matrix * act_col

    X_feat = np.hstack([cond_z_matrix, act_col, interaction_cols])

    best_alpha = 1.0
    best_mse = 999.0
    best_model = None

    # Cross-validation over alpha
    for a in alphas:
        model = Ridge(alpha=a, positive=False, fit_intercept=True)
        model.fit(X_feat, y_amount_true)
        preds = model.predict(X_feat)
        mse = float(np.mean((preds - y_amount_true) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = a
            best_model = model

    max_target_z = float(np.max(y_amount_true))

    return AmountRidgeStackResult(
        ridge_model=best_model,
        alpha=best_alpha,
        best_mse=best_mse,
        model_names=model_names,
        max_train_target_z=max_target_z,
    )


def predict_amount_ridge_stack(
    stack: AmountRidgeStackResult,
    cond_z_matrix: np.ndarray,
    was_active_arr: np.ndarray,
) -> np.ndarray:
    """Predicts non-negative conditional magnitudes conditional_z."""
    act_col = was_active_arr.reshape(-1, 1).astype(float)
    interaction_cols = cond_z_matrix * act_col
    X_feat = np.hstack([cond_z_matrix, act_col, interaction_cols])

    preds = stack.ridge_model.predict(X_feat)
    # Clip to valid range [0, max_train_target_z]
    return np.clip(preds, 0.0, stack.max_train_target_z)
