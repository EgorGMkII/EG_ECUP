"""Specialized CatBoost Models for Reactivation, Churn, and Amount."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor


def train_catboost_reactivation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
) -> CatBoostClassifier:
    """Trains CatBoostClassifier strictly on inactive rows (was_active == False)."""
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_seed,
        verbose=100,
        task_type="CPU",
    )
    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=100, verbose=False)
    return model


def train_catboost_churn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
) -> CatBoostClassifier:
    """Trains CatBoostClassifier strictly on active rows (was_active == True)."""
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_seed,
        verbose=100,
        task_type="CPU",
    )
    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=100, verbose=False)
    return model


def train_catboost_amount(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
) -> CatBoostRegressor:
    """Trains CatBoostRegressor strictly on positive spenders (future_gmv_30d > 0)."""
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_seed,
        verbose=100,
        task_type="CPU",
    )
    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=100, verbose=False)
    return model
