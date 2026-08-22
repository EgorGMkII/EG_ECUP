"""Separate Reactivation and Churn CatBoost Classifiers."""

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from src.transitions.metrics import evaluate_classifier_metrics


def train_reactivation_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    iterations: int = 600,
    depth: int = 6,
    learning_rate: float = 0.065,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
    verbose: Union[bool, int] = False,
) -> Tuple[CatBoostClassifier, np.ndarray]:
    """Trains a specialized CatBoost Classifier for dormant users (past_buyer_30d == 0)."""
    clf = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        thread_count=4,
        verbose=verbose,
    )

    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
    clf.fit(X_tr, y_tr, eval_set=eval_set, early_stopping_rounds=60 if eval_set is not None else None)

    p_val = clf.predict_proba(X_val)[:, 1] if X_val is not None else None
    return clf, p_val


def train_churn_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    iterations: int = 600,
    depth: int = 6,
    learning_rate: float = 0.065,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
    verbose: Union[bool, int] = False,
) -> Tuple[CatBoostClassifier, np.ndarray]:
    """Trains a specialized CatBoost Classifier for recent buyers (past_buyer_30d == 1) to predict churn (1 - future_buyer)."""
    clf = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        thread_count=4,
        verbose=verbose,
    )

    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
    clf.fit(X_tr, y_tr, eval_set=eval_set, early_stopping_rounds=60 if eval_set is not None else None)

    p_val = clf.predict_proba(X_val)[:, 1] if X_val is not None else None
    return clf, p_val
