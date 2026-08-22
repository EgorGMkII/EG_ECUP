"""Module for training, validating, and evaluating the Stage 1 CatBoost Classifier P(target > 0)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import catboost as cb
import numpy as np
import polars as pl
import shap
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, roc_auc_score

from src.data import FEATURES_DIR, read_fold


def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """Extracts feature columns excluding non-feature metadata and targets."""
    return [c for c in df.columns if c not in ["anchor_date", "user_id", "target"]]


def train_single_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
    use_gpu: bool = False,
) -> Tuple[cb.CatBoostClassifier, Dict[str, float]]:
    """Trains a CatBoostClassifier on train set and evaluates on validation set.

    Args:
        X_train: Training feature matrix.
        y_train: Binary training target vector (0 or 1).
        X_val: Validation feature matrix.
        y_val: Binary validation target vector (0 or 1).
        params: Optional dict of hyperparameters.
        use_gpu: Whether to attempt GPU acceleration.

    Returns:
        Tuple of (trained_model, dict_of_val_metrics).
    """
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 50,
        "thread_count": -1,
    }

    if params is not None:
        default_params.update(params)

    train_pool = cb.Pool(X_train, y_train)
    val_pool = cb.Pool(X_val, y_val)

    if use_gpu:
        try:
            gpu_params = default_params.copy()
            gpu_params["task_type"] = "GPU"
            gpu_params["gpu_ram_part"] = 0.5
            model = cb.CatBoostClassifier(**gpu_params)
            model.fit(train_pool, eval_set=val_pool)
        except Exception as e:
            print(f"[Warning] GPU training failed with error: {e}. Falling back to CPU...")
            cpu_params = default_params.copy()
            cpu_params["task_type"] = "CPU"
            model = cb.CatBoostClassifier(**cpu_params)
            model.fit(train_pool, eval_set=val_pool)
    else:
        cpu_params = default_params.copy()
        cpu_params["task_type"] = "CPU"
        model = cb.CatBoostClassifier(**cpu_params)
        model.fit(train_pool, eval_set=val_pool)

    # Validation predictions
    val_probas = model.predict_proba(val_pool)[:, 1]

    auc_score = roc_auc_score(y_val, val_probas)
    logloss_val = log_loss(y_val, val_probas)
    brier_val = brier_score_loss(y_val, val_probas)

    # Compute PR-AUC (Average Precision)
    precisions, recalls, _ = precision_recall_curve(y_val, val_probas)
    pr_auc_score = np.trapz(precisions[::-1], recalls[::-1])

    metrics = {
        "ROC_AUC": round(float(auc_score), 4),
        "PR_AUC": round(float(pr_auc_score), 4),
        "Logloss": round(float(logloss_val), 4),
        "Brier_Score": round(float(brier_val), 4),
        "Best_Iteration": int(model.get_best_iteration()),
    }

    return model, metrics


def evaluate_classifier_time_cv(
    features_dir: Union[str, Path] = FEATURES_DIR,
    n_folds: int = 4,
    params: Optional[Dict] = None,
    use_gpu: bool = False,
) -> Tuple[cb.CatBoostClassifier, pl.DataFrame, pl.DataFrame]:
    """Runs rolling window Time-CV evaluation across time folds.

    Train folds grow progressively (e.g., Train: fold_00..fold_02 -> Val: fold_03).

    Args:
        features_dir: Directory containing parquet feature folds.
        n_folds: Number of time-CV folds.
        params: Model hyperparameters.
        use_gpu: Enable GPU training.

    Returns:
        Tuple of (latest_trained_model, cv_metrics_dataframe, oof_predictions_dataframe).
    """
    fold_dfs = [read_fold(features_dir, f"fold_{i:02d}") for i in range(n_folds)]
    feature_cols = get_feature_columns(fold_dfs[0])

    cv_results = []
    oof_parts = []
    latest_model = None

    # Train on folds 0..k-1, evaluate on fold k
    for k in range(1, n_folds):
        val_df = fold_dfs[k]
        val_name = f"fold_{k:02d}"

        # Combine past folds for training
        train_df = pl.concat(fold_dfs[:k])

        X_tr = train_df.select(feature_cols).to_numpy()
        y_tr = (train_df["target"].to_numpy() > 0).astype(int)

        X_val = val_df.select(feature_cols).to_numpy()
        y_val = (val_df["target"].to_numpy() > 0).astype(int)

        print(f"\n--- Training Classifier for Validation Fold {val_name} (Train: {len(X_tr):,} rows, Val: {len(X_val):,} rows) ---")
        model, metrics = train_single_classifier(X_tr, y_tr, X_val, y_val, params=params, use_gpu=use_gpu)
        metrics["val_fold"] = val_name
        metrics["train_rows"] = len(X_tr)
        cv_results.append(metrics)

        latest_model = model

        # Out of fold predictions
        val_probas = model.predict_proba(X_val)[:, 1]
        oof_df = pl.DataFrame({
            "anchor_date": val_df["anchor_date"],
            "user_id": val_df["user_id"],
            "target": val_df["target"],
            "target_bin": y_val,
            "pred_proba": val_probas,
        })
        oof_parts.append(oof_df)

    cv_summary_df = pl.DataFrame(cv_results)
    oof_all_df = pl.concat(oof_parts) if oof_parts else pl.DataFrame()

    return latest_model, cv_summary_df, oof_all_df


def get_catboost_feature_importance(
    model: cb.CatBoostClassifier, feature_names: List[str]
) -> pl.DataFrame:
    """Extracts native CatBoost Feature Importance scores."""
    importances = model.get_feature_importance()
    imp_df = pl.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort("importance", descending=True)
    return imp_df


def compute_shap_values(
    model: cb.CatBoostClassifier,
    X_sample: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, shap.Explanation]:
    """Computes SHAP values for a given feature matrix sample."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    explanation = shap.Explanation(
        values=shap_vals,
        data=X_sample,
        feature_names=feature_names,
    )
    return shap_vals, explanation
