"""Two-Stage Hurdle ML Pipeline: Calibrated Classifier + Conditional Log1p Regressor + Ensembling."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, auc

NON_FEATURE_COLS = {
    "user_id",
    "anchor_date",
    "history_start",
    "history_end",
    "target_start",
    "target_end",
    "available_history_days",
    "target",
    "will_buy_30d",
}


def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """Extracts purely predictive numeric feature columns."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Logarithmic Error (RMSLE)."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred_clipped)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


def train_hurdle_pipeline(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    use_sample_weights: bool = True,
    sample_weights_tr: Optional[np.ndarray] = None,
    clf_params: Optional[Dict] = None,
    reg_params: Optional[Dict] = None,
    use_gpu: bool = False,
) -> Dict:
    """Trains and evaluates full two-stage Hurdle pipeline + direct regression on validation panel."""
    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    print(f"[*] Features count: {len(feature_cols)}")
    X_tr = train_df.select(feature_cols).to_numpy()
    y_tr_target = train_df["target"].to_numpy()
    y_tr_bin = (y_tr_target > 0).astype(int)

    X_val = val_df.select(feature_cols).to_numpy()
    y_val_target = val_df["target"].to_numpy()
    y_val_bin = (y_val_target > 0).astype(int)

    if clf_params is None:
        clf_params = {
            "iterations": 700,
            "learning_rate": 0.04,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0,
        }

    if reg_params is None:
        reg_params = {
            "iterations": 700,
            "learning_rate": 0.04,
            "depth": 6,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": 0,
        }

    # 1. Stage 1: CatBoost Classifier P(target > 0)
    print("  [1/3] Training Stage 1 Classifier...")
    clf = CatBoostClassifier(**clf_params)
    clf.fit(
        X_tr,
        y_tr_bin,
        sample_weight=sample_weights_tr if use_sample_weights else None,
        eval_set=(X_val, y_val_bin),
        early_stopping_rounds=80,
    )
    p_val_raw = clf.predict_proba(X_val)[:, 1]

    # Stage 1 Classifier Metrics
    roc_auc = float(roc_auc_score(y_val_bin, p_val_raw))
    precision, recall, _ = precision_recall_curve(y_val_bin, p_val_raw)
    pr_auc = float(auc(recall, precision))
    brier = float(brier_score_loss(y_val_bin, p_val_raw))
    logloss = float(log_loss(y_val_bin, p_val_raw))

    # 2. Stage 2: CatBoost Regressor E[log1p(GMV) | target > 0]
    print("  [2/3] Training Stage 2 Conditional Regressor (Buyers only)...")
    tr_buyer_mask = y_tr_target > 0
    X_tr_buyers = X_tr[tr_buyer_mask]
    y_tr_buyers_log = np.log1p(y_tr_target[tr_buyer_mask])
    weights_buyers = sample_weights_tr[tr_buyer_mask] if (use_sample_weights and sample_weights_tr is not None) else None

    val_buyer_mask = y_val_target > 0
    X_val_buyers = X_val[val_buyer_mask]
    y_val_buyers_log = np.log1p(y_val_target[val_buyer_mask])

    reg = CatBoostRegressor(**reg_params)
    reg.fit(
        X_tr_buyers,
        y_tr_buyers_log,
        sample_weight=weights_buyers,
        eval_set=(X_val_buyers, y_val_buyers_log),
        early_stopping_rounds=80,
    )
    pred_val_buyers_log = reg.predict(X_val)

    # 3. Direct Baseline Regressor: X -> log1p(target)
    print("  [3/3] Training Direct Baseline Regressor...")
    direct_reg = CatBoostRegressor(**reg_params)
    direct_reg.fit(
        X_tr,
        np.log1p(y_tr_target),
        sample_weight=sample_weights_tr if use_sample_weights else None,
        eval_set=(X_val, np.log1p(y_val_target)),
        early_stopping_rounds=80,
    )
    pred_val_direct_log = direct_reg.predict(X_val)

    # 4. Combining Rules Evaluation (Stage 16)
    # Variant A: z_hat = P * E[log1p(Y)], Y_hat = exp(z_hat) - 1
    z_hat_A = p_val_raw * pred_val_buyers_log
    pred_Y_A = np.clip(np.expm1(z_hat_A), 0, None)
    rmsle_A = compute_rmsle(y_val_target, pred_Y_A)

    # Variant B: Y_hat = P * (exp(E[log1p(Y)]) - 1)
    cond_Y = np.clip(np.expm1(pred_val_buyers_log), 0, None)
    pred_Y_B = p_val_raw * cond_Y
    rmsle_B = compute_rmsle(y_val_target, pred_Y_B)

    # Variant C: Direct Regression
    pred_Y_C = np.clip(np.expm1(pred_val_direct_log), 0, None)
    rmsle_C = compute_rmsle(y_val_target, pred_Y_C)

    # Variant D: Ensemble (50% Variant A + 50% Variant C)
    pred_Y_D = 0.5 * pred_Y_A + 0.5 * pred_Y_C
    rmsle_D = compute_rmsle(y_val_target, pred_Y_D)

    return {
        "models": {"clf": clf, "reg": reg, "direct_reg": direct_reg},
        "feature_cols": feature_cols,
        "metrics": {
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Brier_Score": brier,
            "Logloss": logloss,
            "RMSLE_Variant_A": rmsle_A,
            "RMSLE_Variant_B": rmsle_B,
            "RMSLE_Variant_C_Direct": rmsle_C,
            "RMSLE_Variant_D_Ensemble": rmsle_D,
        },
        "predictions": {
            "p_val": p_val_raw,
            "pred_Y_A": pred_Y_A,
            "pred_Y_B": pred_Y_B,
            "pred_Y_C": pred_Y_C,
            "pred_Y_D": pred_Y_D,
        }
    }
