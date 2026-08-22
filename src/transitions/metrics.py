"""Diagnostic Metrics & Exact SSE / MSE Decomposition by Transition States."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def evaluate_classifier_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    task_name: str = "Classifier",
) -> Dict[str, Union[float, Dict]]:
    """Calculates ROC-AUC, PR-AUC, LogLoss, Brier, Calibration Error, Decile Lift and Confusion Matrix."""
    y_true = np.asarray(y_true, dtype=np.int32)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    p_pred_clipped = np.clip(p_pred, 1e-7, 1.0 - 1e-7)

    auc = float(roc_auc_score(y_true, p_pred))
    pr_auc = float(average_precision_score(y_true, p_pred))
    loss = float(log_loss(y_true, p_pred_clipped))
    brier = float(brier_score_loss(y_true, p_pred))

    # Calibration error (ECE - Expected Calibration Error over 10 equal bins)
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(p_pred, bins) - 1
    ece = 0.0
    for i in range(10):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(p_pred[mask]))
            bin_weight = float(np.sum(mask) / len(y_true))
            ece += bin_weight * abs(bin_acc - bin_conf)

    # Decile lift (top 10% vs baseline positive rate)
    base_rate = float(np.mean(y_true))
    top_10_threshold = np.percentile(p_pred, 90)
    top_10_mask = p_pred >= top_10_threshold
    top_10_rate = float(np.mean(y_true[top_10_mask])) if np.sum(top_10_mask) > 0 else base_rate
    lift_top_10 = float(top_10_rate / (base_rate + 1e-8))

    # Confusion matrix at multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.5]
    cm_dict = {}
    for t in thresholds:
        cm = confusion_matrix(y_true, (p_pred >= t).astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_dict[f"tau_{t}"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

    return {
        "task_name": task_name,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "log_loss": loss,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "base_rate": base_rate,
        "top_10_lift": lift_top_10,
        "confusion_matrices": cm_dict,
    }


def decompose_mse_by_transitions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    past_buyer_30d: np.ndarray,
) -> Dict[str, Union[float, Dict, pl.DataFrame]]:
    """Calculates exact SSE / MSE / RMSLE decomposition across the 4 transition states."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    past_buyer_30d = np.asarray(past_buyer_30d, dtype=np.int32)

    z_true = np.log1p(np.maximum(y_true, 0.0))
    z_pred = np.log1p(np.maximum(y_pred, 0.0))

    sq_err = (z_pred - z_true) ** 2
    total_sse = float(np.sum(sq_err))
    total_mse = float(np.mean(sq_err))
    total_rmsle = float(np.sqrt(total_mse))

    # Oracle Future-Zero (pred = 0 if target = 0)
    z_oracle = np.where(y_true == 0, 0.0, z_pred)
    sq_err_oracle = (z_oracle - z_true) ** 2
    oracle_sse = float(np.sum(sq_err_oracle))
    oracle_mse = float(np.mean(sq_err_oracle))
    oracle_rmsle = float(np.sqrt(oracle_mse))

    past_act = past_buyer_30d == 1
    fut_act = y_true > 0

    states = [
        ("0 -> 0 (Stable Sleep)", (~past_act) & (~fut_act)),
        ("0 -> >0 (Reactivation)", (~past_act) & (fut_act)),
        (">0 -> 0 (Churn)", (past_act) & (~fut_act)),
        (">0 -> >0 (Retention)", (past_act) & (fut_act)),
    ]

    decomp_rows = []
    for name, mask in states:
        count = int(np.sum(mask))
        share_users = float(count / len(y_true) * 100.0)
        sse_g = float(np.sum(sq_err[mask]))
        mse_g = float(np.mean(sq_err[mask])) if count > 0 else 0.0
        rmsle_g = float(np.sqrt(mse_g)) if count > 0 else 0.0
        share_sse = float(sse_g / total_sse * 100.0) if total_sse > 0 else 0.0

        mean_z_true = float(np.mean(z_true[mask])) if count > 0 else 0.0
        mean_z_pred = float(np.mean(z_pred[mask])) if count > 0 else 0.0
        mean_target_rub = float(np.mean(y_true[mask])) if count > 0 else 0.0
        mean_pred_rub = float(np.mean(y_pred[mask])) if count > 0 else 0.0

        decomp_rows.append({
            "State": name,
            "Count": count,
            "Share_Users_Pct": share_users,
            "Share_Total_MSE_Pct": share_sse,
            "Group_RMSLE": rmsle_g,
            "Group_MSE": mse_g,
            "Group_SSE": sse_g,
            "Mean_z_true": mean_z_true,
            "Mean_z_pred": mean_z_pred,
            "Mean_Target_Rub": mean_target_rub,
            "Mean_Pred_Rub": mean_pred_rub,
        })

    df_table = pl.DataFrame(decomp_rows)

    return {
        "total_rmsle": total_rmsle,
        "total_mse": total_mse,
        "total_sse": total_sse,
        "oracle_future_zero_rmsle": oracle_rmsle,
        "oracle_future_zero_mse": oracle_mse,
        "oracle_future_zero_gap": float(oracle_rmsle - total_rmsle),
        "decomposition_table": df_table,
    }
