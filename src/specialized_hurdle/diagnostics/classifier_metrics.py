"""Comprehensive Classifier Metrics & Calibration Diagnostics."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression


def compute_classifier_metrics(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> Dict[str, Any]:
    """Computes all required classifier ranking, calibration, and diagnostic metrics."""
    y_true_bin = (y_true > 0).astype(int)
    y_prob_clip = np.clip(y_prob, eps, 1.0 - eps)

    p_null = float(np.mean(y_true_bin))
    null_brier = p_null * (1.0 - p_null)
    null_logloss = float(- (p_null * np.log(p_null + eps) + (1.0 - p_null) * np.log(1.0 - p_null + eps)))

    roc_auc = float(roc_auc_score(y_true_bin, y_prob_clip)) if len(np.unique(y_true_bin)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true_bin, y_prob_clip)) if len(np.unique(y_true_bin)) > 1 else p_null
    brier = float(brier_score_loss(y_true_bin, y_prob_clip))
    ll = float(log_loss(y_true_bin, y_prob_clip))

    # Calibration slope & intercept via univariate logistic regression
    logits = np.log(y_prob_clip / (1.0 - y_prob_clip)).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=True, C=1e5)
    lr.fit(logits, y_true_bin)
    calib_slope = float(lr.coef_[0][0])
    calib_intercept = float(lr.intercept_[0])

    # Expected Calibration Error (ECE) with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(y_prob_clip, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true_bin[mask])
            bin_conf = np.mean(y_prob_clip[mask])
            ece += (np.sum(mask) / len(y_true_bin)) * np.abs(bin_acc - bin_conf)

    quantiles = np.percentile(y_prob_clip, [1, 10, 50, 90, 99])

    return {
        "n_samples": int(len(y_true_bin)),
        "positive_rate": p_null,
        "prediction_mean": float(np.mean(y_prob_clip)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": ll,
        "ece": float(ece),
        "calib_slope": calib_slope,
        "calib_intercept": calib_intercept,
        "null_roc_auc": 0.5,
        "null_pr_auc": p_null,
        "null_brier": null_brier,
        "null_logloss": null_logloss,
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
    }
