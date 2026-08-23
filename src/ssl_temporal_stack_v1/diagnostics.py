"""Sealed-holdout validation diagnostics for SSL V1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def _safe_auc(target: np.ndarray, prediction: np.ndarray) -> float | None:
    return float(roc_auc_score(target, prediction)) if np.unique(target).size == 2 else None


def _binary_report(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    probability = np.clip(probability.astype(np.float64), 1e-12, 1 - 1e-12)
    return {
        "rows": int(len(target)),
        "positive_rate": float(np.mean(target)) if len(target) else None,
        "mean_probability": float(np.mean(probability)) if len(target) else None,
        "auc": _safe_auc(target, probability) if len(target) else None,
        "logloss": float(log_loss(target, probability, labels=[0, 1])) if len(target) else None,
        "brier": float(np.mean(np.square(probability - target))) if len(target) else None,
    }

def build_validation_report(
    *,
    target_z: np.ndarray,
    was_active: np.ndarray,
    will_buy: np.ndarray,
    components: dict[str, np.ndarray],
    validation_anchor: str,
    job_id: str | None,
    commit_sha: str,
    config_sha256: str,
    bank_sha256: str,
    meta_sha256: str,
) -> dict[str, Any]:
    target_z = np.asarray(target_z, dtype=np.float64)
    prediction_z = np.asarray(components["prediction_z"], dtype=np.float64)
    was_active = np.asarray(was_active, dtype=np.int8)
    will_buy = np.asarray(will_buy, dtype=np.int8)
    if not (
        target_z.shape == prediction_z.shape == was_active.shape == will_buy.shape
        and target_z.ndim == 1
    ):
        raise ValueError("Validation arrays are not aligned")
    if not np.isfinite(target_z).all() or not np.isfinite(prediction_z).all():
        raise ValueError("Validation target/prediction contains non-finite values")
    squared_error = np.square(prediction_z - target_z)
    transitions: dict[str, Any] = {}
    for previous in (0, 1):
        for future in (0, 1):
            name = f"{previous}{future}"
            mask = (was_active == previous) & (will_buy == future)
            transitions[name] = {
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "mse_logspace": float(squared_error[mask].mean()) if mask.any() else None,
                "rmsle": float(np.sqrt(squared_error[mask].mean())) if mask.any() else None,
                "target_mean_z": float(target_z[mask].mean()) if mask.any() else None,
                "prediction_mean_z": float(prediction_z[mask].mean()) if mask.any() else None,
                "share_total_squared_error": (
                    float(squared_error[mask].sum() / squared_error.sum())
                    if mask.any() and squared_error.sum() > 0 else 0.0
                ),
            }
    inactive = was_active == 0
    active = was_active == 1
    positive = will_buy == 1
    amount_error = np.square(components["conditional_z"][positive] - target_z[positive])
    return {
        "validation_anchor": validation_anchor,
        "rows": int(len(target_z)),
        "mse_logspace": float(squared_error.mean()),
        "rmsle": float(np.sqrt(squared_error.mean())),
        "target_mean_z": float(target_z.mean()),
        "prediction_mean_z": float(prediction_z.mean()),
        "target_buy_rate": float(will_buy.mean()),
        "predicted_buy_rate": float(components["p_buy"].mean()),
        "transitions": transitions,
        "react": _binary_report(will_buy[inactive], components["p_react"][inactive]),
        "churn": _binary_report(1 - will_buy[active], components["p_churn"][active]),
        "amount": {
            "rows": int(positive.sum()),
            "rmse_conditional_z": float(np.sqrt(amount_error.mean())) if positive.any() else None,
            "target_mean_z": float(target_z[positive].mean()) if positive.any() else None,
            "prediction_mean_conditional_z": (
                float(components["conditional_z"][positive].mean()) if positive.any() else None
            ),
        },
        "provenance": {
            "job_id": job_id,
            "commit_sha": commit_sha,
            "config_sha256": config_sha256,
            "prediction_bank_sha256": bank_sha256,
            "frozen_meta_sha256": meta_sha256,
        },
    }
