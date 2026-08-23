"""M-only constrained joint meta optimizer for SSL_TEMPORAL_STACK_V1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .contract import EXPERIMENT
from .predictions import AMOUNT_COLUMNS, CHURN_COLUMNS, REACT_COLUMNS


META_SCHEMA_VERSION = 1
PARAMETER_COUNT = 13


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values.astype(np.float64, copy=False) - mean) / scale


def predict_z(
    bank: dict[str, np.ndarray],
    parameters: np.ndarray,
    amount_mean: np.ndarray,
    amount_scale: np.ndarray,
    *,
    alpha: float = 1.1,
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (PARAMETER_COUNT,):
        raise ValueError(f"Meta parameter shape must be {(PARAMETER_COUNT,)}")
    react = parameters[:4]
    churn = parameters[4:8]
    amount = parameters[8:12]
    intercept = parameters[12]
    p_react = expit(bank["react"] @ react)
    p_churn = expit(bank["churn"] @ churn)
    p_buy = np.where(bank["active"] == 0, p_react, 1.0 - p_churn)
    conditional_z = np.clip(
        _standardize(bank["amount"], amount_mean, amount_scale) @ amount + intercept,
        0.0,
        None,
    )
    return np.clip(np.power(p_buy, alpha) * conditional_z, 0.0, None)


def prediction_components(
    bank: dict[str, np.ndarray],
    parameters: np.ndarray,
    amount_mean: np.ndarray,
    amount_scale: np.ndarray,
    *,
    alpha: float = 1.1,
) -> dict[str, np.ndarray]:
    parameters = np.asarray(parameters, dtype=np.float64)
    react = parameters[:4]
    churn = parameters[4:8]
    amount = parameters[8:12]
    intercept = parameters[12]
    p_react = expit(bank["react"] @ react)
    p_churn = expit(bank["churn"] @ churn)
    p_buy = np.where(bank["active"] == 0, p_react, 1.0 - p_churn)
    conditional_z = np.clip(
        _standardize(bank["amount"], amount_mean, amount_scale) @ amount + intercept,
        0.0,
        None,
    )
    return {
        "p_react": p_react,
        "p_churn": p_churn,
        "p_buy": p_buy,
        "conditional_z": conditional_z,
        "prediction_z": np.clip(np.power(p_buy, alpha) * conditional_z, 0.0, None),
    }


def fit_meta(
    bank: dict[str, np.ndarray],
    *,
    prediction_bank_sha256: str,
    code_commit_sha: str,
    config_sha256: str,
) -> dict[str, Any]:
    if bank["react"].shape[1] != 4 or bank["churn"].shape[1] != 4 or bank["amount"].shape[1] != 4:
        raise ValueError("SSL V1 meta requires exactly four models per task")
    amount_mean = np.mean(bank["amount"].astype(np.float64), axis=0)
    amount_scale = np.std(bank["amount"].astype(np.float64), axis=0)
    amount_scale[amount_scale < 1e-12] = 1.0
    canonical = np.r_[np.full(4, 0.25), np.full(4, 0.25), np.ones(4), 0.0]
    rng = np.random.default_rng(EXPERIMENT.root_seed)
    starts = [canonical]
    starts.extend(
        np.r_[rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4)), rng.random(4), 0.0]
        for _ in range(8)
    )
    constraints = [
        {"type": "eq", "fun": lambda values: values[:4].sum() - 1.0},
        {"type": "eq", "fun": lambda values: values[4:8].sum() - 1.0},
    ]
    bounds = [(0.0, 1.0)] * 8 + [(0.0, None)] * 4 + [(None, None)]

    def objective(parameters: np.ndarray) -> float:
        prediction = predict_z(bank, parameters, amount_mean, amount_scale)
        return float(np.mean(np.square(prediction - bank["target"])))

    attempts: list[dict[str, Any]] = []
    for start_index, start in enumerate(starts):
        result = minimize(
            objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        attempts.append({
            "start_index": start_index,
            "success": bool(result.success),
            "message": str(result.message),
            "objective": float(result.fun),
            "parameters": result.x.astype(float).tolist(),
            "iterations": int(result.nit),
        })
    successful = [
        attempt for attempt in attempts
        if attempt["success"] and np.isfinite(attempt["objective"])
    ]
    if not successful:
        raise RuntimeError("No successful finite SLSQP meta result")
    best = min(successful, key=lambda attempt: attempt["objective"])
    package: dict[str, Any] = {
        "meta_schema_version": META_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT.experiment_id,
        "feature_order": {
            "react": list(REACT_COLUMNS),
            "churn": list(CHURN_COLUMNS),
            "amount": list(AMOUNT_COLUMNS),
        },
        "alpha": 1.1,
        "amount_mean": amount_mean.tolist(),
        "amount_scale": amount_scale.tolist(),
        "parameters": best["parameters"],
        "objective_mse_logspace": best["objective"],
        "rmsle": float(np.sqrt(best["objective"])),
        "attempts": attempts,
        "prediction_bank_sha256": prediction_bank_sha256,
        "code_commit_sha": code_commit_sha,
        "config_sha256": config_sha256,
    }
    canonical_bytes = json.dumps(package, sort_keys=True, separators=(",", ":")).encode("utf-8")
    package["package_content_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    return package


def apply_meta(package: dict[str, Any], bank: dict[str, np.ndarray]) -> np.ndarray:
    if package.get("meta_schema_version") != META_SCHEMA_VERSION:
        raise ValueError("Unsupported SSL meta schema version")
    if package.get("experiment_id") != EXPERIMENT.experiment_id:
        raise ValueError("Meta package belongs to a different experiment")
    expected_order = {
        "react": list(REACT_COLUMNS),
        "churn": list(CHURN_COLUMNS),
        "amount": list(AMOUNT_COLUMNS),
    }
    if package.get("feature_order") != expected_order:
        raise ValueError("Meta package feature order differs from prediction schema")
    return predict_z(
        bank,
        np.asarray(package["parameters"], dtype=np.float64),
        np.asarray(package["amount_mean"], dtype=np.float64),
        np.asarray(package["amount_scale"], dtype=np.float64),
        alpha=float(package["alpha"]),
    )


def apply_meta_components(package: dict[str, Any], bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    apply_meta(package, bank)  # validate the package before exposing components
    return prediction_components(
        bank,
        np.asarray(package["parameters"], dtype=np.float64),
        np.asarray(package["amount_mean"], dtype=np.float64),
        np.asarray(package["amount_scale"], dtype=np.float64),
        alpha=float(package["alpha"]),
    )
