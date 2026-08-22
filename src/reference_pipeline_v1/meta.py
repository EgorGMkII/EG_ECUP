"""Frozen, M-only constrained hurdle meta optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

AMOUNT_COLUMNS = ("cb_amount_z", "s1_amount_z", "s2_amount_z", "ett_amount_z")
REACT_COLUMNS = tuple(x.replace("amount_z", "react_logit") for x in AMOUNT_COLUMNS)
CHURN_COLUMNS = tuple(x.replace("amount_z", "churn_logit") for x in AMOUNT_COLUMNS)


def predict(bank: dict[str, np.ndarray], params: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    react, churn, amount, intercept = params[:4], params[4:8], params[8:12], params[12]
    p_react = expit(bank["react"] @ react); p_churn = expit(bank["churn"] @ churn)
    p_buy = np.where(bank["active"] == 0, p_react, 1.0 - p_churn)
    conditional = np.clip(scaler.transform(bank["amount"]) @ amount + intercept, 0.0, None)
    return np.clip(np.power(p_buy, 1.1) * conditional, 0.0, None)


def fit_meta(bank: dict[str, np.ndarray], prediction_bank_sha256: str, historical_start: np.ndarray | None = None) -> dict:
    scaler = StandardScaler().fit(bank["amount"].astype(np.float64))
    rng = np.random.default_rng(42)
    starts = [np.r_[np.full(4, .25), np.full(4, .25), np.ones(4), 0.0]]
    if historical_start is not None: starts.append(historical_start)
    starts.extend(np.r_[rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4)), rng.random(4), 0.] for _ in range(8))
    constraints = [{"type": "eq", "fun": lambda p: p[:4].sum() - 1}, {"type": "eq", "fun": lambda p: p[4:8].sum() - 1}]
    bounds = [(0., 1.)] * 8 + [(0., None)] * 4 + [(None, None)]
    results = []
    for start in starts:
        result = minimize(lambda p: float(np.mean((predict(bank, p, scaler) - bank["target"]) ** 2)), start, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
        results.append({"x": result.x, "fun": float(result.fun), "success": bool(result.success), "message": str(result.message)})
    finite = [result for result in results if result["success"] and np.isfinite(result["fun"])]
    if not finite: raise RuntimeError("No finite SLSQP meta result")
    best = min(finite, key=lambda item: item["fun"])
    return {"alpha": 1.1, "model_order": ["CatBoost", "S1", "S2", "ETT"], "amount_feature_order": list(AMOUNT_COLUMNS), "amount_dtype": "float64", "amount_scaler_mean": scaler.mean_.tolist(), "amount_scaler_scale": scaler.scale_.tolist(), "prediction_bank_sha256": prediction_bank_sha256, "parameters": best["x"].tolist(), "objective": best["fun"], "starts": [{"objective": r["fun"], "success": r["success"], "message": r["message"], "parameters": r["x"].tolist()} for r in results]}


def load_predict(package: dict, bank: dict[str, np.ndarray]) -> np.ndarray:
    scaler = StandardScaler(); scaler.mean_ = np.asarray(package["amount_scaler_mean"]); scaler.scale_ = np.asarray(package["amount_scaler_scale"]); scaler.n_features_in_ = 4
    return predict(bank, np.asarray(package["parameters"]), scaler)
