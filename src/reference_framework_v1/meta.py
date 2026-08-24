"""Named-column constrained joint meta model (schema v2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .predictions import PredictionSchema


META_SCHEMA_VERSION = 2
DIRECT_META_SCHEMA_VERSION = 3


def _hurdle_predict(bank: dict[str, np.ndarray], parameters: np.ndarray, mean: np.ndarray, scale: np.ndarray, schema: PredictionSchema, alpha: float) -> np.ndarray:
    r, c, a = len(schema.react_columns), len(schema.churn_columns), len(schema.amount_columns)
    react, churn, amount, intercept = parameters[:r], parameters[r:r+c], parameters[r+c:r+c+a], parameters[-1]
    p_react, p_churn = expit(bank["react"] @ react), expit(bank["churn"] @ churn)
    p_buy = np.where(bank["active"] == 0, p_react, 1.0 - p_churn)
    conditional = np.clip(((bank["amount"] - mean) / scale) @ amount + intercept, 0.0, None)
    return np.clip(np.power(p_buy, alpha) * conditional, 0.0, None)


def _predict(bank: dict[str, np.ndarray], parameters: np.ndarray, mean: np.ndarray, scale: np.ndarray, schema: PredictionSchema, alpha: float) -> np.ndarray:
    hurdle = _hurdle_predict(bank, parameters, mean, scale, schema, alpha)
    if not schema.direct_columns:
        return hurdle
    base = len(schema.react_columns) + len(schema.churn_columns) + len(schema.amount_columns) + 1
    blend = parameters[base:]
    branches = np.column_stack((hurdle, bank["direct"]))
    return np.clip(branches @ blend, 0.0, None)


def fit_meta(bank: dict[str, np.ndarray], schema: PredictionSchema, *, root_seed: int, prediction_bank_sha256: str, commit_sha: str, config_sha256: str) -> dict[str, Any]:
    r, c, a = len(schema.react_columns), len(schema.churn_columns), len(schema.amount_columns)
    mean, scale = bank["amount"].mean(axis=0), bank["amount"].std(axis=0)
    scale[scale < 1e-12] = 1.0
    alpha = 1.1
    branch_count = 1 + len(schema.direct_columns)
    canonical_blend = np.r_[1.0, np.zeros(branch_count - 1)] if schema.direct_columns else np.empty(0)
    canonical = np.r_[np.full(r, 1.0 / r), np.full(c, 1.0 / c), np.ones(a), 0.0, canonical_blend]
    rng = np.random.default_rng(root_seed)
    starts = [canonical] + [np.r_[rng.dirichlet(np.ones(r)), rng.dirichlet(np.ones(c)), rng.random(a), 0.0, rng.dirichlet(np.ones(branch_count)) if schema.direct_columns else np.empty(0)] for _ in range(8)]
    constraints = [
        {"type": "eq", "fun": lambda x: x[:r].sum() - 1.0},
        {"type": "eq", "fun": lambda x: x[r:r+c].sum() - 1.0},
    ]
    if schema.direct_columns:
        constraints.append({"type": "eq", "fun": lambda x: x[r+c+a+1:].sum() - 1.0})
    bounds = [(0.0, 1.0)] * (r + c) + [(0.0, None)] * a + [(None, None)] + ([(0.0, 1.0)] * branch_count if schema.direct_columns else [])
    attempts: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        result = minimize(lambda x: float(np.mean(np.square(_predict(bank, x, mean, scale, schema, alpha) - bank["target"]))), start, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-10})
        attempts.append({"start_index": index, "success": bool(result.success), "objective": float(result.fun), "parameters": result.x.astype(float).tolist(), "message": str(result.message)})
    successful = [item for item in attempts if item["success"] and np.isfinite(item["objective"])]
    if not successful:
        raise RuntimeError("No finite meta optimization result")
    best = min(successful, key=lambda item: item["objective"])
    values = np.asarray(best["parameters"], dtype=np.float64)
    package: dict[str, Any] = {
        "meta_schema_version": DIRECT_META_SCHEMA_VERSION if schema.direct_columns else META_SCHEMA_VERSION,
        "feature_order": {"react": list(schema.react_columns), "churn": list(schema.churn_columns), "amount": list(schema.amount_columns), **({"direct": list(schema.direct_columns)} if schema.direct_columns else {})},
        "alpha": alpha,
        "amount_mean": mean.tolist(), "amount_scale": scale.tolist(),
        "react_weights": dict(zip(schema.react_columns, values[:r], strict=True)),
        "churn_weights": dict(zip(schema.churn_columns, values[r:r+c], strict=True)),
        "amount_coefficients": dict(zip(schema.amount_columns, values[r+c:r+c+a], strict=True)),
        "amount_intercept": float(values[-1]),
        "objective_mse_logspace": best["objective"], "rmsle": float(np.sqrt(best["objective"])),
        "attempts": attempts, "prediction_bank_sha256": prediction_bank_sha256,
        "code_commit_sha": commit_sha, "config_sha256": config_sha256,
    }
    amount_intercept_index = r + c + a
    package["amount_intercept"] = float(values[amount_intercept_index])
    if schema.direct_columns:
        package["late_blend_weights"] = dict(zip(("hurdle", *schema.direct_columns), values[amount_intercept_index + 1:], strict=True))
    package["package_content_sha256"] = hashlib.sha256(json.dumps(package, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return package


def apply_meta(package: dict[str, Any], bank: dict[str, np.ndarray], schema: PredictionSchema) -> np.ndarray:
    return apply_meta_components(package, bank, schema)["prediction_z"]


def apply_meta_components(package: dict[str, Any], bank: dict[str, np.ndarray], schema: PredictionSchema) -> dict[str, np.ndarray]:
    expected = {"react": list(schema.react_columns), "churn": list(schema.churn_columns), "amount": list(schema.amount_columns), **({"direct": list(schema.direct_columns)} if schema.direct_columns else {})}
    expected_version = DIRECT_META_SCHEMA_VERSION if schema.direct_columns else META_SCHEMA_VERSION
    if package.get("meta_schema_version") != expected_version or package.get("feature_order") != expected:
        raise ValueError("Frozen meta package schema differs from prediction bank")
    parameters = np.r_[
        [package["react_weights"][name] for name in schema.react_columns],
        [package["churn_weights"][name] for name in schema.churn_columns],
        [package["amount_coefficients"][name] for name in schema.amount_columns],
        package["amount_intercept"],
    ]
    r, c, a = len(schema.react_columns), len(schema.churn_columns), len(schema.amount_columns)
    params = np.asarray(parameters, dtype=np.float64)
    p_react, p_churn = expit(bank["react"] @ params[:r]), expit(bank["churn"] @ params[r:r+c])
    p_buy = np.where(bank["active"] == 0, p_react, 1.0 - p_churn)
    conditional = np.clip(((bank["amount"] - np.asarray(package["amount_mean"])) / np.asarray(package["amount_scale"])) @ params[r+c:r+c+a] + params[-1], 0.0, None)
    hurdle = _hurdle_predict(bank, params, np.asarray(package["amount_mean"]), np.asarray(package["amount_scale"]), schema, float(package["alpha"]))
    if schema.direct_columns:
        blend = np.asarray([package["late_blend_weights"][name] for name in ("hurdle", *schema.direct_columns)], dtype=np.float64)
        prediction = np.column_stack((hurdle, bank["direct"])) @ blend
    else:
        prediction = hurdle
    return {"p_react": p_react, "p_churn": p_churn, "p_buy": p_buy, "conditional_z": conditional, "hurdle_prediction_z": hurdle, "prediction_z": np.clip(prediction, 0.0, None)}
