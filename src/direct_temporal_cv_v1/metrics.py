"""Direct prediction validation and fold-level metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Metrics:
    rmsle: float
    mse_z: float
    prediction_mean: float
    prediction_median: float
    prediction_min: float
    prediction_max: float
    zero_rate: float


def evaluate_z(target_z: np.ndarray, prediction_z: np.ndarray) -> Metrics:
    if target_z.shape != prediction_z.shape or target_z.ndim != 1:
        raise ValueError("Target and prediction must be aligned one-dimensional arrays")
    if not np.isfinite(prediction_z).all():
        raise ValueError("Predictions contain non-finite values")
    mse = float(np.mean(np.square(target_z - prediction_z)))
    values = np.maximum(np.expm1(prediction_z), 0.0)
    return Metrics(math.sqrt(mse), mse, float(values.mean()), float(np.median(values)), float(values.min()), float(values.max()), float(np.mean(values == 0.0)))


def metrics_dict(metrics: Metrics) -> dict[str, float]:
    return asdict(metrics)
