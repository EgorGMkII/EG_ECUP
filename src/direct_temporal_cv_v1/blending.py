"""Leakage-safe direct blend protocol.

F1--F3 can fit weights; F4 is exclusively an evaluation gate.  This module is
not used until independent model prediction banks exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FrozenBlend:
    model_ids: tuple[str, ...]
    weights: tuple[float, ...]
    fit_folds: tuple[str, ...]


def fit_nonnegative_simplex_blend(predictions: Mapping[str, np.ndarray], target_z: np.ndarray, *, fit_folds: tuple[str, ...] = ("F1", "F2", "F3")) -> FrozenBlend:
    """TODO: fit convex weights only on concatenated development-fold banks."""
    raise NotImplementedError


def apply_blend(blend: FrozenBlend, predictions: Mapping[str, np.ndarray]) -> np.ndarray:
    columns = [predictions[model_id] for model_id in blend.model_ids]
    return np.asarray(columns).T @ np.asarray(blend.weights)
