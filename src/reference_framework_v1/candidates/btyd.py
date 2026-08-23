"""Leakage-safe BTYD features computed only from a frame's past state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

import polars as pl


@dataclass(frozen=True)
class BTYDRecipe:
    feature_set_id: str = "btyd_leakage_safe_v1"
    fit_scope: str = "train_anchors_only"
    unavailable_value: float = 0.0
    penalizer_coef: float = 0.001
    horizon_days: int = 30


class BTYDFeatureProvider(Protocol):
    """Future provider must fit only on each run's allowed train anchors."""

    feature_set_id: str

    def fit(self, frames: list[pl.DataFrame]) -> "BTYDFeatureProvider": ...

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame: ...


class LifetimesBTYDFeatureProvider:
    """BG/NBD + Gamma-Gamma features fitted on permitted training snapshots.

    All inputs are already causal snapshot columns.  Labels and future GMV are
    deliberately not read in either ``fit`` or ``transform``.
    """

    feature_set_id = "btyd_bg_nbd_gamma_gamma_v1"
    feature_names = (
        "btyd_frequency_90d", "btyd_recency_90d", "btyd_T_90d",
        "btyd_monetary_value_90d", "btyd_p_alive", "btyd_expected_purchases_30d",
        "btyd_expected_gmv_30d",
    )

    def __init__(self, recipe: BTYDRecipe = BTYDRecipe()) -> None:
        self.recipe = recipe
        self._bgf = None
        self._ggf = None

    @staticmethod
    def _inputs(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frequency = np.maximum(frame["purchase_days_90d"].to_numpy().astype(np.float64) - 1.0, 0.0)
        recency = np.where(frequency > 0, np.clip(90.0 - frame["days_since_last_order"].to_numpy().astype(np.float64), 0.0, 90.0), 0.0)
        age = np.clip(frame["available_history_days"].to_numpy().astype(np.float64), 1.0, 90.0)
        monetary = frame["gmv_sum_90d"].to_numpy().astype(np.float64) / np.maximum(frame["purchase_days_90d"].to_numpy().astype(np.float64), 1.0)
        return frequency, recency, age, np.maximum(monetary, 0.0)

    def fit(self, frames: list[pl.DataFrame]) -> "LifetimesBTYDFeatureProvider":
        from lifetimes import BetaGeoFitter, GammaGammaFitter

        if not frames:
            raise ValueError("BTYD requires at least one permitted training frame")
        frequency, recency, age, monetary = (np.concatenate(values) for values in zip(*(self._inputs(frame) for frame in frames), strict=True))
        self._bgf = BetaGeoFitter(penalizer_coef=self.recipe.penalizer_coef)
        self._bgf.fit(frequency, recency, age, verbose=False)
        mask = (frequency > 0) & (monetary > 0)
        if int(mask.sum()) >= 10:
            self._ggf = GammaGammaFitter(penalizer_coef=self.recipe.penalizer_coef)
            self._ggf.fit(frequency[mask], monetary[mask], verbose=False)
        return self

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self._bgf is None:
            raise RuntimeError("BTYD provider is not fitted")
        frequency, recency, age, monetary = self._inputs(frame)
        p_alive = np.nan_to_num(self._bgf.conditional_probability_alive(frequency, recency, age), nan=0.0, posinf=0.0, neginf=0.0)
        expected = np.nan_to_num(self._bgf.conditional_expected_number_of_purchases_up_to_time(self.recipe.horizon_days, frequency, recency, age), nan=0.0, posinf=0.0, neginf=0.0)
        if self._ggf is None:
            expected_gmv = expected * monetary
        else:
            value = monetary.copy()
            mask = frequency > 0
            value[mask] = self._ggf.conditional_expected_average_profit(frequency[mask], monetary[mask])
            expected_gmv = expected * np.maximum(value, 0.0)
        return pl.DataFrame({
            "btyd_frequency_90d": frequency.astype(np.float32), "btyd_recency_90d": recency.astype(np.float32),
            "btyd_T_90d": age.astype(np.float32), "btyd_monetary_value_90d": monetary.astype(np.float32),
            "btyd_p_alive": np.clip(p_alive, 0.0, 1.0).astype(np.float32),
            "btyd_expected_purchases_30d": np.maximum(expected, 0.0).astype(np.float32),
            "btyd_expected_gmv_30d": np.maximum(expected_gmv, 0.0).astype(np.float32),
        })
