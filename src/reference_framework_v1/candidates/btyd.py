"""Reserved leakage-safe BTYD feature-provider contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class BTYDRecipe:
    feature_set_id: str = "btyd_leakage_safe_v1"
    fit_scope: str = "train_anchors_only"
    unavailable_value: float = 0.0


class BTYDFeatureProvider(Protocol):
    """Future provider must fit only on each run's allowed train anchors."""

    feature_set_id: str

    def fit(self, frames: list[pl.DataFrame]) -> "BTYDFeatureProvider": ...

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame: ...
