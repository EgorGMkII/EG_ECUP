"""Feature-provider skeleton for the audited sparse CatBoost contract.

Fill this module from the supplied implementation guide.  Never use future
rows and never build a user-by-day grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import polars as pl


WINDOWS = (7, 14, 30, 60, 90, 180, 365)


@dataclass(frozen=True)
class TabularSnapshots:
    train: pl.DataFrame
    validation: pl.DataFrame
    feature_order: tuple[str, ...]
    manifest: dict[str, Any]


class FeatureProvider(ABC):
    provider_id: str

    @abstractmethod
    def build_pair(self, raw: pl.DataFrame, users: Sequence[int], train_anchor: date, validation_anchor: date) -> TabularSnapshots:
        raise NotImplementedError


class SparseAggregateFeatureProvider(FeatureProvider):
    """TODO: implement the external baseline's causal sparse aggregates."""

    provider_id = "sparse_aggregate_v1"

    def build_pair(self, raw: pl.DataFrame, users: Sequence[int], train_anchor: date, validation_anchor: date) -> TabularSnapshots:
        raise NotImplementedError("Implement sparse causal snapshot aggregation")


class BTYDFeatureProvider(FeatureProvider):
    """TODO: append audited BTYD classification features to base snapshots."""

    provider_id = "btyd_v1"

    def build_pair(self, raw: pl.DataFrame, users: Sequence[int], train_anchor: date, validation_anchor: date) -> TabularSnapshots:
        raise NotImplementedError("Wrap src.btyd_pipeline without labels or leakage")
