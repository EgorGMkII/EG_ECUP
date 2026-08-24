"""BTYD provider adapter contract for direct CatBoost ablation."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import polars as pl

from .features import FeatureProvider, TabularSnapshots


class DirectBTYDFeatureProvider(FeatureProvider):
    """Append only the audited causal BTYD classifier features.

    The implementation is intentionally separate from the target extractor:
    it must fit BG/NBD and Gamma-Gamma parameters on the training snapshot,
    then transform train/validation without exposing future labels. Amount
    BTYD outputs and ``will_buy_30d`` are prohibited in this provider.
    """

    provider_id = "btyd_v1"
    allowed_columns = ("btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive")

    def build_pair(self, raw: pl.DataFrame, users: Sequence[int], train_anchor: date, validation_anchor: date) -> TabularSnapshots:
        raise NotImplementedError("Wrap src.btyd_pipeline with train-anchor fitting and causal validation transform")
