"""BTYD provider adapter contract for direct CatBoost ablation."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import polars as pl

from .features import FeatureProvider, SparseAggregateFeatureProvider, TabularSnapshots


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
        from src.btyd_pipeline import generate_btyd_dataset_for_anchor

        base = SparseAggregateFeatureProvider().build_pair(raw, users, train_anchor, validation_anchor)
        train_btyd, bg_model, gamma_model = generate_btyd_dataset_for_anchor(
            raw, list(users), train_anchor, fit_models=True
        )
        valid_btyd, _, _ = generate_btyd_dataset_for_anchor(
            raw, list(users), validation_anchor, bgnbd_model=bg_model, gamma_model=gamma_model, fit_models=False
        )
        columns = list(self.allowed_columns)
        train_extra = train_btyd.select(["user_id", *columns])
        valid_extra = valid_btyd.select(["user_id", *columns])
        train = base.train.join(train_extra, on="user_id", how="left")
        validation = base.validation.join(valid_extra, on="user_id", how="left")
        if train.null_count().sum_horizontal().item() or validation.null_count().sum_horizontal().item():
            raise ValueError("BTYD provider produced null values")
        order = (*base.feature_order, *columns)
        return TabularSnapshots(
            train=train.select(["user_id", *order]),
            validation=validation.select(["user_id", *order]),
            feature_order=order,
            manifest={
                **base.manifest,
                "provider_id": "sparse_aggregate_v1+btyd_v1",
                "btyd_fit_anchor": train_anchor.isoformat(),
                "btyd_validation_transform_anchor": validation_anchor.isoformat(),
                "btyd_columns": columns,
                "btyd_excluded": ["will_buy_30d", "btyd_expected_gmv_30d", "btyd_log_expected_gmv_30d"],
            },
        )
