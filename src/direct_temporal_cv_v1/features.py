"""Feature-provider skeleton for the audited sparse CatBoost contract.

Fill this module from the supplied implementation guide.  Never use future
rows and never build a user-by-day grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
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
    """Build sparse, user-level snapshots using only rows up to an anchor.

    This provider deliberately does not materialise a ``user x day`` panel.
    Each window is reduced directly from observed rows and then left-joined to
    the template user index.  Keeping the aggregation here (instead of using
    the historical snapshot builder) makes the direct-CV feature contract
    auditable and independent from the SSL/reference pipelines.
    """

    provider_id = "sparse_aggregate_v1"

    def build_pair(self, raw: pl.DataFrame, users: Sequence[int], train_anchor: date, validation_anchor: date) -> TabularSnapshots:
        train = self._build_snapshot(raw, users, train_anchor)
        validation = self._build_snapshot(raw, users, validation_anchor)
        order = tuple(c for c in train.columns if c != "user_id")
        if tuple(c for c in validation.columns if c != "user_id") != order:
            raise RuntimeError("Train and validation feature orders differ")
        manifest = {
            "provider_id": self.provider_id,
            "windows_days": list(WINDOWS),
            "feature_order": list(order),
            "causal_rule": "event_date <= anchor",
            "sparse_aggregation": True,
            "excluded_columns": ["target", "will_buy_30d", "target_start", "target_end"],
            "train_anchor": train_anchor.isoformat(),
            "validation_anchor": validation_anchor.isoformat(),
        }
        return TabularSnapshots(train, validation, order, manifest)

    @staticmethod
    def _normalise_dates(raw: pl.DataFrame) -> pl.DataFrame:
        if raw.schema.get("event_date") == pl.String:
            return raw.with_columns(pl.col("event_date").str.to_date())
        if raw.schema.get("event_date") == pl.Datetime:
            return raw.with_columns(pl.col("event_date").dt.date())
        return raw

    def _build_snapshot(self, raw: pl.DataFrame, users: Sequence[int], anchor: date) -> pl.DataFrame:
        if "user_id" not in raw.columns or "event_date" not in raw.columns:
            raise ValueError("raw data must contain user_id and event_date")
        data = self._normalise_dates(raw)
        user_frame = pl.DataFrame({"user_id": list(users)})
        if user_frame["user_id"].n_unique() != len(users):
            raise ValueError("users must be unique and template ordered")
        available = set(data.columns)
        required = {"gmv", "gmv_search", "gmv_cat", "searches", "to_ord", "to_cart"}
        missing = required - available
        if missing:
            raise ValueError(f"raw data missing required feature columns: {sorted(missing)}")

        # Restrict once at the maximum window. All rolling expressions below
        # operate on this historical-only sparse frame. Lifetime/recency use
        # the full causal history, still without materialising a daily grid.
        causal = data.filter(
            pl.col("user_id").is_in(list(users))
            & (pl.col("event_date") <= anchor)
        )
        history = causal.filter(
            pl.col("user_id").is_in(list(users))
            & pl.col("event_date").is_between(anchor - timedelta(days=364), anchor, closed="both")
        )
        result = user_frame
        feature_order: list[str] = []
        active = (pl.col("searches") > 0) | (pl.col("to_cart") > 0) | (pl.col("to_ord") > 0) | (pl.col("gmv") > 0)
        orders = (pl.col("to_ord") > 0) | (pl.col("gmv") > 0)
        metric_specs = (("gmv", "gmv"), ("gmv_search", "gmv_search"), ("gmv_cat", "gmv_cat"),
                        ("searches", "searches"), ("to_ord", "orders"), ("to_cart", "carts"))
        for days in WINDOWS:
            start = anchor - timedelta(days=days - 1)
            window = history.filter(pl.col("event_date") >= start)
            expressions: list[pl.Expr] = [
                active.cast(pl.Int8).sum().alias(f"active_days_{days}d"),
                orders.cast(pl.Int8).sum().alias(f"order_days_{days}d"),
                (pl.col("to_cart") > 0).cast(pl.Int8).sum().alias(f"cart_days_{days}d"),
            ]
            names = [f"active_days_{days}d", f"order_days_{days}d", f"cart_days_{days}d"]
            for source, output in metric_specs:
                expressions.append(pl.col(source).fill_null(0.0).sum().alias(f"{output}_sum_{days}d"))
                names.append(f"{output}_sum_{days}d")
            grouped = window.group_by("user_id").agg(expressions)
            result = result.join(grouped, on="user_id", how="left")
            feature_order.extend(names)

        # Lifetime and recency features use all rows observed before the
        # anchor, not a future-labelled snapshot.
        recency_exprs = [
            (pl.lit(anchor) - pl.col("event_date").max()).dt.total_days().alias("days_since_last_activity"),
            (pl.lit(anchor) - pl.when(orders).then(pl.col("event_date")).otherwise(None).max()).dt.total_days().alias("days_since_last_order"),
            (pl.lit(anchor) - pl.when(pl.col("to_cart") > 0).then(pl.col("event_date")).otherwise(None).max()).dt.total_days().alias("days_since_last_cart"),
            (pl.lit(anchor) - pl.when(pl.col("gmv") > 0).then(pl.col("event_date")).otherwise(None).max()).dt.total_days().alias("days_since_last_gmv"),
            (pl.lit(anchor) - pl.col("event_date").min()).dt.total_days().alias("customer_age_days"),
            active.cast(pl.Int8).sum().alias("lifetime_active_days"),
        ]
        lifetime = causal.group_by("user_id").agg(recency_exprs)
        result = result.join(lifetime, on="user_id", how="left")
        feature_order.extend(["days_since_last_activity", "days_since_last_order", "days_since_last_cart", "days_since_last_gmv", "customer_age_days", "lifetime_active_days"])

        recency = {"days_since_last_activity", "days_since_last_order", "days_since_last_cart", "days_since_last_gmv", "customer_age_days"}
        fill = [pl.col(c).fill_null(999.0) if c in recency else pl.col(c).fill_null(0.0) for c in feature_order]
        result = result.with_columns(fill).select(["user_id", *feature_order])

        # Cadence, interval, momentum and dormant browsing features
        cadence_exprs = [
            # 1. Mean order interval over 365d (IPI)
            ((pl.col("customer_age_days") - pl.col("days_since_last_order")) / pl.when(pl.col("order_days_365d") > 1).then(pl.col("order_days_365d") - 1).otherwise(1.0))
            .clip(1.0, 365.0)
            .alias("mean_order_interval_365d"),

            # 2. Recency-to-cadence ratio
            (pl.col("days_since_last_order") / (
                ((pl.col("customer_age_days") - pl.col("days_since_last_order")) / pl.when(pl.col("order_days_365d") > 1).then(pl.col("order_days_365d") - 1).otherwise(1.0)).clip(1.0, 365.0)
            ))
            .clip(0.0, 50.0)
            .alias("recency_to_cadence_ratio"),

            # 3. Search without order gap (active shopper intent before anchor)
            (pl.col("days_since_last_order") - pl.col("days_since_last_activity"))
            .clip(0.0, 999.0)
            .alias("search_without_order_gap"),

            # 4. Cart intent gap
            (pl.col("days_since_last_order") - pl.col("days_since_last_cart"))
            .clip(0.0, 999.0)
            .alias("cart_intent_gap"),

            # 5. Dormant browsing signals (active in searches/carts despite 0 orders in 30d)
            ((pl.col("days_since_last_order") > 30) & (pl.col("days_since_last_activity") <= 7)).cast(pl.Float32).alias("is_searching_dormant"),
            ((pl.col("days_since_last_order") > 30) & (pl.col("days_since_last_cart") <= 14)).cast(pl.Float32).alias("is_carting_dormant"),
            (pl.col("searches_sum_30d") / (pl.col("customer_age_days") + 1.0)).clip(0.0, 100.0).alias("dormant_search_density"),
            (pl.col("days_since_last_order") > (1.5 * ((pl.col("customer_age_days") - pl.col("days_since_last_order")) / pl.when(pl.col("order_days_365d") > 1).then(pl.col("order_days_365d") - 1).otherwise(1.0)).clip(1.0, 365.0))).cast(pl.Float32).alias("is_overdue"),
            (pl.col("cart_days_30d") / (pl.col("active_days_30d") + 0.1)).clip(0.0, 10.0).alias("cart_to_search_ratio_30d"),

            # 6. Velocities: 30d vs 90d (30d rate / 90d average per 30d)
            (pl.col("order_days_30d") / (pl.col("order_days_90d") / 3.0 + 0.1)).alias("order_velocity_30_to_90"),
            (pl.col("cart_days_30d") / (pl.col("cart_days_90d") / 3.0 + 0.1)).alias("cart_velocity_30_to_90"),
            (pl.col("searches_sum_30d") / (pl.col("searches_sum_90d") / 3.0 + 1.0)).alias("searches_velocity_30_to_90"),
            (pl.col("gmv_sum_30d") / (pl.col("gmv_sum_90d") / 3.0 + 1.0)).alias("gmv_velocity_30_to_90"),

            # 7. Velocities: 7d vs 30d (7d rate / 30d average per 7d)
            (pl.col("order_days_7d") / (pl.col("order_days_30d") * (7.0 / 30.0) + 0.1)).alias("order_velocity_7_to_30"),
            (pl.col("searches_sum_7d") / (pl.col("searches_sum_30d") * (7.0 / 30.0) + 1.0)).alias("searches_velocity_7_to_30"),
            (pl.col("cart_days_7d") / (pl.col("cart_days_30d") * (7.0 / 30.0) + 0.1)).alias("cart_velocity_7_to_30"),
            (pl.col("gmv_sum_7d") / (pl.col("gmv_sum_30d") * (7.0 / 30.0) + 1.0)).alias("gmv_velocity_7_to_30"),
        ]
        cadence_names = [
            "mean_order_interval_365d", "recency_to_cadence_ratio", "search_without_order_gap",
            "cart_intent_gap", "is_searching_dormant", "is_carting_dormant", "dormant_search_density",
            "is_overdue", "cart_to_search_ratio_30d",
            "order_velocity_30_to_90", "cart_velocity_30_to_90", "searches_velocity_30_to_90", "gmv_velocity_30_to_90",
            "order_velocity_7_to_30", "searches_velocity_7_to_30", "cart_velocity_7_to_30", "gmv_velocity_7_to_30",
        ]
        result = result.with_columns(cadence_exprs)
        feature_order.extend(cadence_names)

        # 8. BTYD Probabilistic Features (BG/NBD & Gamma-Gamma)
        from src.btyd_pipeline import generate_btyd_dataset_for_anchor
        btyd_df, _, _ = generate_btyd_dataset_for_anchor(causal, list(users), anchor, fit_models=True)
        btyd_cols = [c for c in btyd_df.columns if c.startswith("btyd_")]
        result = result.join(btyd_df.select(["user_id", *btyd_cols]), on="user_id", how="left")
        btyd_fill = [pl.col(c).fill_null(0.0) for c in btyd_cols]
        result = result.with_columns(btyd_fill)
        feature_order.extend(btyd_cols)

        values = result.select(feature_order)
        # Polars can produce null/invalid values from malformed input. Make
        # the failure explicit before a model sees them.
        if values.null_count().sum_horizontal().item() != 0:
            raise ValueError("Sparse feature provider produced null feature values")
        return result


class BTYDFeatureProvider(FeatureProvider):
    """TODO: append audited BTYD classification features to base snapshots."""

    provider_id = "btyd_v1"

    def build_pair(self, raw: pl.DataFrame, users: Sequence[int], train_anchor: date, validation_anchor: date) -> TabularSnapshots:
        raise NotImplementedError("Wrap src.btyd_pipeline without labels or leakage")
