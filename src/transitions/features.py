"""Lifecycle, Cadence, Decay, Buying Intent, and Last-Year Feature Engineering Engine."""

import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl

from src.snapshots import TRAIN_PARQUET


MAJOR_HOLIDAYS_MD = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 14), (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4), (12, 31)]


def compute_calendar_context(anchor_date: date) -> Dict[str, float]:
    """Computes fixed calendar features known at anchor date for the upcoming 30-day window."""
    target_start = anchor_date + timedelta(days=1)
    target_end = anchor_date + timedelta(days=30)

    # Days of year
    doy = anchor_date.timetuple().tm_yday
    doy_sin = math.sin(2 * math.pi * doy / 365.25)
    doy_cos = math.cos(2 * math.pi * doy / 365.25)

    # Target window composition
    target_days = [target_start + timedelta(days=i) for i in range(30)]
    weekends_count = sum(1 for d in target_days if d.weekday() in (5, 6))

    holiday_count = sum(1 for d in target_days if (d.month, d.day) in MAJOR_HOLIDAYS_MD)

    # Days to nearest major holiday
    min_dist = 365
    for m, d in MAJOR_HOLIDAYS_MD:
        try:
            h_date_curr = date(anchor_date.year, m, d)
            diff = (h_date_curr - anchor_date).days
            if diff >= 0 and diff < min_dist:
                min_dist = diff
            h_date_next = date(anchor_date.year + 1, m, d)
            diff_next = (h_date_next - anchor_date).days
            if diff_next >= 0 and diff_next < min_dist:
                min_dist = diff_next
        except ValueError:
            continue

    return {
        "cal_anchor_month": float(anchor_date.month),
        "cal_anchor_doy_sin": float(doy_sin),
        "cal_anchor_doy_cos": float(doy_cos),
        "cal_weekends_in_target_window": float(weekends_count),
        "cal_holiday_days_in_target_window": float(holiday_count),
        "cal_days_to_nearest_holiday": float(min_dist),
    }


def compute_lifecycle_features(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
) -> pl.DataFrame:
    """Computes Recency, Cadence, Activity Decay, Buying Intent, and Streak features up to anchor_date."""
    # Filter historical logs up to anchor_date (past 365 days)
    history_start = anchor_date - timedelta(days=364)
    hist = data.filter(
        (pl.col("user_id").is_in(user_ids))
        & (pl.col("event_date") >= history_start)
        & (pl.col("event_date") <= anchor_date)
    )

    base_users_df = pl.DataFrame({"user_id": user_ids})

    if hist.is_empty():
        # Return all nulls / defaults
        return base_users_df

    # Total active days and purchase days in windows
    agg_df = hist.group_by("user_id").agg([
        # Recency: last date of events
        pl.col("event_date").max().alias("max_act_date"),
        pl.col("event_date").filter(pl.col("searches") > 0).max().alias("max_search_date"),
        pl.col("event_date").filter(pl.col("to_cart") > 0).max().alias("max_cart_date"),
        pl.col("event_date").filter(pl.col("to_ord") > 0).max().alias("max_order_date"),
        pl.col("event_date").min().alias("first_seen_date"),

        # Counts
        (pl.col("event_date") >= (anchor_date - timedelta(days=29))).sum().alias("active_days_30d"),
        (pl.col("event_date") >= (anchor_date - timedelta(days=89))).sum().alias("active_days_90d"),
        ((pl.col("event_date") >= (anchor_date - timedelta(days=29))) & (pl.col("to_ord") > 0)).sum().alias("purchase_days_30d"),
        ((pl.col("event_date") >= (anchor_date - timedelta(days=89))) & (pl.col("to_ord") > 0)).sum().alias("purchase_days_90d"),

        # Short window aggregates for decay
        pl.col("searches").filter(pl.col("event_date") >= (anchor_date - timedelta(days=2))).sum().alias("searches_3d"),
        pl.col("to_cart").filter(pl.col("event_date") >= (anchor_date - timedelta(days=2))).sum().alias("carts_3d"),
        pl.col("to_ord").filter(pl.col("event_date") >= (anchor_date - timedelta(days=2))).sum().alias("orders_3d"),
        pl.col("gmv").filter(pl.col("event_date") >= (anchor_date - timedelta(days=2))).sum().alias("gmv_3d"),

        pl.col("searches").filter(pl.col("event_date") >= (anchor_date - timedelta(days=6))).sum().alias("searches_7d"),
        pl.col("to_cart").filter(pl.col("event_date") >= (anchor_date - timedelta(days=6))).sum().alias("carts_7d"),
        pl.col("to_ord").filter(pl.col("event_date") >= (anchor_date - timedelta(days=6))).sum().alias("orders_7d"),
        pl.col("gmv").filter(pl.col("event_date") >= (anchor_date - timedelta(days=6))).sum().alias("gmv_7d"),

        pl.col("searches").filter(pl.col("event_date") >= (anchor_date - timedelta(days=13))).sum().alias("searches_14d"),
        pl.col("to_cart").filter(pl.col("event_date") >= (anchor_date - timedelta(days=13))).sum().alias("carts_14d"),
        pl.col("to_ord").filter(pl.col("event_date") >= (anchor_date - timedelta(days=13))).sum().alias("orders_14d"),
        pl.col("gmv").filter(pl.col("event_date") >= (anchor_date - timedelta(days=13))).sum().alias("gmv_14d"),

        pl.col("searches").filter(pl.col("event_date") >= (anchor_date - timedelta(days=29))).sum().alias("searches_30d"),
        pl.col("to_cart").filter(pl.col("event_date") >= (anchor_date - timedelta(days=29))).sum().alias("carts_30d"),
        pl.col("to_ord").filter(pl.col("event_date") >= (anchor_date - timedelta(days=29))).sum().alias("orders_30d"),
        pl.col("gmv").filter(pl.col("event_date") >= (anchor_date - timedelta(days=29))).sum().alias("gmv_30d"),

        pl.col("searches").filter(pl.col("event_date") >= (anchor_date - timedelta(days=59))).sum().alias("searches_60d"),
        pl.col("to_cart").filter(pl.col("event_date") >= (anchor_date - timedelta(days=59))).sum().alias("carts_60d"),
        pl.col("to_ord").filter(pl.col("event_date") >= (anchor_date - timedelta(days=59))).sum().alias("orders_60d"),
        pl.col("gmv").filter(pl.col("event_date") >= (anchor_date - timedelta(days=59))).sum().alias("gmv_60d"),
    ])


    # Join with base users
    res = base_users_df.join(agg_df, on="user_id", how="left")

    # Compute Recency & Missing Flags
    res = res.with_columns([
        # Days since last events
        pl.when(pl.col("max_act_date").is_not_null())
        .then((pl.lit(anchor_date) - pl.col("max_act_date")).dt.total_days())
        .otherwise(365.0)
        .alias("days_since_last_activity"),

        pl.when(pl.col("max_search_date").is_not_null())
        .then((pl.lit(anchor_date) - pl.col("max_search_date")).dt.total_days())
        .otherwise(365.0)
        .alias("days_since_last_search"),

        pl.when(pl.col("max_cart_date").is_not_null())
        .then((pl.lit(anchor_date) - pl.col("max_cart_date")).dt.total_days())
        .otherwise(365.0)
        .alias("days_since_last_cart"),

        pl.when(pl.col("max_order_date").is_not_null())
        .then((pl.lit(anchor_date) - pl.col("max_order_date")).dt.total_days())
        .otherwise(365.0)
        .alias("days_since_last_order"),

        # Missing flags
        pl.col("max_act_date").is_null().cast(pl.Float32).alias("is_missing_last_activity"),
        pl.col("max_search_date").is_null().cast(pl.Float32).alias("is_missing_last_search"),
        pl.col("max_cart_date").is_null().cast(pl.Float32).alias("is_missing_last_cart"),
        pl.col("max_order_date").is_null().cast(pl.Float32).alias("is_missing_last_order"),

        # Tenure
        pl.when(pl.col("first_seen_date").is_not_null())
        .then((pl.lit(anchor_date) - pl.col("first_seen_date")).dt.total_days())
        .otherwise(0.0)
        .alias("user_tenure_days"),
    ])

    # Activity Decay Ratios and Differences
    res = res.with_columns([
        # Ratios (short / long)
        (pl.col("searches_7d") / (pl.col("searches_30d") + 1e-4)).alias("decay_searches_7d_vs_30d"),
        (pl.col("carts_7d") / (pl.col("carts_30d") + 1e-4)).alias("decay_carts_7d_vs_30d"),
        (pl.col("orders_7d") / (pl.col("orders_30d") + 1e-4)).alias("decay_orders_7d_vs_30d"),
        (pl.col("gmv_7d") / (pl.col("gmv_30d") + 1e-4)).alias("decay_gmv_7d_vs_30d"),
        (pl.col("searches_3d") / (pl.col("searches_14d") + 1e-4)).alias("decay_searches_3d_vs_14d"),
        (pl.col("carts_3d") / (pl.col("carts_14d") + 1e-4)).alias("decay_carts_3d_vs_14d"),

        # Normalized Differences (recent week vs expected week)
        (pl.col("searches_7d") - (pl.col("searches_30d") / 4.2857)).alias("diff_searches_7d_norm_30d"),
        (pl.col("carts_7d") - (pl.col("carts_30d") / 4.2857)).alias("diff_carts_7d_norm_30d"),
        (pl.col("orders_14d") - (pl.col("orders_60d") / 4.2857)).alias("diff_orders_14d_norm_60d"),
        (pl.col("gmv_14d") - (pl.col("gmv_60d") / 4.2857)).alias("diff_gmv_14d_norm_60d"),

        # Buying Intent (Unfinished Funnels)
        ((pl.col("searches_7d") > 0) & (pl.col("orders_7d") == 0)).cast(pl.Float32).alias("intent_search_without_order_7d"),
        ((pl.col("carts_7d") > 0) & (pl.col("orders_7d") == 0)).cast(pl.Float32).alias("intent_cart_without_order_7d"),
        ((pl.col("searches_3d") > 0) & (pl.col("orders_3d") == 0)).cast(pl.Float32).alias("intent_search_without_order_3d"),
        ((pl.col("carts_3d") > 0) & (pl.col("orders_3d") == 0)).cast(pl.Float32).alias("intent_cart_without_order_3d"),
        (pl.col("carts_3d") - pl.col("orders_3d")).alias("intent_unfinished_funnel_3d"),
        (pl.col("carts_7d") - pl.col("orders_7d")).alias("intent_unfinished_funnel_7d"),

        # Accelerations (daily rate ratio)
        ((pl.col("searches_3d") / 3.0) / (pl.col("searches_14d") / 14.0 + 1e-4)).alias("intent_search_accel_3d_vs_14d"),
        ((pl.col("carts_3d") / 3.0) / (pl.col("carts_14d") / 14.0 + 1e-4)).alias("intent_cart_accel_3d_vs_14d"),
    ])

    # Fill NaNs/Nulls only on feature columns (not user_id)
    drop_cols = ["max_act_date", "max_search_date", "max_cart_date", "max_order_date", "first_seen_date"]
    res = res.drop([c for c in drop_cols if c in res.columns])
    feat_cols = [c for c in res.columns if c != "user_id"]
    res = res.with_columns([pl.col("user_id").cast(pl.Int64)] + [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in feat_cols])

    return res


def compute_last_year_features(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
) -> pl.DataFrame:
    """Computes last-year target window analog features [anchor + 1d - 1yr, anchor + 30d - 1yr]."""
    base_users_df = pl.DataFrame({"user_id": user_ids}, schema={"user_id": pl.Int64})

    # Calendar shift by 1 year (safe for leap years)
    try:
        ly_target_start = date(anchor_date.year - 1, anchor_date.month, anchor_date.day) + timedelta(days=1)
    except ValueError:
        # Leap year Feb 29
        ly_target_start = date(anchor_date.year - 1, anchor_date.month, 28) + timedelta(days=1)

    ly_target_end = ly_target_start + timedelta(days=29)

    # Check if last-year period is available in dataset
    dataset_min_date = data["event_date"].min()
    ly_available = (ly_target_start >= dataset_min_date)

    if not ly_available:
        # Return missing defaults with flag = 0.0
        return base_users_df.with_columns([
            pl.lit(0.0).cast(pl.Float32).alias("ly_window_available"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_buyer"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_gmv"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_orders"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_purchase_days"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_active_days"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_searches"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_carts"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_activity_vs_curr"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_gmv_vs_curr"),
        ])

    ly_slice = data.filter(
        (pl.col("user_id").is_in(user_ids))
        & (pl.col("event_date") >= ly_target_start)
        & (pl.col("event_date") <= ly_target_end)
    )

    if ly_slice.is_empty():
        return base_users_df.with_columns([
            pl.lit(1.0).cast(pl.Float32).alias("ly_window_available"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_buyer"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_gmv"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_orders"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_purchase_days"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_active_days"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_searches"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_target_window_carts"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_activity_vs_curr"),
            pl.lit(0.0).cast(pl.Float32).alias("ly_gmv_vs_curr"),
        ])

    ly_agg = ly_slice.group_by("user_id").agg([
        (pl.col("gmv").sum() > 0).cast(pl.Float32).alias("ly_target_window_buyer"),
        pl.col("gmv").sum().cast(pl.Float32).alias("ly_target_window_gmv"),
        pl.col("to_ord").sum().cast(pl.Float32).alias("ly_target_window_orders"),
        (pl.col("to_ord") > 0).sum().cast(pl.Float32).alias("ly_target_window_purchase_days"),
        pl.col("event_date").count().cast(pl.Float32).alias("ly_target_window_active_days"),
        pl.col("searches").sum().cast(pl.Float32).alias("ly_target_window_searches"),
        pl.col("to_cart").sum().cast(pl.Float32).alias("ly_target_window_carts"),
    ])

    res = base_users_df.join(ly_agg, on="user_id", how="left").with_columns([
        pl.lit(1.0).cast(pl.Float32).alias("ly_window_available")
    ])
    feat_cols = [c for c in res.columns if c != "user_id"]
    res = res.with_columns([pl.col("user_id").cast(pl.Int64)] + [pl.col(c).fill_null(0.0).cast(pl.Float32) for c in feat_cols])

    return res



def compute_all_transition_features(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
) -> pl.DataFrame:
    """Combines lifecycle, last-year, and calendar context features into a single Polars DataFrame."""
    lifecycle_df = compute_lifecycle_features(data, user_ids, anchor_date)
    ly_df = compute_last_year_features(data, user_ids, anchor_date)
    cal_dict = compute_calendar_context(anchor_date)

    combined = lifecycle_df.join(ly_df, on="user_id", how="left")

    # Add constant calendar features
    for col_name, val in cal_dict.items():
        combined = combined.with_columns(pl.lit(val).alias(col_name))

    return combined
