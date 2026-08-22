"""Comprehensive Feature Engineering Engine for Multi-Window, Regime Shifts, Lifetime, Global, and Calendar features."""

import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

# Core metrics in log data
BASE_METRICS = [
    "gmv",
    "to_ord",
    "to_cart",
    "searches",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
]

WINDOWS = [
    ("7d", 7),
    ("14d", 14),
    ("30d", 30),
    ("60d", 60),
    ("90d", 90),
]

HOLIDAYS = [
    date(2025, 1, 1),
    date(2025, 2, 23),
    date(2025, 3, 8),
    date(2025, 11, 11),
    date(2026, 1, 1),
    date(2026, 2, 23),
    date(2026, 3, 8),
]


def build_window_aggregations(anchor: date, windows: List[Tuple[str, int]] = WINDOWS) -> List[pl.Expr]:
    """Generates windowed aggregations: sum, active_days, calendar/active means, max, std."""
    exprs: List[pl.Expr] = []
    a_lit = pl.lit(anchor)

    # Base active day indicators across activity types
    activity_expr = (pl.col("searches") > 0) | (pl.col("to_cart") > 0) | (pl.col("to_ord") > 0) | (pl.col("gmv") > 0)
    purchase_expr = (pl.col("to_ord") > 0) | (pl.col("gmv") > 0)
    search_expr = pl.col("searches") > 0
    cart_expr = pl.col("to_cart") > 0

    for name, days in windows:
        w_start = anchor - timedelta(days=days - 1)
        in_w = pl.col("event_date").is_between(w_start, anchor)

        # 1. Day counts
        exprs.extend([
            pl.when(in_w & activity_expr).then(1).otherwise(0).sum().alias(f"active_days_{name}"),
            pl.when(in_w & purchase_expr).then(1).otherwise(0).sum().alias(f"purchase_days_{name}"),
            pl.when(in_w & search_expr).then(1).otherwise(0).sum().alias(f"search_days_{name}"),
            pl.when(in_w & cart_expr).then(1).otherwise(0).sum().alias(f"cart_days_{name}"),
        ])

        # 2. Metric aggregations
        for m in BASE_METRICS:
            val_in_w = pl.when(in_w).then(pl.col(m)).otherwise(0.0)
            val_in_w_null = pl.when(in_w).then(pl.col(m)).otherwise(None)

            exprs.extend([
                val_in_w.sum().alias(f"{m}_sum_{name}"),
                (val_in_w.sum() / float(days)).alias(f"{m}_mean_cal_{name}"),
                (val_in_w.sum() / (pl.when(in_w & (pl.col(m) > 0)).then(1).otherwise(0).sum() + 1e-5)).alias(f"{m}_mean_act_{name}"),
                val_in_w.max().alias(f"{m}_max_{name}"),
                val_in_w_null.std().fill_null(0.0).alias(f"{m}_std_cal_{name}"),
            ])

    return exprs


def build_recency_and_lifetime_exprs(anchor: date) -> List[pl.Expr]:
    """Generates Recency, Lifetime, and inter-order interval features."""
    a_lit = pl.lit(anchor)
    return [
        (a_lit - pl.col("event_date").min()).dt.total_days().alias("customer_age_days"),
        (a_lit - pl.col("event_date").max()).dt.total_days().alias("days_since_last_activity"),
        (a_lit - pl.when(pl.col("searches") > 0).then(pl.col("event_date")).otherwise(None).max()).dt.total_days().alias("days_since_last_search"),
        (a_lit - pl.when(pl.col("to_cart") > 0).then(pl.col("event_date")).otherwise(None).max()).dt.total_days().alias("days_since_last_cart"),
        (a_lit - pl.when(pl.col("to_ord") > 0).then(pl.col("event_date")).otherwise(None).max()).dt.total_days().alias("days_since_last_order"),
        pl.when(pl.col("to_ord") > 0).then(1).otherwise(0).max().alias("has_ever_ordered"),
        pl.col("to_ord").sum().alias("lifetime_orders"),
        pl.col("gmv").sum().alias("lifetime_gmv"),
        pl.when(pl.col("to_ord") > 0).then(1).otherwise(0).sum().alias("lifetime_purchase_days"),
    ]


def build_time_series_dynamics_exprs(anchor: date) -> List[pl.Expr]:
    """Generates high-signal time-series dynamics and volatility features over 90-day history."""
    in_90d = pl.col("event_date").is_between(anchor - timedelta(days=89), anchor)
    in_14d = pl.col("event_date").is_between(anchor - timedelta(days=13), anchor)
    in_15_45d = pl.col("event_date").is_between(anchor - timedelta(days=44), anchor - timedelta(days=14))

    val_gmv_90d = pl.when(in_90d).then(pl.col("gmv")).otherwise(0.0)
    val_gmv_90d_null = pl.when(in_90d).then(pl.col("gmv")).otherwise(None)

    mean_gmv_90d = val_gmv_90d.sum() / 90.0

    is_weekend = in_90d & (pl.col("event_date").dt.weekday() >= 6)
    is_weekday = in_90d & (pl.col("event_date").dt.weekday() <= 5)

    return [
        # 1. Extreme spike count: days with GMV > 2.5 * mean
        pl.when(in_90d & (pl.col("gmv") > (mean_gmv_90d * 2.5))).then(1).otherwise(0).sum().alias("ts_gmv_spike_count_90d"),
        # 2. Conversion efficiency: purchase days / active search days
        (pl.when(in_90d & (pl.col("gmv") > 0)).then(1).otherwise(0).sum() / (pl.when(in_90d & (pl.col("searches") > 0)).then(1).otherwise(0).sum() + 1e-5)).alias("ts_active_conversion_efficiency"),
        # 3. Momentum & Acceleration: GMV last 14d / prior 30d
        (pl.when(in_14d).then(pl.col("gmv")).otherwise(0.0).sum() / (pl.when(in_15_45d).then(pl.col("gmv")).otherwise(0.0).sum() / 2.0 + 1e-5)).alias("ts_gmv_acceleration_ratio"),
        (pl.when(in_14d).then(pl.col("searches")).otherwise(0.0).sum() / (pl.when(in_15_45d).then(pl.col("searches")).otherwise(0.0).sum() / 2.0 + 1e-5)).alias("ts_search_acceleration_ratio"),
        # 4. Weekend vs weekday volume bias
        (pl.when(is_weekend).then(pl.col("gmv")).otherwise(0.0).sum() / (pl.when(is_weekday).then(pl.col("gmv")).otherwise(0.0).sum() / 2.5 + 1e-5)).alias("ts_weekend_bias_ratio"),
        # 5. Volatility and peak ratio
        (val_gmv_90d.max() / (mean_gmv_90d + 1e-5)).alias("ts_peak_to_mean_gmv_90d"),
        (val_gmv_90d_null.std().fill_null(0.0) / (mean_gmv_90d + 1e-5)).alias("ts_gmv_volatility_coef_90d"),
        val_gmv_90d_null.skew().fill_null(0.0).alias("ts_gmv_skewness_90d"),
        val_gmv_90d_null.kurtosis().fill_null(0.0).alias("ts_gmv_kurtosis_90d"),
    ]



def compute_global_platform_table(data: pl.DataFrame) -> pl.DataFrame:
    """Computes full daily platform aggregates across history for global trend features."""
    return (
        data.group_by("event_date")
        .agg(
            pl.col("user_id").n_unique().alias("global_dau"),
            pl.when((pl.col("to_ord") > 0) | (pl.col("gmv") > 0)).then(pl.col("user_id")).otherwise(None).n_unique().alias("global_buyers"),
            pl.col("gmv").sum().alias("global_gmv"),
            pl.col("to_ord").sum().alias("global_orders"),
        )
        .with_columns(
            (pl.col("global_buyers") / (pl.col("global_dau") + 1e-5)).alias("global_buyer_rate"),
            (pl.col("global_gmv") / (pl.col("global_dau") + 1e-5)).alias("global_gmv_per_active"),
            (pl.col("global_orders") / (pl.col("global_buyers") + 1e-5)).alias("global_orders_per_buyer"),
            (pl.col("global_gmv") / (pl.col("global_orders") + 1e-5)).alias("global_gmv_per_order"),
        )
        .sort("event_date")
    )


def build_global_and_calendar_features(anchor: date, global_table: pl.DataFrame) -> Dict[str, float]:
    """Extracts platform-level historical macro stats and target window calendar/holiday features."""
    # Historical slice strictly <= anchor
    hist_global = global_table.filter(pl.col("event_date") <= anchor)

    g7 = hist_global.tail(7)
    g30 = hist_global.tail(30)

    g_dau_7d = float(g7["global_dau"].mean()) if len(g7) > 0 else 0.0
    g_dau_30d = float(g30["global_dau"].mean()) if len(g30) > 0 else 0.0
    g_gmv_per_active_7d = float(g7["global_gmv_per_active"].mean()) if len(g7) > 0 else 0.0
    g_gmv_per_active_30d = float(g30["global_gmv_per_active"].mean()) if len(g30) > 0 else 0.0
    g_buyer_rate_30d = float(g30["global_buyer_rate"].mean()) if len(g30) > 0 else 0.0

    # Target window dates [anchor + 1d, anchor + 30d]
    t_start = anchor + timedelta(days=1)
    t_mid = anchor + timedelta(days=15)
    t_end = anchor + timedelta(days=30)

    t_mid_doy = float(t_mid.timetuple().tm_yday)
    sin_doy = math.sin(2.0 * math.pi * t_mid_doy / 365.25)
    cos_doy = math.cos(2.0 * math.pi * t_mid_doy / 365.25)

    # Holidays in target window
    holidays_in_target = [h for h in HOLIDAYS if t_start <= h <= t_end]
    has_ny = 1.0 if any(h.month == 1 and h.day == 1 for h in holidays_in_target) else 0.0
    has_feb23 = 1.0 if any(h.month == 2 and h.day == 23 for h in holidays_in_target) else 0.0
    has_mar8 = 1.0 if any(h.month == 3 and h.day == 8 for h in holidays_in_target) else 0.0
    has_bf = 1.0 if any(h.month == 11 and h.day == 11 for h in holidays_in_target) else 0.0

    # Distance to nearest holiday from anchor
    holiday_diffs = [(h - anchor).days for h in HOLIDAYS if (h - anchor).days >= 0]
    days_to_nearest = float(min(holiday_diffs)) if holiday_diffs else 999.0

    return {
        "global_dau_7d_mean": g_dau_7d,
        "global_dau_30d_mean": g_dau_30d,
        "global_gmv_per_active_7d": g_gmv_per_active_7d,
        "global_gmv_per_active_30d": g_gmv_per_active_30d,
        "global_buyer_rate_30d": g_buyer_rate_30d,
        "anchor_month": float(anchor.month),
        "anchor_day_of_week": float(anchor.weekday()),
        "anchor_day_of_year": float(anchor.timetuple().tm_yday),
        "target_start_doy": float(t_start.timetuple().tm_yday),
        "target_midpoint_doy": t_mid_doy,
        "target_end_doy": float(t_end.timetuple().tm_yday),
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
        "target_contains_new_year": has_ny,
        "target_contains_feb23": has_feb23,
        "target_contains_mar8": has_mar8,
        "target_contains_black_friday": has_bf,
        "target_contains_any_holiday": 1.0 if holidays_in_target else 0.0,
        "number_of_holidays_in_target": float(len(holidays_in_target)),
        "days_to_nearest_holiday": days_to_nearest,
    }


def compute_derived_rates_and_segments(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates rates, rate differences, ratios with zero-denom flags, recency ratios, and segments."""
    derived_exprs: List[pl.Expr] = []

    # 1. Daily intensities & diffs for key metrics
    key_metrics = ["gmv", "to_ord", "to_cart", "searches"]
    for m in key_metrics:
        s7 = pl.col(f"{m}_sum_7d")
        s14 = pl.col(f"{m}_sum_14d")
        s30 = pl.col(f"{m}_sum_30d")
        s60 = pl.col(f"{m}_sum_60d")
        s90 = pl.col(f"{m}_sum_90d")

        r7 = s7 / 7.0
        r14 = s14 / 14.0
        r30 = s30 / 30.0
        r60 = s60 / 60.0
        r90 = s90 / 90.0

        derived_exprs.extend([
            r7.alias(f"{m}_rate_7d"),
            r14.alias(f"{m}_rate_14d"),
            r30.alias(f"{m}_rate_30d"),
            r60.alias(f"{m}_rate_60d"),
            r90.alias(f"{m}_rate_90d"),
            (r7 - r30).alias(f"{m}_rate_diff_7_30d"),
            (r14 - r60).alias(f"{m}_rate_diff_14_60d"),
            (r30 - r90).alias(f"{m}_rate_diff_30_90d"),
            # Ratios with zero denominator indicators
            (s7 / (s30 + 1e-5)).alias(f"{m}_ratio_7_30d"),
            pl.when(s30 == 0).then(1.0).otherwise(0.0).alias(f"{m}_is_zero_denom_7_30d"),
            (s14 / (s60 + 1e-5)).alias(f"{m}_ratio_14_60d"),
            pl.when(s60 == 0).then(1.0).otherwise(0.0).alias(f"{m}_is_zero_denom_14_60d"),
            (s30 / (s90 + 1e-5)).alias(f"{m}_ratio_30_90d"),
            pl.when(s90 == 0).then(1.0).otherwise(0.0).alias(f"{m}_is_zero_denom_30_90d"),
        ])

    # Day counts rates & ratios
    for m in ["active_days", "purchase_days", "search_days", "cart_days"]:
        d7 = pl.col(f"{m}_7d")
        d14 = pl.col(f"{m}_14d")
        d30 = pl.col(f"{m}_30d")
        d60 = pl.col(f"{m}_60d")
        d90 = pl.col(f"{m}_90d")

        r7 = d7 / 7.0
        r14 = d14 / 14.0
        r30 = d30 / 30.0
        r60 = d60 / 60.0
        r90 = d90 / 90.0

        derived_exprs.extend([
            r7.alias(f"{m}_rate_7d"),
            r14.alias(f"{m}_rate_14d"),
            r30.alias(f"{m}_rate_30d"),
            r60.alias(f"{m}_rate_60d"),
            r90.alias(f"{m}_rate_90d"),
            (r7 - r30).alias(f"{m}_rate_diff_7_30d"),
            (r14 - r60).alias(f"{m}_rate_diff_14_60d"),
            (r30 - r90).alias(f"{m}_rate_diff_30_90d"),
            (d7 / (d30 + 1e-5)).alias(f"{m}_ratio_7_30d"),
            pl.when(d30 == 0).then(1.0).otherwise(0.0).alias(f"{m}_is_zero_denom_7_30d"),
            (d14 / (d60 + 1e-5)).alias(f"{m}_ratio_14_60d"),
            pl.when(d60 == 0).then(1.0).otherwise(0.0).alias(f"{m}_is_zero_denom_14_60d"),
            (d30 / (d90 + 1e-5)).alias(f"{m}_ratio_30_90d"),
            pl.when(d90 == 0).then(1.0).otherwise(0.0).alias(f"{m}_is_zero_denom_30_90d"),
        ])

    # 2. Inter-order regularity & recency ratio
    # Average purchase interval approx = 90 / (purchase_days_90d + 1e-5)
    mean_interval = 90.0 / (pl.col("purchase_days_90d") + 1e-5)
    derived_exprs.extend([
        mean_interval.alias("approx_order_interval_90d"),
        (pl.col("days_since_last_order") / (mean_interval + 1e-5)).alias("recency_ratio_90d"),
        # Relative user vs global metrics
        (pl.col("gmv_sum_30d") / (pl.col("global_gmv_per_active_30d") + 1e-5)).alias("user_gmv_vs_global_30d"),
        (pl.col("active_days_30d") / (pl.col("global_dau_30d_mean") * 30.0 / 250000.0 + 1e-5)).alias("user_active_vs_global_30d"),
    ])

    # 3. User Segment (Integer encoded: 0=cold_start, 1=never_active, 2=never_dormant, 3=returning_active, 4=returning_dormant, 5=fully_dormant)
    segment_expr = (
        pl.when(pl.col("customer_age_days") < 30)
        .then(0)  # cold_start
        .when(pl.col("days_since_last_activity") >= 90)
        .then(5)  # fully_dormant_90d
        .when((pl.col("has_ever_ordered") == 0) & (pl.col("days_since_last_activity") <= 14))
        .then(1)  # never_buyer_active
        .when((pl.col("has_ever_ordered") == 0) & (pl.col("days_since_last_activity") > 14))
        .then(2)  # never_buyer_dormant
        .when((pl.col("has_ever_ordered") == 1) & (pl.col("days_since_last_activity") <= 14))
        .then(3)  # returning_buyer_active
        .otherwise(4)  # returning_buyer_dormant
        .alias("user_segment_id")
    )
    derived_exprs.append(segment_expr)

    return df.with_columns(derived_exprs)
