"""Cadence, Inter-Purchase Gaps, Cycle Phase, Regularity, and Streaks Feature Extractor (100% Vectorized Native Polars)."""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Union

import polars as pl


def extract_cadence_features_for_anchor(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    history_days: int = 180,
) -> pl.DataFrame:
    """Computes comprehensive anchor-safe cadence, gap, cycle phase, and regularity features using native Polars."""
    start_date = anchor_date - timedelta(days=history_days - 1)
    base_df = pl.DataFrame({"user_id": user_ids}, schema={"user_id": pl.Int64})

    # Filter logs strictly within [start_date, anchor_date]
    hist = data.filter(
        (pl.col("event_date") >= start_date)
        & (pl.col("event_date") <= anchor_date)
        & (pl.col("user_id").is_in(user_ids))
    )

    if hist.is_empty():
        return base_df.with_columns([
            pl.lit(365.0).alias("days_since_second_last_purchase"),
            pl.lit(365.0).alias("days_since_third_last_purchase"),
            pl.lit(365.0).alias("days_since_first_purchase"),
            pl.lit(0.0).alias("purchase_days_180d"),
            pl.lit(0.0).alias("active_weeks_30d"),
            pl.lit(0.0).alias("active_weeks_90d"),
            pl.lit(0.0).alias("active_weeks_180d"),
            pl.lit(0.0).alias("share_of_weeks_with_purchase_90d"),
            pl.lit(0.0).alias("share_of_weeks_with_purchase_180d"),
            pl.lit(180.0).alias("last_interpurchase_gap"),
            pl.lit(180.0).alias("previous_interpurchase_gap"),
            pl.lit(180.0).alias("mean_interpurchase_gap"),
            pl.lit(180.0).alias("median_interpurchase_gap"),
            pl.lit(0.0).alias("std_interpurchase_gap"),
            pl.lit(180.0).alias("min_interpurchase_gap"),
            pl.lit(180.0).alias("max_interpurchase_gap"),
            pl.lit(180.0).alias("p25_interpurchase_gap"),
            pl.lit(180.0).alias("p75_interpurchase_gap"),
            pl.lit(0.0).alias("cv_interpurchase_gap"),
            pl.lit(1.0).alias("trend_interpurchase_gap"),
            pl.lit(0.0).alias("number_of_observed_gaps"),
            pl.lit(0.0).alias("gap_ratio"),
            pl.lit(0.0).alias("overdue_days"),
            pl.lit(0.0).alias("gap_zscore"),
            pl.lit(0.0).alias("is_cycle_overdue"),
            pl.lit(0.0).alias("cycle_estimate_available"),
            pl.lit(0.0).alias("has_2_purchase_days"),
            pl.lit(0.0).alias("has_3_purchase_days"),
            pl.lit(180.0).alias("longest_inactivity_streak"),
            pl.lit(0.0).alias("mean_active_streak"),
            pl.lit(0.0).alias("purchase_week_regularness"),
            pl.lit(0.0).alias("active_week_entropy"),
            pl.lit(0.0).alias("is_history_censored"),
        ])

    # 1. Purchase Events Timeline
    purch_df = (
        hist.filter(pl.col("gmv") > 0)
        .select(["user_id", "event_date"])
        .unique()
        .sort(["user_id", "event_date"])
    )

    # Compute purchase gaps per user
    purch_gaps = (
        purch_df.with_columns(
            (pl.col("event_date") - pl.col("event_date").shift(1).over("user_id")).dt.total_days().alias("gap")
        )
    )

    # Purchase aggregates in native Polars
    purch_agg = purch_gaps.group_by("user_id").agg([
        pl.len().alias("purchase_days_180d"),
        (pl.lit(anchor_date) - pl.col("event_date").max()).dt.total_days().alias("days_since_last_p"),
        (pl.lit(anchor_date) - pl.col("event_date").min()).dt.total_days().alias("days_since_first_purchase"),
        (pl.lit(anchor_date) - pl.col("event_date").slice(-2, 1).first()).dt.total_days().alias("d_sec_raw"),
        (pl.lit(anchor_date) - pl.col("event_date").slice(-3, 1).first()).dt.total_days().alias("d_third_raw"),
        
        # Gaps
        pl.col("gap").drop_nulls().len().alias("number_of_observed_gaps"),
        pl.col("gap").drop_nulls().last().alias("last_gap_raw"),
        pl.col("gap").drop_nulls().slice(-2, 1).first().alias("prev_gap_raw"),
        pl.col("gap").drop_nulls().mean().alias("mean_gap_raw"),
        pl.col("gap").drop_nulls().median().alias("median_gap_raw"),
        pl.col("gap").drop_nulls().std().alias("std_gap_raw"),
        pl.col("gap").drop_nulls().min().alias("min_gap_raw"),
        pl.col("gap").drop_nulls().max().alias("max_gap_raw"),
        pl.col("gap").drop_nulls().quantile(0.25).alias("p25_gap_raw"),
        pl.col("gap").drop_nulls().quantile(0.75).alias("p75_gap_raw"),
        
        # Weekly coverage
        ((pl.lit(anchor_date) - pl.col("event_date")).dt.total_days() < 90).sum().alias("p_days_90d"),
    ])

    # 2. Activity Events Timeline
    act_df = (
        hist.select(["user_id", "event_date"])
        .unique()
        .sort(["user_id", "event_date"])
        .with_columns(
            (pl.col("event_date") - pl.col("event_date").shift(1).over("user_id")).dt.total_days().alias("act_gap"),
            ((pl.lit(anchor_date) - pl.col("event_date")).dt.total_days() // 7).alias("week_idx"),
        )
    )

    act_agg = act_df.group_by("user_id").agg([
        pl.len().alias("total_act_days"),
        (pl.lit(anchor_date) - pl.col("event_date").min()).dt.total_days().alias("user_history_span"),
        pl.col("act_gap").drop_nulls().max().alias("max_inact_gap"),
        (pl.col("week_idx") < 4).sum().alias("act_w_30d"),
        (pl.col("week_idx") < 13).sum().alias("act_w_90d"),
        pl.col("week_idx").n_unique().alias("act_w_180d"),
    ])

    # Join with base users
    res = (
        base_df.join(purch_agg, on="user_id", how="left")
        .join(act_agg, on="user_id", how="left")
    )

    # Post-process columns and add derived ratios
    res = res.with_columns([
        pl.col("purchase_days_180d").fill_null(0.0).cast(pl.Float32),
        pl.col("days_since_first_purchase").fill_null(365.0).cast(pl.Float32),
        pl.when(pl.col("purchase_days_180d") >= 2).then(pl.col("d_sec_raw")).otherwise(365.0).fill_null(365.0).cast(pl.Float32).alias("days_since_second_last_purchase"),
        pl.when(pl.col("purchase_days_180d") >= 3).then(pl.col("d_third_raw")).otherwise(365.0).fill_null(365.0).cast(pl.Float32).alias("days_since_third_last_purchase"),
        
        # Flags
        (pl.col("purchase_days_180d") >= 2).cast(pl.Float32).fill_null(0.0).alias("has_2_purchase_days"),
        (pl.col("purchase_days_180d") >= 3).cast(pl.Float32).fill_null(0.0).alias("has_3_purchase_days"),
        (pl.col("purchase_days_180d") >= 2).cast(pl.Float32).fill_null(0.0).alias("cycle_estimate_available"),
        
        # Gaps
        pl.col("number_of_observed_gaps").fill_null(0.0).cast(pl.Float32),
        pl.col("last_gap_raw").fill_null(180.0).cast(pl.Float32).alias("last_interpurchase_gap"),
        pl.col("prev_gap_raw").fill_null(180.0).cast(pl.Float32).alias("previous_interpurchase_gap"),
        pl.col("mean_gap_raw").fill_null(180.0).cast(pl.Float32).alias("mean_interpurchase_gap"),
        pl.col("median_gap_raw").fill_null(180.0).cast(pl.Float32).alias("median_interpurchase_gap"),
        pl.col("std_gap_raw").fill_null(0.0).cast(pl.Float32).alias("std_interpurchase_gap"),
        pl.col("min_gap_raw").fill_null(180.0).cast(pl.Float32).alias("min_interpurchase_gap"),
        pl.col("max_gap_raw").fill_null(180.0).cast(pl.Float32).alias("max_interpurchase_gap"),
        pl.col("p25_gap_raw").fill_null(180.0).cast(pl.Float32).alias("p25_interpurchase_gap"),
        pl.col("p75_gap_raw").fill_null(180.0).cast(pl.Float32).alias("p75_interpurchase_gap"),
        
        # Activity
        pl.col("act_w_30d").fill_null(0.0).cast(pl.Float32).alias("active_weeks_30d"),
        pl.col("act_w_90d").fill_null(0.0).cast(pl.Float32).alias("active_weeks_90d"),
        pl.col("act_w_180d").fill_null(0.0).cast(pl.Float32).alias("active_weeks_180d"),
        (pl.col("p_days_90d").fill_null(0.0) / 13.0).cast(pl.Float32).alias("share_of_weeks_with_purchase_90d"),
        (pl.col("purchase_days_180d").fill_null(0.0) / 26.0).cast(pl.Float32).alias("share_of_weeks_with_purchase_180d"),
        pl.col("max_inact_gap").fill_null(180.0).cast(pl.Float32).alias("longest_inactivity_streak"),
        (pl.col("total_act_days").fill_null(0.0) / (pl.col("act_w_180d").fill_null(1.0) + 1e-4)).cast(pl.Float32).alias("mean_active_streak"),
        (1.0 / (pl.col("std_gap_raw").fill_null(0.0) + 1.0)).cast(pl.Float32).alias("purchase_week_regularness"),
        (pl.col("act_w_180d").fill_null(0.0) / 26.0).cast(pl.Float32).alias("active_week_entropy"),
        (pl.col("user_history_span").fill_null(0.0) < 180.0).cast(pl.Float32).alias("is_history_censored"),
    ])

    # Dynamic Cycle Phase & Gap Ratios
    res = res.with_columns([
        (pl.col("std_interpurchase_gap") / (pl.col("mean_interpurchase_gap") + 1e-4)).alias("cv_interpurchase_gap"),
        (pl.col("last_interpurchase_gap") / (pl.col("previous_interpurchase_gap") + 1e-4)).alias("trend_interpurchase_gap"),
        pl.when(pl.col("cycle_estimate_available") > 0.5)
        .then(pl.col("days_since_last_p") / (pl.col("median_interpurchase_gap") + 1e-4))
        .otherwise(0.0)
        .alias("gap_ratio"),
        pl.when(pl.col("cycle_estimate_available") > 0.5)
        .then(pl.col("days_since_last_p") - pl.col("median_interpurchase_gap"))
        .otherwise(0.0)
        .alias("overdue_days"),
        pl.when(pl.col("cycle_estimate_available") > 0.5)
        .then((pl.col("days_since_last_p") - pl.col("mean_interpurchase_gap")) / (pl.col("std_interpurchase_gap") + 1e-4))
        .otherwise(0.0)
        .alias("gap_zscore"),
        pl.when((pl.col("cycle_estimate_available") > 0.5) & (pl.col("days_since_last_p") > pl.col("median_interpurchase_gap")))
        .then(1.0)
        .otherwise(0.0)
        .alias("is_cycle_overdue"),
    ])

    drop_raw = [c for c in res.columns if c.endswith("_raw") or c in ["days_since_last_p", "p_days_90d", "total_act_days", "user_history_span", "max_inact_gap"]]
    return res.drop(drop_raw)
