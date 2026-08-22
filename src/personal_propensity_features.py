"""Personal Transition Propensity and Exact Target-Aligned Last-Year Feature Engines."""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from src.sequential.dataset import extract_anchor_targets


def compute_personal_propensity_features(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    available_anchors: List[date],
) -> pl.DataFrame:
    """Computes leakage-safe personal propensity and transition history across completed past target windows."""
    # Strict leakage boundary: window must be completely finished before anchor_date
    completed_anchors = [a for a in available_anchors if (a + timedelta(days=30)) <= anchor_date]
    n_users = len(user_ids)
    base_df = pl.DataFrame({"user_id": user_ids}, schema={"user_id": pl.Int64})

    if not completed_anchors:
        # No completed past target windows available
        return base_df.with_columns([
            pl.lit(0.0).alias("personal_buy_rate_raw"),
            pl.lit(0.5398).alias("personal_buy_rate_k5"),
            pl.lit(0.5398).alias("personal_buy_rate_k20"),
            pl.lit(0.2800).alias("personal_react_rate_k5"),
            pl.lit(0.2570).alias("personal_churn_rate_k5"),
            pl.lit(0.0).alias("personal_mean_log_gmv"),
            pl.lit(0.0).alias("n_completed_past_target_windows"),
            pl.lit(0.0).alias("propensity_available_flag"),
        ])

    # Extract past outcomes across all completed anchors
    buy_history = []
    gmv_history = []

    for a in completed_anchors:
        t_start = a + timedelta(days=1)
        t_end = a + timedelta(days=30)
        hist_a = data.filter((pl.col("event_date") >= t_start) & (pl.col("event_date") <= t_end))
        agg_a = (
            hist_a.group_by("user_id")
            .agg(pl.col("gmv").sum().alias("win_gmv"))
        )
        joined_a = base_df.join(agg_a, on="user_id", how="left").with_columns(pl.col("win_gmv").fill_null(0.0))
        gmv_arr = joined_a["win_gmv"].to_numpy()
        buy_history.append((gmv_arr > 0).astype(np.float32))
        gmv_history.append(np.log1p(gmv_arr).astype(np.float32))

    buy_mat = np.column_stack(buy_history)  # [N, n_completed]
    gmv_mat = np.column_stack(gmv_history)  # [N, n_completed]

    n_comp = buy_mat.shape[1]
    buy_counts = np.sum(buy_mat, axis=1)
    raw_rate = buy_counts / float(n_comp)

    # Bayesian m-estimate smoothing
    k5_rate = (buy_counts + 5.0 * 0.5398) / (float(n_comp) + 5.0)
    k20_rate = (buy_counts + 20.0 * 0.5398) / (float(n_comp) + 20.0)

    # Mean log GMV on positive events
    pos_counts = np.maximum(buy_counts, 1e-5)
    mean_log_gmv = np.where(buy_counts > 0, np.sum(gmv_mat, axis=1) / pos_counts, 0.0)

    # Reactivation and Churn propensities
    react_rate = (buy_counts + 5.0 * 0.28) / (float(n_comp) + 5.0)
    churn_rate = ((float(n_comp) - buy_counts) + 5.0 * 0.26) / (float(n_comp) + 5.0)

    return base_df.with_columns([
        pl.Series("personal_buy_rate_raw", raw_rate.astype(np.float32)),
        pl.Series("personal_buy_rate_k5", k5_rate.astype(np.float32)),
        pl.Series("personal_buy_rate_k20", k20_rate.astype(np.float32)),
        pl.Series("personal_react_rate_k5", react_rate.astype(np.float32)),
        pl.Series("personal_churn_rate_k5", churn_rate.astype(np.float32)),
        pl.Series("personal_mean_log_gmv", mean_log_gmv.astype(np.float32)),
        pl.lit(float(n_comp)).alias("n_completed_past_target_windows"),
        pl.lit(1.0).alias("propensity_available_flag"),
    ])


def compute_exact_last_year_target_features(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
) -> pl.DataFrame:
    """Computes exact last-year target window analog [anchor + 1d - 365d, anchor + 30d - 365d]."""
    base_df = pl.DataFrame({"user_id": user_ids}, schema={"user_id": pl.Int64})

    ly_start = anchor_date + timedelta(days=1) - timedelta(days=365)
    ly_end = anchor_date + timedelta(days=30) - timedelta(days=365)

    data_min_date = date(2025, 1, 1)
    is_avail = (ly_start >= data_min_date)

    if not is_avail:
        return base_df.with_columns([
            pl.lit(0.0).alias("ly_exact_target_buy"),
            pl.lit(0.0).alias("ly_exact_target_gmv"),
            pl.lit(0.0).alias("ly_exact_target_log_gmv"),
            pl.lit(0.0).alias("ly_exact_target_purchase_days"),
            pl.lit(0.0).alias("ly_exact_target_orders"),
            pl.lit(0.0).alias("ly_exact_target_searches"),
            pl.lit(0.0).alias("ly_exact_target_carts"),
            pl.lit(0.0).alias("ly_exact_target_active_days"),
            pl.lit(0.0).alias("ly_exact_target_available"),
        ])

    user_set = set(user_ids)
    ly_hist = data.filter(
        (pl.col("event_date") >= ly_start)
        & (pl.col("event_date") <= ly_end)
        & (pl.col("user_id").is_in(user_set))
    )

    if ly_hist.is_empty():
        return base_df.with_columns([
            pl.lit(0.0).alias("ly_exact_target_buy"),
            pl.lit(0.0).alias("ly_exact_target_gmv"),
            pl.lit(0.0).alias("ly_exact_target_log_gmv"),
            pl.lit(0.0).alias("ly_exact_target_purchase_days"),
            pl.lit(0.0).alias("ly_exact_target_orders"),
            pl.lit(0.0).alias("ly_exact_target_searches"),
            pl.lit(0.0).alias("ly_exact_target_carts"),
            pl.lit(0.0).alias("ly_exact_target_active_days"),
            pl.lit(1.0).alias("ly_exact_target_available"),
        ])

    agg = (
        ly_hist.group_by("user_id")
        .agg([
            pl.col("gmv").sum().alias("gmv_sum"),
            pl.col("to_ord").sum().alias("ord_sum"),
            pl.col("searches").sum().alias("search_sum"),
            pl.col("to_cart").sum().alias("cart_sum"),
            ((pl.col("gmv") > 0) | (pl.col("to_ord") > 0)).sum().alias("purch_days"),
            pl.len().alias("act_days"),
        ])
    )

    joined = base_df.join(agg, on="user_id", how="left").with_columns([
        pl.col("gmv_sum").fill_null(0.0),
        pl.col("ord_sum").fill_null(0.0),
        pl.col("search_sum").fill_null(0.0),
        pl.col("cart_sum").fill_null(0.0),
        pl.col("purch_days").fill_null(0.0),
        pl.col("act_days").fill_null(0.0),
    ])

    return joined.select([
        "user_id",
        (pl.col("gmv_sum") > 0).cast(pl.Float32).alias("ly_exact_target_buy"),
        pl.col("gmv_sum").cast(pl.Float32).alias("ly_exact_target_gmv"),
        pl.col("gmv_sum").log1p().cast(pl.Float32).alias("ly_exact_target_log_gmv"),
        pl.col("purch_days").cast(pl.Float32).alias("ly_exact_target_purchase_days"),
        pl.col("ord_sum").cast(pl.Float32).alias("ly_exact_target_orders"),
        pl.col("search_sum").cast(pl.Float32).alias("ly_exact_target_searches"),
        pl.col("cart_sum").cast(pl.Float32).alias("ly_exact_target_carts"),
        pl.col("act_days").cast(pl.Float32).alias("ly_exact_target_active_days"),
        pl.lit(1.0).alias("ly_exact_target_available"),
    ])
