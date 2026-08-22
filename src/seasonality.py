"""Special Seasonality Experiment for 2025 Anchor and Final Test Set Drift Analysis."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from src.features import compute_global_platform_table
from src.snapshots import build_snapshot, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET


def evaluate_seasonal_uplift_2025(data: Optional[pl.DataFrame] = None) -> Dict:
    """Constructs 2025-02-13 snapshot (calendar twin of test) and computes seasonal uplift index."""
    if data is None:
        data = pl.read_parquet(TRAIN_PARQUET)

    user_ids = get_or_create_selected_users(n_users=100_000)
    global_table = compute_global_platform_table(data)

    # 1. Target window for Feb 13, 2025 (Feb 14 -> Mar 15: covers Feb 23 & Mar 8)
    snap_feb2025 = build_snapshot(
        data=data,
        user_ids=user_ids,
        anchor_date=date(2025, 2, 13),
        global_table=global_table,
        history_days=44,  # Available history from Jan 1
        target_days=30,
    )

    # 2. Baseline comparison window in regular Spring (e.g. Mar 31, 2025)
    snap_mar2025 = pl.read_parquet(SNAPSHOTS_DIR / "snapshot_2025-03-31.parquet")

    gmv_feb_holiday = float(snap_feb2025["target"].sum())
    gmv_mar_regular = float(snap_mar2025["target"].sum())
    buyer_rate_feb = float((snap_feb2025["target"] > 0).mean())
    buyer_rate_mar = float((snap_mar2025["target"] > 0).mean())

    seasonal_index_gmv = gmv_feb_holiday / (gmv_mar_regular + 1e-5)
    seasonal_index_buyers = buyer_rate_feb / (buyer_rate_mar + 1e-5)

    return {
        "gmv_feb_holiday_2025": gmv_feb_holiday,
        "gmv_mar_regular_2025": gmv_mar_regular,
        "buyer_rate_feb_2025": buyer_rate_feb,
        "buyer_rate_mar_2025": buyer_rate_mar,
        "seasonal_index_gmv": seasonal_index_gmv,
        "seasonal_index_buyers": seasonal_index_buyers,
    }


def analyze_test_drift(
    last_train_path: Path = SNAPSHOTS_DIR / "snapshot_2026-01-14.parquet",
    test_path: Path = SNAPSHOTS_DIR / "snapshot_2026-02-13_test.parquet",
    top_n_features: int = 15,
) -> pl.DataFrame:
    """Computes KS statistics, zero rates, and PSI drift between last labeled train anchor and test."""
    df_train = pl.read_parquet(last_train_path)
    df_test = pl.read_parquet(test_path)

    feature_cols = [c for c in df_train.columns if c not in {
        "user_id", "anchor_date", "history_start", "history_end", "target_start", "target_end", "available_history_days", "target", "will_buy_30d"
    }]

    drift_records = []
    for col in feature_cols:
        tr_vals = df_train[col].to_numpy()
        ts_vals = df_test[col].to_numpy()

        ks_stat, p_val = ks_2samp(tr_vals, ts_vals)
        z_tr = np.mean(tr_vals == 0)
        z_ts = np.mean(ts_vals == 0)

        drift_records.append({
            "feature": col,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
            "train_mean": float(np.mean(tr_vals)),
            "test_mean": float(np.mean(ts_vals)),
            "train_zero_rate": float(z_tr),
            "test_zero_rate": float(z_ts),
        })

    drift_df = pl.DataFrame(drift_records).sort("ks_statistic", descending=True)
    return drift_df
