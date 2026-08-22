"""Transition State Labels Definition & Categorization."""

from typing import Dict, Tuple
import numpy as np
import polars as pl

TRANSITION_STATES = {
    0: "0 -> 0 (Stable Sleep)",
    1: "0 -> >0 (Reactivation)",
    2: ">0 -> 0 (Churn)",
    3: ">0 -> >0 (Retention)",
}


def compute_transition_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Computes binary indicator flags and transition states from past_gmv_30d and target."""
    return df.with_columns([
        (pl.col("gmv_sum_30d") > 0).cast(pl.Int32).alias("past_buyer_30d"),
        (pl.col("target") > 0).cast(pl.Int32).alias("future_buyer_30d"),
    ]).with_columns([
        pl.when((pl.col("past_buyer_30d") == 0) & (pl.col("future_buyer_30d") == 0)).then(pl.lit(0))
        .when((pl.col("past_buyer_30d") == 0) & (pl.col("future_buyer_30d") == 1)).then(pl.lit(1))
        .when((pl.col("past_buyer_30d") == 1) & (pl.col("future_buyer_30d") == 0)).then(pl.lit(2))
        .otherwise(pl.lit(3))
        .alias("transition_state"),
        # Reactivation target: only meaningful when past_buyer_30d == 0
        pl.when(pl.col("past_buyer_30d") == 0).then(pl.col("future_buyer_30d")).otherwise(None).alias("y_reactivation"),
        # Churn target: only meaningful when past_buyer_30d == 1 (1 if churned to 0, 0 if retained)
        pl.when(pl.col("past_buyer_30d") == 1).then(1 - pl.col("future_buyer_30d")).otherwise(None).alias("y_churn"),
    ])
