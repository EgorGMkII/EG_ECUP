"""Specialized Dataset & Manifest Builder for Specialized Hurdle Stack."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from src.specialized_hurdle.definitions import (
    TransitionState,
    compute_transition_state,
    ALL_AVAILABLE_ANCHORS,
    JANUARY_VALIDATION_ANCHOR,
)


def build_specialized_manifest_for_anchor(
    snapshot_path: Path,
    anchor_str: str,
    user_ids: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """Extracts ground truth past/future metrics and transition states for a given anchor."""
    snap = pl.read_parquet(snapshot_path)
    if user_ids is not None:
        snap = snap.filter(pl.col("user_id").is_in(set(user_ids)))

    anchor_dt = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    history_start = str(anchor_dt - timedelta(days=364))
    history_end = anchor_str
    target_start = str(anchor_dt + timedelta(days=1))
    target_end = str(anchor_dt + timedelta(days=30))

    # Past 30-day activity is determined strictly by past 30-day purchases / lifetime GMV
    past_gmv_col = "lifetime_gmv" if "lifetime_gmv" in snap.columns else "gmv_sum_30d"
    past_gmv_30d = snap[past_gmv_col].to_numpy().astype(np.float64)
    future_gmv_30d = snap["target"].to_numpy().astype(np.float64)
    target_z = np.log1p(np.maximum(0.0, future_gmv_30d)).astype(np.float32)

    was_active = (past_gmv_30d > 0).astype(bool)
    will_buy = (future_gmv_30d > 0).astype(bool)

    states = [
        compute_transition_state(bool(act), bool(buy)).value
        for act, buy in zip(was_active, will_buy)
    ]

    source_set = "validation" if anchor_str == JANUARY_VALIDATION_ANCHOR else "train"

    df_manifest = pl.DataFrame({
        "user_id": snap["user_id"],
        "anchor": [anchor_str] * len(snap),
        "history_start": [history_start] * len(snap),
        "history_end": [history_end] * len(snap),
        "target_start": [target_start] * len(snap),
        "target_end": [target_end] * len(snap),
        "past_gmv_30d": past_gmv_30d,
        "future_gmv_30d": future_gmv_30d,
        "target_z": target_z,
        "was_active": was_active,
        "will_buy": will_buy,
        "transition_state": states,
        "source_anchor_set": [source_set] * len(snap),
    })

    return df_manifest


def build_all_specialized_datasets(
    snapshots_dir: Path,
    out_dir: Path,
    anchors: List[str] = ALL_AVAILABLE_ANCHORS,
    user_ids: Optional[np.ndarray] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Builds the unified manifest and the 3 specialized datasets (React, Churn, Amount)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_chunks = []

    for anchor in anchors:
        snap_file = snapshots_dir / f"snapshot_{anchor}.parquet"
        if not snap_file.exists():
            continue
        chunk = build_specialized_manifest_for_anchor(snap_file, anchor, user_ids=user_ids)
        manifest_chunks.append(chunk)

    df_manifest = pl.concat(manifest_chunks)
    manifest_path = out_dir / "dataset_manifest.parquet"
    df_manifest.write_parquet(manifest_path)

    # 1. Reactivation Dataset: was_active == False, y_react = int(will_buy)
    df_react = df_manifest.filter(pl.col("was_active") == False).with_columns([
        pl.col("will_buy").cast(pl.Int32).alias("y_react")
    ])
    react_path = out_dir / "reactivation_dataset.parquet"
    df_react.write_parquet(react_path)

    # 2. Churn Dataset: was_active == True, y_churn = int(not will_buy)
    df_churn = df_manifest.filter(pl.col("was_active") == True).with_columns([
        (~pl.col("will_buy")).cast(pl.Int32).alias("y_churn")
    ])
    churn_path = out_dir / "churn_dataset.parquet"
    df_churn.write_parquet(churn_path)

    # 3. Amount Dataset: future_gmv_30d > 0, y_amount = log1p(future_gmv_30d)
    df_amount = df_manifest.filter(pl.col("future_gmv_30d") > 0.0).with_columns([
        pl.col("target_z").alias("y_amount")
    ])
    amount_path = out_dir / "amount_dataset.parquet"
    df_amount.write_parquet(amount_path)

    return df_manifest, df_react, df_churn, df_amount
