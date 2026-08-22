"""Module for purged time validation split, expanding window backtesting, and anchor weighting."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from src.snapshots import SNAPSHOTS_DIR


def get_snapshot_path(anchor: date, snapshots_dir: Path = SNAPSHOTS_DIR) -> Path:
    """Returns path to a specific anchor snapshot."""
    return snapshots_dir / f"snapshot_{anchor.strftime('%Y-%m-%d')}.parquet"


def get_purged_train_anchors(
    all_anchors: List[date],
    val_anchor: date,
    target_days: int = 30,
) -> List[date]:
    """Strictly filters training anchors whose 30-day target window has closed before val_anchor."""
    return [a for a in all_anchors if a + timedelta(days=target_days) <= val_anchor]


def compute_anchor_sample_weights(
    anchor_col: Union[np.ndarray, pl.Series],
    base_weight: float = 1.0,
    late_weight: float = 1.5,
    post_ny_weight: float = 2.5,
) -> np.ndarray:
    """Assigns sample weights based on snapshot freshness and regime relevance."""
    if isinstance(anchor_col, pl.Series):
        anchors = anchor_col.to_list()
    else:
        anchors = anchor_col

    weights = np.zeros(len(anchors), dtype=np.float32)
    for i, a in enumerate(anchors):
        if isinstance(a, str):
            a = date.fromisoformat(a)
        if a >= date(2026, 1, 1):
            weights[i] = post_ny_weight
        elif a >= date(2025, 11, 1):
            weights[i] = late_weight
        else:
            weights[i] = base_weight

    return weights


def load_purged_train_val(
    all_anchors: List[date],
    val_anchor: date,
    feature_cols: Optional[List[str]] = None,
    snapshots_dir: Path = SNAPSHOTS_DIR,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Loads purged expanding training panel and validation snapshot without data leakage."""
    train_anchors = get_purged_train_anchors(all_anchors, val_anchor)
    assert len(train_anchors) > 0, f"No purged training anchors found before val_anchor {val_anchor}"

    print(f"[*] Loading {len(train_anchors)} purged train anchors: {train_anchors[0]} ... {train_anchors[-1]}")
    train_dfs = [pl.read_parquet(get_snapshot_path(a, snapshots_dir)) for a in train_anchors]
    train_df = pl.concat(train_dfs)

    val_df = pl.read_parquet(get_snapshot_path(val_anchor, snapshots_dir))
    print(f"[*] Loaded train: {train_df.height:,} rows | val ({val_anchor}): {val_df.height:,} rows")

    return train_df, val_df
