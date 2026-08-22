"""Module for multi-anchor panel snapshot dataset generation, storage, and validation."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from src.features import (
    build_global_and_calendar_features,
    build_recency_and_lifetime_exprs,
    build_time_series_dynamics_exprs,
    build_window_aggregations,
    compute_derived_rates_and_segments,
    compute_global_platform_table,
)


DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
ARTIFACTS_DIR = Path("artifacts") if Path("artifacts").exists() else Path(".")
TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")


def get_or_create_selected_users(
    data_path: Union[str, Path] = TRAIN_PARQUET,
    n_users: int = 100_000,
    seed: int = 42,
    save_path: Union[str, Path] = ARTIFACTS_DIR / "selected_users_100k.parquet",
) -> List[int]:
    """Deterministically samples and saves fixed subset of users for panel snapshots."""
    save_path = Path(save_path)
    if not save_path.exists() and Path(save_path.name).exists():
        save_path = Path(save_path.name)
    if save_path.exists():
        users_df = pl.read_parquet(save_path)
        return users_df["user_id"].to_list()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    all_users = (
        pl.scan_parquet(data_path)
        .select(pl.col("user_id").unique())
        .collect()["user_id"]
        .sort()
    )

    np.random.seed(seed)
    sampled_indices = np.random.choice(len(all_users), size=n_users, replace=False)
    selected_users = all_users[sorted(sampled_indices)].to_list()

    pl.DataFrame({"user_id": selected_users}).write_parquet(save_path)
    print(f"[+] Saved {len(selected_users):,} selected users to {save_path}")
    return selected_users


def generate_panel_anchors(
    data_start: date = date(2025, 1, 1),
    data_end: date = date(2026, 2, 13),
    history_days: int = 90,
    target_days: int = 30,
    step_days: int = 14,
) -> List[date]:
    """Generates anchor dates with full history and known target windows."""
    min_anchor = data_start + timedelta(days=history_days - 1)
    max_anchor = data_end - timedelta(days=target_days)

    anchors = []
    current = min_anchor
    while current <= max_anchor:
        anchors.append(current)
        current += timedelta(days=step_days)

    if anchors[-1] != max_anchor:
        anchors.append(max_anchor)

    return sorted(list(set(anchors)))


def build_snapshot(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    global_table: Optional[pl.DataFrame] = None,
    history_days: int = 90,
    target_days: int = 30,
    is_test: bool = False,
) -> pl.DataFrame:
    """Constructs a complete feature snapshot for given anchor date on user subset."""
    if global_table is None:
        global_table = compute_global_platform_table(data)

    hist_start = anchor_date - timedelta(days=history_days - 1)
    target_start = anchor_date + timedelta(days=1)
    target_end = anchor_date + timedelta(days=target_days)

    # 1. Filter historical data strictly <= anchor_date
    hist_data = data.filter(
        pl.col("user_id").is_in(user_ids)
        & pl.col("event_date").is_between(hist_start, anchor_date)
    )

    # 2. Windowed aggregations, recency, and time-series dynamics
    agg_exprs = (
        build_window_aggregations(anchor_date)
        + build_recency_and_lifetime_exprs(anchor_date)
        + build_time_series_dynamics_exprs(anchor_date)
    )
    hist_features = (
        hist_data.group_by("user_id")
        .agg(agg_exprs)
    )

    # 3. Join with master user index
    index_df = pl.DataFrame({"user_id": user_ids})
    snapshot = index_df.join(hist_features, on="user_id", how="left")

    # 4. Fill missing values (999.0 for recencies, 0.0 for counts/sums)
    feature_cols = [c for c in snapshot.columns if c != "user_id"]
    rec_cols = [c for c in feature_cols if c.startswith("days_since_") or c == "customer_age_days"]
    num_cols = [c for c in feature_cols if c not in rec_cols]

    fill_exprs = [pl.col(c).fill_null(999.0) for c in rec_cols] + [pl.col(c).fill_null(0.0) for c in num_cols]
    snapshot = snapshot.with_columns(fill_exprs)

    # 5. Global platform metrics & Calendar features
    global_calendar_dict = build_global_and_calendar_features(anchor_date, global_table)
    snapshot = snapshot.with_columns([pl.lit(v).alias(k) for k, v in global_calendar_dict.items()])

    # 6. Derived rates, ratios, and segments
    snapshot = compute_derived_rates_and_segments(snapshot)

    # 7. Technical tracking columns
    avail_days = min(history_days, (anchor_date - date(2025, 1, 1)).days + 1)
    tech_cols = [
        pl.lit(anchor_date).alias("anchor_date"),
        pl.lit(hist_start).alias("history_start"),
        pl.lit(anchor_date).alias("history_end"),
        pl.lit(target_start).alias("target_start"),
        pl.lit(target_end).alias("target_end"),
        pl.lit(avail_days).alias("available_history_days"),
    ]
    snapshot = snapshot.with_columns(tech_cols)

    # 8. Target Calculation (if not test)
    if not is_test:
        target_data = data.filter(
            pl.col("user_id").is_in(user_ids)
            & pl.col("event_date").is_between(target_start, target_end)
        )
        target_agg = (
            target_data.group_by("user_id")
            .agg(pl.col("gmv").sum().alias("target"))
        )
        snapshot = snapshot.join(target_agg, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0)
        )
        snapshot = snapshot.with_columns(
            (pl.col("target") > 0).cast(pl.Int32).alias("will_buy_30d")
        )
    else:
        snapshot = snapshot.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("target"),
            pl.lit(None, dtype=pl.Int32).alias("will_buy_30d"),
        ])

    verify_snapshot_integrity(snapshot, expected_n_users=len(user_ids), expected_history_days=history_days, is_test=is_test)
    return snapshot


def verify_snapshot_integrity(
    snapshot: pl.DataFrame,
    expected_n_users: int = 100_000,
    expected_history_days: int = 90,
    is_test: bool = False,
) -> bool:
    """Performs rigorous quality and anti-leak assertions on a generated snapshot."""
    assert snapshot.height == expected_n_users, f"Expected {expected_n_users} rows, got {snapshot.height}"
    assert snapshot["user_id"].n_unique() == expected_n_users, "Duplicate user_ids found in snapshot"
    assert snapshot["anchor_date"].n_unique() == 1, "Snapshot contains multiple anchor dates"

    anchor = snapshot["anchor_date"][0]
    assert (snapshot["history_end"] <= anchor).all(), "Data leak: events after anchor found in history"

    if not is_test:
        assert (snapshot["target_start"] == anchor + timedelta(days=1)).all(), "Target start mismatch"
        assert (snapshot["target_end"] == anchor + timedelta(days=30)).all(), "Target end mismatch"
        assert snapshot["target"].is_null().sum() == 0, "Null targets found"

    return True


def generate_and_save_all_snapshots(
    data: Optional[pl.DataFrame] = None,
    snapshots_dir: Union[str, Path] = SNAPSHOTS_DIR,
    n_users: int = 100_000,
) -> List[Path]:
    """Generates and persists all panel snapshots to parquet files."""
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        print("[*] Loading raw train.parquet...")
        data = pl.read_parquet(TRAIN_PARQUET)

    user_ids = get_or_create_selected_users(n_users=n_users)
    anchors = generate_panel_anchors()
    global_table = compute_global_platform_table(data)

    print(f"[*] Starting panel generation: {len(anchors)} anchors for {len(user_ids):,} users...")
    saved_paths = []

    for i, a in enumerate(anchors):
        file_path = snapshots_dir / f"snapshot_{a.strftime('%Y-%m-%d')}.parquet"
        print(f"  [{i+1}/{len(anchors)}] Generating snapshot for anchor {a}...")
        snap = build_snapshot(data, user_ids, a, global_table=global_table)
        snap.write_parquet(file_path)
        saved_paths.append(file_path)
        print(f"      -> Saved {file_path.name} ({snap.shape[0]:,} rows, {snap.shape[1]} cols)")

    # Also build test snapshot (anchor: 2026-02-13)
    test_anchor = date(2026, 2, 13)
    test_path = snapshots_dir / f"snapshot_{test_anchor.strftime('%Y-%m-%d')}_test.parquet"
    print(f"[*] Generating test snapshot for anchor {test_anchor}...")
    test_snap = build_snapshot(data, user_ids, test_anchor, global_table=global_table, is_test=True)
    test_snap.write_parquet(test_path)
    saved_paths.append(test_path)
    print(f"      -> Saved test {test_path.name}")

    print(f"[+] All {len(saved_paths)} snapshots successfully generated in {snapshots_dir}!")
    return saved_paths
