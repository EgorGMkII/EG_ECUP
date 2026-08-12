import glob
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import polars as pl

# Default settings and configurations
DEFAULT_AGGS: List[str] = ["sum", "max", "std", "mean"]
DEFAULT_VALUE_COLS: List[str] = ["gmv", "searches"]
DEFAULT_WINDOWS: List[Tuple[str, int, int]] = [
    ("30d", 29, 0),   # [t-29, t-0] (includes anchor date)
    ("60d", 59, 30),  # [t-59, t-30]
    ("90d", 89, 60),  # [t-89, t-60]
]
FEATURES_DIR: str = "data/v2/features"
N_FOLDS: int = 4
BATCH_SIZE: int = 50_000


def generate_cv_anchor_dates(
    data: pl.DataFrame,
    prediction_horizon_days: int = 30,
    stride_days: int = 14,
    min_history_days: int = 90,
    n_folds: Optional[int] = N_FOLDS,
) -> List[date]:
    """Generates time-CV anchor dates for rolling window validation.

    Args:
        data: Polars DataFrame containing 'event_date'.
        prediction_horizon_days: Forecast horizon in days (e.g. 30).
        stride_days: Step size between consecutive anchor dates.
        min_history_days: Minimum historical days required prior to anchor.
        n_folds: Number of latest folds to return.

    Returns:
        List of sorted anchor dates.
    """
    min_date: date = data["event_date"].min()
    max_date: date = data["event_date"].max()

    latest_anchor = max_date - timedelta(days=prediction_horizon_days)
    earliest_anchor = min_date + timedelta(days=min_history_days - 1)

    if latest_anchor < earliest_anchor:
        raise ValueError(
            f"Not enough data: need >= {min_history_days + prediction_horizon_days - 1} days, "
            f"got {(max_date - min_date).days} days."
        )

    n_steps = (latest_anchor - earliest_anchor).days // stride_days
    all_anchors = [
        latest_anchor - timedelta(days=i * stride_days)
        for i in range(n_steps + 1)
    ]
    all_anchors = sorted(all_anchors)

    if n_folds is not None:
        if n_folds <= 0:
            raise ValueError("n_folds must be a positive integer.")
        return all_anchors[-n_folds:]

    return all_anchors


def build_window_agg_exprs(
    anchor_val: date,
    windows: List[Tuple[str, int, int]] = DEFAULT_WINDOWS,
    value_cols: List[str] = DEFAULT_VALUE_COLS,
    aggs: List[str] = DEFAULT_AGGS,
) -> List[pl.Expr]:
    """Constructs native Polars expressions for window-based feature aggregations.

    Args:
        anchor_val: Target anchor date.
        windows: List of (window_name, start_offset_days, end_offset_days).
        value_cols: Columns to aggregate (e.g. ['gmv', 'searches']).
        aggs: Aggregation functions ('sum', 'max', 'std', 'mean', 'count').

    Returns:
        List of Polars Expressions.
    """
    exprs: List[pl.Expr] = []
    for w_name, start_off, end_off in windows:
        w_start = anchor_val - timedelta(days=start_off)
        w_end = anchor_val - timedelta(days=end_off)
        mask = pl.col("event_date").is_between(w_start, w_end)

        for col in value_cols:
            for agg in aggs:
                if agg == "sum":
                    e = pl.when(mask).then(pl.col(col)).otherwise(0.0).sum()
                elif agg == "max":
                    e = pl.when(mask).then(pl.col(col)).otherwise(None).max()
                elif agg == "std":
                    e = pl.when(mask).then(pl.col(col)).otherwise(None).std()
                elif agg == "mean":
                    e = pl.when(mask).then(pl.col(col)).otherwise(None).mean()
                elif agg == "count":
                    e = pl.when(mask).then(1).otherwise(0).sum()
                else:
                    raise ValueError(f"Unknown aggregation method: {agg}")
                exprs.append(e.alias(f"{col}_{agg}_{w_name}"))
    return exprs


def generate_features(
    data: pl.DataFrame,
    anchor_dates: List[date],
    user_ids: Optional[List[int]] = None,
    value_cols: List[str] = DEFAULT_VALUE_COLS,
    windows: List[Tuple[str, int, int]] = DEFAULT_WINDOWS,
    aggs: List[str] = DEFAULT_AGGS,
) -> pl.DataFrame:
    """Generates windowed aggregations for given user IDs and anchor dates.

    Args:
        data: Input event DataFrame.
        anchor_dates: Target anchor dates.
        user_ids: Optional subset of user_ids to calculate features for.
        value_cols: Value columns for aggregation.
        windows: Window definitions.
        aggs: Aggregation operations.

    Returns:
        Polars DataFrame containing [anchor_date, user_id, feature_1, feature_2, ...]
    """
    if user_ids is None:
        user_ids = data["user_id"].unique().sort().to_list()

    max_back = max(w[1] for w in windows)
    min_anchor, max_anchor = min(anchor_dates), max(anchor_dates)

    # Filter data within max potential history boundary
    data_filtered = data.filter(
        pl.col("user_id").is_in(user_ids)
        & pl.col("event_date").is_between(
            min_anchor - timedelta(days=max_back),
            max_anchor,
        )
    )

    parts: List[pl.DataFrame] = []
    for a in anchor_dates:
        ad = data_filtered.filter(
            pl.col("event_date") >= a - timedelta(days=max_back)
        )
        if len(ad) > 0:
            features = (
                ad.group_by("user_id")
                .agg(build_window_agg_exprs(a, windows, value_cols, aggs))
                .with_columns(anchor_date=pl.lit(a))
            )
            parts.append(features)

    features_df = (
        pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    )

    # Full cross product index of anchor_date x user_id
    index_df = pl.DataFrame({"anchor_date": anchor_dates}).join(
        pl.DataFrame({"user_id": user_ids}), how="cross"
    )

    result = index_df.join(
        features_df, on=["anchor_date", "user_id"], how="left"
    )

    feature_cols = [
        c for c in result.columns if c not in ["anchor_date", "user_id"]
    ]
    fill_exprs = [pl.col(c).fill_null(0.0) for c in feature_cols]

    return result.with_columns(fill_exprs)


def generate_targets(
    data: pl.DataFrame,
    anchor_dates: List[date],
    user_ids: Optional[List[int]] = None,
    horizon_days: int = 30,
    target_col: str = "gmv",
) -> pl.DataFrame:
    """Generates future sum target (e.g., LTV over horizon_days) after anchor date.

    Args:
        data: Input event DataFrame.
        anchor_dates: Target anchor dates.
        user_ids: Optional list of user IDs.
        horizon_days: Target evaluation period (days).
        target_col: Column to sum for target (default 'gmv').

    Returns:
        DataFrame containing [anchor_date, user_id, target].
    """
    if user_ids is None:
        user_ids = data["user_id"].unique().sort().to_list()

    index_df = pl.DataFrame({"anchor_date": anchor_dates}).join(
        pl.DataFrame({"user_id": user_ids}), how="cross"
    )

    parts: List[pl.DataFrame] = []
    for a in anchor_dates:
        t_start = a + timedelta(days=1)
        t_end = a + timedelta(days=horizon_days)

        tgt = (
            data.filter(
                pl.col("user_id").is_in(user_ids)
                & pl.col("event_date").is_between(t_start, t_end)
            )
            .group_by("user_id")
            .agg(pl.col(target_col).sum().alias("target"))
            .with_columns(anchor_date=pl.lit(a))
        )
        parts.append(tgt)

    tgt_df = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    targets = index_df.join(tgt_df, on=["anchor_date", "user_id"], how="left")
    return targets.with_columns(pl.col("target").fill_null(0.0))


def process_all_folds(
    data: pl.DataFrame,
    output_dir: Union[str, Path] = FEATURES_DIR,
    n_folds: int = N_FOLDS,
    batch_size: int = BATCH_SIZE,
    horizon_days: int = 30,
) -> None:
    """Batch-processes features and targets for all time-CV folds + test fold.

    Args:
        data: Event DataFrame.
        output_dir: Folder to store Parquet feature folds.
        n_folds: Number of time-CV folds.
        batch_size: User batch size for RAM protection.
        horizon_days: Target horizon days.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anchors_time_folds = generate_cv_anchor_dates(data, n_folds=n_folds)
    anchor_end_of_time = data["event_date"].max()

    user_ids = data["user_id"].unique().sort().to_list()
    n_users = len(user_ids)
    n_batches = (n_users + batch_size - 1) // batch_size

    print(f"Processing {len(anchors_time_folds)} CV folds + 1 Test fold...")
    print(f"Total Users: {n_users:,} -> {n_batches} batches (batch_size={batch_size:,})")

    # 1. Process CV Folds
    for fold_idx, anchor in enumerate(anchors_time_folds):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = output_path / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        for batch in range(n_batches):
            current_batch = user_ids[
                batch * batch_size : (batch + 1) * batch_size
            ]
            features = generate_features(data, [anchor], user_ids=current_batch)
            targets = generate_targets(
                data, [anchor], user_ids=current_batch, horizon_days=horizon_days
            )

            out_df = features.join(
                targets, on=["anchor_date", "user_id"], how="left"
            ).with_columns(pl.col("target").fill_null(0.0))

            out_df.write_parquet(fold_dir / f"batch_{batch:04d}.parquet")

        print(f"  [+] Completed {fold_name} (anchor: {anchor})")

    # 2. Process Final Prediction / Test Fold
    fold_dir = output_path / "fold_end"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for batch in range(n_batches):
        current_batch = user_ids[
            batch * batch_size : (batch + 1) * batch_size
        ]
        feats = generate_features(data, [anchor_end_of_time], user_ids=current_batch)
        out_df = feats.with_columns(pl.lit(None).cast(pl.Float64).alias("target"))
        out_df.write_parquet(fold_dir / f"batch_{batch:04d}.parquet")

    print(f"  [+] Completed fold_end (anchor: {anchor_end_of_time})")


def read_fold(
    output_dir: Union[str, Path] = FEATURES_DIR, fold_name: str = "fold_03"
) -> pl.DataFrame:
    """Fast multi-parquet reader for a specific feature fold.

    Args:
        output_dir: Base features directory.
        fold_name: Subfolder name (e.g., 'fold_00', 'fold_03', 'fold_end').

    Returns:
        Merged Polars DataFrame of all batches in that fold.
    """
    pattern = str(Path(output_dir) / fold_name / "batch_*.parquet")
    return pl.read_parquet(pattern)
