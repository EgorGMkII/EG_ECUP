"""Leakage-safe sparse snapshots for the unconditional direct CatBoost branch."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from src.ssl_temporal_stack_v1.runtime import derive_seed, progress


WINDOWS = (7, 14, 30, 60, 90, 180, 365)
SUM_METRICS = ("gmv", "gmv_search", "gmv_cat", "searches", "to_ord", "to_cart")
TARGET_COLUMNS = frozenset({"target", "z_target", "future_gmv_30d", "will_buy", "will_buy_30d"})


def build_direct_snapshot(raw: pl.DataFrame, users: Sequence[int], anchor: str, *, with_target: bool) -> pl.DataFrame:
    """Build one sparse user snapshot using only events at or before ``anchor``."""
    anchor_date = date.fromisoformat(anchor)
    history_start = anchor_date - timedelta(days=max(WINDOWS) - 1)
    index = pl.DataFrame({"user_id": list(users)})
    hist = raw.filter(
        pl.col("user_id").is_in(users)
        & pl.col("event_date").is_between(history_start, anchor_date)
    )
    active = (pl.col("searches") > 0) | (pl.col("to_cart") > 0) | (pl.col("to_ord") > 0) | (pl.col("gmv") > 0)
    expressions: list[pl.Expr] = []
    for days in WINDOWS:
        inside = pl.col("event_date") >= anchor_date - timedelta(days=days - 1)
        for metric in SUM_METRICS:
            expressions.append(pl.when(inside).then(pl.col(metric)).otherwise(0).sum().alias(f"direct_{metric}_sum_{days}d"))
        expressions.extend(
            [
                pl.when(inside & active).then(1).otherwise(0).sum().alias(f"direct_active_days_{days}d"),
                pl.when(inside & (pl.col("to_ord") > 0)).then(1).otherwise(0).sum().alias(f"direct_order_days_{days}d"),
                pl.when(inside & (pl.col("to_cart") > 0)).then(1).otherwise(0).sum().alias(f"direct_cart_days_{days}d"),
                pl.when(inside & (pl.col("searches") > 0)).then(1).otherwise(0).sum().alias(f"direct_search_days_{days}d"),
            ]
        )
    expressions.extend(
        [
            (pl.lit(anchor_date) - pl.col("event_date").min()).dt.total_days().alias("direct_history_age_days"),
            active.cast(pl.Int16).sum().alias("direct_lifetime_active_days"),
            (pl.lit(anchor_date) - pl.col("event_date").max()).dt.total_days().alias("direct_days_since_activity"),
            (pl.lit(anchor_date) - pl.when(pl.col("to_ord") > 0).then(pl.col("event_date")).max()).dt.total_days().alias("direct_days_since_order"),
            (pl.lit(anchor_date) - pl.when(pl.col("gmv") > 0).then(pl.col("event_date")).max()).dt.total_days().alias("direct_days_since_positive_gmv"),
        ]
    )
    snapshot = index.join(hist.group_by("user_id").agg(expressions), on="user_id", how="left")
    recency = {"direct_history_age_days", "direct_days_since_activity", "direct_days_since_order", "direct_days_since_positive_gmv"}
    snapshot = snapshot.with_columns(
        [pl.col(name).fill_null(366.0 if name in recency else 0.0).cast(pl.Float32) for name in snapshot.columns if name != "user_id"]
    )
    if with_target:
        target = (
            raw.filter(
                pl.col("user_id").is_in(users)
                & pl.col("event_date").is_between(anchor_date + timedelta(days=1), anchor_date + timedelta(days=30))
            )
            .group_by("user_id")
            .agg(pl.col("gmv").sum().alias("future_gmv_30d"))
        )
        snapshot = snapshot.join(target, on="user_id", how="left").with_columns(
            pl.col("future_gmv_30d").fill_null(0.0),
            pl.col("future_gmv_30d").fill_null(0.0).log1p().alias("z_target"),
        )
    if snapshot.height != len(users) or snapshot["user_id"].to_list() != list(users):
        raise RuntimeError("Direct CatBoost snapshot lost cohort order")
    return snapshot


def fit_predict_direct_catboost(context: Any, values: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the direct branch on the latest causal training snapshot and predict holdout z."""
    from catboost import CatBoostRegressor, Pool, __version__ as catboost_version

    if context.raw_events is None:
        raise RuntimeError("Direct CatBoost requires raw causal events")
    policy = str(values.get("training_anchors", "latest"))
    if policy != "latest":
        raise ValueError("catboost_direct.training_anchors currently supports only 'latest'")
    train_anchor = context.train_anchors[-1]
    started = time.perf_counter()
    progress("DIRECT_CATBOOST_SNAPSHOT_START", run=context.run_name, train_anchor=train_anchor, holdout_anchor=context.holdout_anchor)
    train = build_direct_snapshot(context.raw_events, context.users, train_anchor, with_target=True)
    holdout = build_direct_snapshot(context.raw_events, context.users, context.holdout_anchor, with_target=False)
    features = tuple(name for name in train.columns if name != "user_id" and name not in TARGET_COLUMNS)
    if set(features) & TARGET_COLUMNS:
        raise RuntimeError("Target leakage in direct CatBoost features")
    params = {
        "iterations": int(values["iterations"]),
        "depth": int(values["depth"]),
        "learning_rate": float(values["learning_rate"]),
        "l2_leaf_reg": float(values["l2_leaf_reg"]),
        "loss_function": "RMSE",
        "random_seed": derive_seed(context.root_seed, context.run_name, "catboost_direct"),
        "task_type": str(values["task_type"]),
        "devices": str(values["devices"]),
        "verbose": 50,
        "allow_writing_files": False,
    }
    progress("DIRECT_CATBOOST_TRAIN_START", run=context.run_name, rows=train.height, features=len(features), params=params)
    model = CatBoostRegressor(**params)
    model.fit(Pool(train.select(features).to_pandas(), label=train["z_target"].to_numpy()))
    prediction = np.clip(model.predict(Pool(holdout.select(features).to_pandas())), 0.0, None).astype(np.float64)
    report = {
        "catboost_version": catboost_version,
        "train_anchor": train_anchor,
        "holdout_anchor": context.holdout_anchor,
        "training_anchor_policy": policy,
        "rows": train.height,
        "feature_count": len(features),
        "feature_names": list(features),
        "history_days": max(WINDOWS),
        "windows": list(WINDOWS),
        "target": "log1p(future_gmv_30d)",
        "resolved_parameters": model.get_all_params(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    progress("DIRECT_CATBOOST_DONE", run=context.run_name, **{key: report[key] for key in ("rows", "feature_count", "elapsed_seconds")})
    return prediction, report
