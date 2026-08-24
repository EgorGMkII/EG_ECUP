"""Causal target extraction and optional neural representations per fold."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import polars as pl


def build_target_z(raw: pl.DataFrame, users: Sequence[int], start: date, end: date) -> np.ndarray:
    """Return template-aligned ``log1p`` GMV for an inclusive date interval.

    The function intentionally starts from the supplied user sequence and left
    joins the sparse aggregation.  Consequently users with no observed rows
    receive a zero target and are never silently dropped.  ``start`` and
    ``end`` are both included; callers are responsible for temporal contract
    validation.
    """
    if end < start:
        raise ValueError(f"Invalid target interval: {start} > {end}")
    if "event_date" not in raw.columns or "user_id" not in raw.columns or "gmv" not in raw.columns:
        raise ValueError("raw data must contain event_date, user_id and gmv")
    user_frame = pl.DataFrame({"user_id": list(users)})
    if user_frame.height != len(users) or user_frame["user_id"].n_unique() != len(users):
        raise ValueError("users must be unique and template ordered")
    dates = raw.get_column("event_date")
    if dates.dtype == pl.String:
        raw_dates = raw.with_columns(pl.col("event_date").str.to_date())
    elif dates.dtype == pl.Datetime:
        raw_dates = raw.with_columns(pl.col("event_date").dt.date())
    else:
        raw_dates = raw
    agg = (
        raw_dates.filter(
            pl.col("user_id").is_in(list(users))
            & pl.col("event_date").is_between(start, end, closed="both")
        )
        .group_by("user_id")
        .agg(pl.col("gmv").fill_null(0.0).sum().alias("_target_gmv"))
    )
    aligned = user_frame.join(agg, on="user_id", how="left").with_columns(
        pl.col("_target_gmv").fill_null(0.0).cast(pl.Float64)
    )
    values = aligned["_target_gmv"].to_numpy().astype(np.float64, copy=False)
    return np.log1p(np.maximum(values, 0.0))


def build_daily_tensor(*args, **kwargs):  # type: ignore[no-untyped-def]
    """TODO: reuse a causal 180-day daily representation for direct TCN."""
    raise NotImplementedError


def build_event_sequences(*args, **kwargs):  # type: ignore[no-untyped-def]
    """TODO: reuse FP16 event memmap construction for direct ETT."""
    raise NotImplementedError
