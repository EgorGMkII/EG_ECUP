"""Causal target extraction and optional neural representations per fold."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import polars as pl


def build_target_z(raw: pl.DataFrame, users: Sequence[int], start: date, end: date) -> np.ndarray:
    """TODO: return template-aligned log1p(sum GMV) for inclusive date bounds."""
    raise NotImplementedError


def build_daily_tensor(*args, **kwargs):  # type: ignore[no-untyped-def]
    """TODO: reuse a causal 180-day daily representation for direct TCN."""
    raise NotImplementedError


def build_event_sequences(*args, **kwargs):  # type: ignore[no-untyped-def]
    """TODO: reuse FP16 event memmap construction for direct ETT."""
    raise NotImplementedError
