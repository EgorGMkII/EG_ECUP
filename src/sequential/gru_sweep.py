"""Dataset handling, sequence slicing, and anchor set management for MultiTask GRU Sweep."""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET

RAW_DATA_START = date(2025, 1, 1)


def generate_full_calendar_anchors(
    data_start: date = date(2025, 1, 1),
    data_end: date = date(2026, 2, 13),
    step_days: int = 14,
    target_days: int = 30,
) -> List[date]:
    """Generates anchors starting from 2025-01-31, explicitly including early spring 2025-02-13."""
    earliest_anchor = date(2025, 1, 31)  # minimum 30 days of 2025 history
    max_anchor = data_end - timedelta(days=target_days)  # 2026-01-14

    anchors = []
    current = earliest_anchor
    while current <= max_anchor:
        anchors.append(current)
        current += timedelta(days=step_days)

    # Explicitly include early spring anchor mirroring test period
    spring_anchor_2025 = date(2025, 2, 13)
    if spring_anchor_2025 not in anchors and spring_anchor_2025 <= max_anchor:
        anchors.append(spring_anchor_2025)

    if max_anchor not in anchors:
        anchors.append(max_anchor)

    return sorted(list(set(anchors)))


def get_anchor_set(name: str) -> List[date]:
    """Returns exact list of anchor dates for a named set."""
    if name == "recent_14":
        all_22 = generate_panel_anchors()
        return all_22[-14:]
    elif name == "all_existing_22":
        return generate_panel_anchors()
    elif name == "full_calendar":
        return generate_full_calendar_anchors()
    else:
        raise ValueError(f"Unknown anchor set: {name}")


def extract_or_load_master_sequence_tensor(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    max_seq_len: int = 180,
    include_is_observed: bool = False,
) -> np.ndarray:
    """Loads or generates master 180d sequence tensor with optional is_observed channel."""
    obs_suffix = "_obs" if include_is_observed else ""
    filename = f"seq_tensor_{anchor_date.strftime('%Y-%m-%d')}_u{len(user_ids)}_t{max_seq_len}{obs_suffix}.npy"
    file_path = CACHE_DIR / filename

    if file_path.exists():
        return np.load(file_path, mmap_mode="r")

    print(f"[*] Generating master sequence tensor ({max_seq_len}d, obs={include_is_observed}) for anchor {anchor_date}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Standard 15-channel tensor
    raw_15 = get_cached_sequence_tensor(data, user_ids, anchor_date, seq_len=max_seq_len)

    if not include_is_observed:
        return raw_15

    # Add 16th channel: is_observed
    # Real history exists from 2025-01-01 to anchor_date
    history_start = anchor_date - timedelta(days=max_seq_len - 1)
    n_users = len(user_ids)
    arr_16 = np.zeros((n_users, max_seq_len, 16), dtype=np.float32)
    arr_16[:, :, :15] = raw_15

    # Day-by-day observation flag
    for day_idx in range(max_seq_len):
        cur_d = history_start + timedelta(days=day_idx)
        if cur_d >= RAW_DATA_START:
            arr_16[:, day_idx, 15] = 1.0
        else:
            # Pre-2025 padding: zero out all features and sin/cos
            arr_16[:, day_idx, :] = 0.0

    np.save(file_path, arr_16)
    return np.load(file_path, mmap_mode="r")


class SlicedMemmapDataset(Dataset):
    """Dynamic time-slice dataset streaming directly from master memory-mapped tensors."""

    def __init__(
        self,
        tensor_paths: List[Path],
        targets_list: List[np.ndarray],
        past_buyer_list: List[np.ndarray],
        seq_len: int = 90,
        scaler: Optional[SequentialScaler] = None,
    ):
        self.mmaps = [np.load(p, mmap_mode="r") for p in tensor_paths]
        self.lengths = [len(m) for m in self.mmaps]
        self.cum_len = np.cumsum([0] + self.lengths)
        self.seq_len = seq_len
        self.scaler = scaler

        y_all = np.concatenate(targets_list).astype(np.float32)
        past_b_all = np.concatenate(past_buyer_list).astype(np.float32)

        self.y_true = torch.from_numpy(y_all).float()
        self.y_log = torch.log1p(torch.clamp(self.y_true, min=0.0))
        self.past_buyer = torch.from_numpy(past_b_all).float()
        self.fut_buyer = (self.y_true > 0.0).float()

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, idx: int):
        t_idx = np.searchsorted(self.cum_len, idx, side="right") - 1
        l_idx = idx - self.cum_len[t_idx]

        # Slice the last seq_len days directly from master tensor
        raw_full = self.mmaps[t_idx][l_idx]  # (max_len, channels)
        raw_sliced = raw_full[-self.seq_len :, :]

        if self.scaler is not None:
            mean = self.scaler.mean[: raw_sliced.shape[-1]]
            std = self.scaler.std[: raw_sliced.shape[-1]]
            sc_seq = (raw_sliced - mean) / std
        else:
            sc_seq = raw_sliced

        x_tensor = torch.from_numpy(sc_seq.astype(np.float32)).float()
        return x_tensor, self.y_log[idx], self.past_buyer[idx], self.fut_buyer[idx]
