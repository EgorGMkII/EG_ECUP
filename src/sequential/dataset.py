"""Dataset and Tensor Builder for 3D Sequential Daily Logs."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.sequential.preprocessing import CHANNELS, NUMERIC_CHANNELS, SequentialScaler
from src.snapshots import TRAIN_PARQUET


CACHE_DIR = Path("data/sequential_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_user_sequence_tensor(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    seq_len: int = 90,
    cache_dir: Path = CACHE_DIR,
) -> np.ndarray:

    """Builds a dense [N_users, seq_len, n_channels] float32 tensor for given users and anchor."""
    start_date = anchor_date - timedelta(days=seq_len - 1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"seq_tensor_{anchor_date.strftime('%Y-%m-%d')}_u{len(user_ids)}_t{seq_len}.npy"
    cache_path = cache_dir / filename
    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    # 1. Filter raw log strictly within [start_date, anchor_date] for given user_ids
    user_set = set(user_ids)
    hist_df = (
        data.filter(
            (pl.col("event_date") >= start_date)
            & (pl.col("event_date") <= anchor_date)
            & (pl.col("user_id").is_in(user_set))
        )
        .sort(["user_id", "event_date"])
    )

    n_users = len(user_ids)
    n_channels = len(CHANNELS)
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # Direct-to-disk allocation (Zero RAM overhead)
    tensor = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype="float32",
        shape=(n_users, seq_len, n_channels),
    )

    # Fill calendar/position channels for all days across all users
    date_grid = [start_date + timedelta(days=d) for d in range(seq_len)]
    for t, d in enumerate(date_grid):
        w = d.weekday()
        tensor[:, t, CHANNELS.index("sin_dow")] = np.sin(2.0 * np.pi * w / 7.0)
        tensor[:, t, CHANNELS.index("cos_dow")] = np.cos(2.0 * np.pi * w / 7.0)
        tensor[:, t, CHANNELS.index("normalized_position")] = float(t) / float(seq_len - 1)

    if hist_df.height > 0:
        user_col = hist_df["user_id"].to_numpy()
        date_col = hist_df["event_date"].to_list()

        num_vals = {}
        for ch in NUMERIC_CHANNELS:
            if ch in hist_df.columns:
                num_vals[ch] = np.log1p(np.maximum(hist_df[ch].to_numpy().astype(np.float32), 0.0))
            else:
                num_vals[ch] = np.zeros(hist_df.height, dtype=np.float32)

        gmv_arr = hist_df["gmv"].to_numpy() if "gmv" in hist_df.columns else np.zeros(hist_df.height)
        ord_arr = hist_df["to_ord"].to_numpy() if "to_ord" in hist_df.columns else np.zeros(hist_df.height)
        is_purchase = ((gmv_arr > 0) | (ord_arr > 0)).astype(np.float32)

        idx_active = CHANNELS.index("is_active")
        idx_purch = CHANNELS.index("is_purchase_day")

        for i in range(len(user_col)):
            u = user_col[i]
            d = date_col[i]
            u_idx = user_to_idx.get(u)
            if u_idx is None:
                continue
            t_idx = (d - start_date).days
            if 0 <= t_idx < seq_len:
                tensor[u_idx, t_idx, idx_active] = 1.0
                tensor[u_idx, t_idx, idx_purch] = is_purchase[i]
                for ch in NUMERIC_CHANNELS:
                    tensor[u_idx, t_idx, CHANNELS.index(ch)] = num_vals[ch][i]

        del user_col, date_col, num_vals, gmv_arr, ord_arr, is_purchase

    tensor.flush()
    del tensor, hist_df
    import gc
    gc.collect()

    return np.load(cache_path, mmap_mode="r")


def extract_anchor_targets(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    horizon_days: int = 30,
) -> np.ndarray:
    """Extracts ground-truth 30d GMV targets for users in given anchor window."""
    t_start = anchor_date + timedelta(days=1)
    t_end = anchor_date + timedelta(days=horizon_days)

    user_set = set(user_ids)
    target_data = data.filter(
        (pl.col("user_id").is_in(user_set))
        & (pl.col("event_date") >= t_start)
        & (pl.col("event_date") <= t_end)
    )

    agg = target_data.group_by("user_id").agg(pl.col("gmv").sum().alias("target"))
    index_df = pl.DataFrame({"user_id": user_ids})
    merged = index_df.join(agg, on="user_id", how="left").fill_null(0.0)
    return merged["target"].to_numpy().astype(np.float32)


def get_cached_sequence_tensor(
    data: Optional[pl.DataFrame],
    user_ids: List[int],
    anchor_date: date,
    seq_len: int = 90,
    cache_dir: Path = CACHE_DIR,
) -> np.ndarray:
    """Loads cached sequence tensor if exists, otherwise builds, saves, and returns."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"seq_tensor_{anchor_date.strftime('%Y-%m-%d')}_u{len(user_ids)}_t{seq_len}.npy"
    cache_path = cache_dir / filename

    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    if data is None:
        data = pl.read_parquet(TRAIN_PARQUET)

    return build_user_sequence_tensor(data, user_ids, anchor_date, seq_len=seq_len, cache_dir=cache_dir)



class OzonSequenceDataset(Dataset):
    """PyTorch Dataset for (Sequence_Tensor, Target, Is_Buyer, User_Id)."""

    def __init__(
        self,
        tensors: np.ndarray,
        targets: Optional[np.ndarray] = None,
        user_ids: Optional[List[int]] = None,
    ):
        self.tensors = torch.from_numpy(tensors).float() if isinstance(tensors, np.ndarray) else tensors
        self.targets = torch.from_numpy(targets).float() if isinstance(targets, np.ndarray) and targets is not None else targets
        self.user_ids = user_ids

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int):
        x = self.tensors[idx]
        if self.targets is not None:
            y = self.targets[idx]
            y_log = torch.log1p(torch.clamp(y, min=0.0))
            is_buyer = (y > 0.0).float()
            return x, y_log, is_buyer, y
        return x


class MemmapMultiAnchorDataset(Dataset):
    """Zero-RAM PyTorch Dataset reading multi-anchor sequences directly from disk via memmap."""

    def __init__(
        self,
        tensor_paths: List[Path],
        targets_list: List[np.ndarray],
        scaler: Optional[SequentialScaler] = None,
    ):
        self.tensor_paths = tensor_paths
        self.tensors = [np.load(p, mmap_mode="r") for p in tensor_paths]
        self.targets = [torch.from_numpy(t).float() for t in targets_list]
        self.lens = [len(t) for t in self.targets]
        self.cum_lens = np.cumsum(self.lens)
        self.total_len = int(self.cum_lens[-1])
        self.scaler = scaler

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        file_idx = int(np.searchsorted(self.cum_lens, idx, side="right"))
        local_idx = idx if file_idx == 0 else idx - int(self.cum_lens[file_idx - 1])

        raw_np = np.array(self.tensors[file_idx][local_idx], dtype=np.float32)
        if self.scaler is not None and self.scaler.is_fit:
            raw_np = (raw_np - self.scaler.mean) / self.scaler.std

        x = torch.from_numpy(raw_np).float()
        y = self.targets[file_idx][local_idx]
        y_log = torch.log1p(torch.clamp(y, min=0.0))
        is_buyer = (y > 0.0).float()
        return x, y_log, is_buyer, y
