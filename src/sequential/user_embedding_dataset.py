"""Dataset and DataLoader utilities for User-Embedding GRU experiments."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.sequential.preprocessing import SequentialScaler


class UserMemmapDataset(Dataset):
    """Memory-mapped sequence dataset returning (x_seq, y_log, past_b, fut_b, user_idx)."""

    def __init__(
        self,
        tensor_paths: List[Path],
        targets_list: List[np.ndarray],
        past_buyer_list: List[np.ndarray],
        user_idx_list: List[np.ndarray],
        seq_len: int = 180,
        scaler: Optional[SequentialScaler] = None,
        shuffle_user_idx: bool = False,
        seed: int = 42,
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

        # Handle user_idx arrays across anchors
        processed_u_idx = []
        rng = np.random.default_rng(seed)
        for u_arr in user_idx_list:
            if shuffle_user_idx:
                # Independently permute user_idx within each anchor
                permuted = u_arr.copy()
                rng.shuffle(permuted)
                processed_u_idx.append(permuted)
            else:
                processed_u_idx.append(u_arr)

        u_idx_all = np.concatenate(processed_u_idx).astype(np.int64)
        self.user_idx = torch.from_numpy(u_idx_all).long()

    def __len__(self):
        return self.cum_len[-1]

    def __getitem__(self, idx: int):
        t_idx = int(np.searchsorted(self.cum_len, idx, side="right") - 1)
        l_idx = int(idx - self.cum_len[t_idx])

        raw_full = self.mmaps[t_idx][l_idx]  # (max_len, channels)
        raw_sliced = raw_full[-self.seq_len :, :]

        if self.scaler is not None:
            mean = self.scaler.mean[: raw_sliced.shape[-1]]
            std = self.scaler.std[: raw_sliced.shape[-1]]
            sc_seq = (raw_sliced - mean) / std
        else:
            sc_seq = raw_sliced

        x_tensor = torch.from_numpy(sc_seq.astype(np.float32)).float()
        return x_tensor, self.y_log[idx], self.past_buyer[idx], self.fut_buyer[idx], self.user_idx[idx]
