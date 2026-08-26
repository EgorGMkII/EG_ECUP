"""Contrastive Learning on Event Sequences (CoLES) for dense user representation learning."""
from __future__ import annotations

import time
from typing import Optional
import numpy as np
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class EventSequenceDataset(Dataset):
    def __init__(self, event_memmap: np.ndarray, seq_len: int = 64, sub_len: int = 32):
        self.memmap = event_memmap  # shape: (N, max_events, feature_dim)
        self.n_users = event_memmap.shape[0]
        self.max_events = event_memmap.shape[1]
        self.sub_len = min(sub_len, self.max_events // 2)

    def __len__(self) -> int:
        return self.n_users

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.memmap[idx]  # (max_events, d)
        # Find active events (non-zero)
        active_mask = (seq != 0).any(axis=-1)
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) < 4:
            # Padding fallback
            w1 = seq[:self.sub_len]
            w2 = seq[-self.sub_len:]
        else:
            mid = len(active_idx) // 2
            idx1 = active_idx[:mid]
            idx2 = active_idx[mid:]
            
            # Slice or pad to sub_len
            w1 = seq[idx1[-self.sub_len:]] if len(idx1) >= self.sub_len else np.pad(seq[idx1], ((0, self.sub_len - len(idx1)), (0, 0)))
            w2 = seq[idx2[-self.sub_len:]] if len(idx2) >= self.sub_len else np.pad(seq[idx2], ((0, self.sub_len - len(idx2)), (0, 0)))
            
        return torch.from_numpy(w1.astype(np.float32)), torch.from_numpy(w2.astype(np.float32))


class FullEventSequenceDataset(Dataset):
    def __init__(self, event_memmap: np.ndarray, seq_len: int = 64):
        self.memmap = event_memmap
        self.n_users = event_memmap.shape[0]
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n_users

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.memmap[idx, -self.seq_len:]
        return torch.from_numpy(seq.astype(np.float32))


class CoLESEncoder(nn.Module):
    def __init__(self, in_features: int = 15, hidden_dim: int = 64, out_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.input_proj(x)
        out, _ = self.gru(h)
        # Global max + mean pooling over time
        pooled = out.mean(dim=1) + out.max(dim=1)[0]
        z = self.head(pooled)
        return F.normalize(z, p=2, dim=-1)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    # z1, z2: (B, D)
    batch_size = z1.size(0)
    # Cosine similarities
    sim_matrix = torch.matmul(z1, z2.T) / temperature  # (B, B)
    labels = torch.arange(batch_size, device=z1.device)
    loss1 = F.cross_entropy(sim_matrix, labels)
    loss2 = F.cross_entropy(sim_matrix.T, labels)
    return 0.5 * (loss1 + loss2)


def train_coles_embeddings(
    train_memmap: np.ndarray | tuple,
    valid_memmap: np.ndarray | tuple,
    user_ids: list[int],
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 512,
    lr: float = 0.001,
    out_dim: int = 32
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train self-supervised CoLES encoder and extract dense 32-dim user representations."""
    if isinstance(train_memmap, tuple):
        train_memmap = train_memmap[0]
    if isinstance(valid_memmap, tuple):
        valid_memmap = valid_memmap[0]
    in_features = train_memmap.shape[-1]
    dataset = EventSequenceDataset(train_memmap)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    model = CoLESEncoder(in_features=in_features, hidden_dim=64, out_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        for w1, w2 in loader:
            w1 = w1.to(device)
            w2 = w2.to(device)
            optimizer.zero_grad()
            z1 = model(w1)
            z2 = model(w2)
            loss = info_nce_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
    # Extract representations for train and valid
    model.eval()
    
    def extract(memmap: np.ndarray) -> pl.DataFrame:
        eval_ds = FullEventSequenceDataset(memmap)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        embeddings = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                z = model(batch).cpu().numpy()
                embeddings.append(z)
        emb_matrix = np.concatenate(embeddings, axis=0)
        cols = {f"coles_{i}": emb_matrix[:, i].astype(np.float32) for i in range(out_dim)}
        cols["user_id"] = np.array(user_ids, dtype=np.int64)
        return pl.DataFrame(cols)

    train_emb = extract(train_memmap)
    valid_emb = extract(valid_memmap)
    return train_emb, valid_emb
