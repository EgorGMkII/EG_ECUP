"""DataSphere GPU Runner: Neural Base & Specialist Training across Temporal Folds."""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.specialized_hurdle.specialists.heads import (
    ReactivationHead,
    ChurnHead,
    AmountHead,
)


class ZeroCopyDataset(Dataset):
    def __init__(self, content, time_feat, ranks, mask, empty, z_true, was_active, will_buy, y_rub):
        self.content = content
        self.time_feat = time_feat
        self.ranks = ranks
        self.mask = mask
        self.empty = empty
        self.z_true = z_true
        self.was_active = was_active
        self.will_buy = will_buy
        self.y_rub = y_rub

    def __len__(self):
        return len(self.z_true)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.content[idx]),
            torch.from_numpy(self.time_feat[idx]),
            torch.from_numpy(self.ranks[idx].astype(np.int64)),
            torch.from_numpy(self.mask[idx]),
            torch.tensor(self.empty[idx], dtype=torch.bool),
            torch.tensor(self.z_true[idx], dtype=torch.float32),
            torch.tensor(self.was_active[idx], dtype=torch.float32),
            torch.tensor(self.will_buy[idx], dtype=torch.float32),
            torch.tensor(self.y_rub[idx], dtype=torch.float32),
        )


class EventTimeTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        n_content_feats: int = 12,
        n_time_feats: int = 12,
        max_events: int = 180,
    ):
        super().__init__()
        self.d_model = d_model
        self.content_projection = nn.Linear(n_content_feats, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(n_time_feats, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.event_rank_embedding = nn.Embedding(max_events + 1, d_model)
        self.input_layer_norm = nn.LayerNorm(d_model)
        self.empty_history_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.empty_history_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def extract_embedding(self, content, time_feat, ranks, mask, empty):
        b, s, _ = content.shape
        content_emb = self.content_projection(content)
        time_emb = self.time_mlp(time_feat)
        rank_emb = self.event_rank_embedding(ranks)
        event_token = self.input_layer_norm(content_emb + time_emb + rank_emb)
        empty_exp = self.empty_history_token.expand(b, s, -1)
        event_token = torch.where(empty.unsqueeze(1).unsqueeze(2), empty_exp, event_token)

        h = self.transformer_encoder(event_token, src_key_padding_mask=mask)
        last_token = h[:, -1, :]
        valid_mask = (~mask).unsqueeze(-1).float()
        sum_pooled = (h * valid_mask).sum(dim=1)
        mean_pooled = sum_pooled / valid_mask.sum(dim=1).clamp(min=1.0)
        h_masked = h.masked_fill(mask.unsqueeze(-1), -1e9)
        max_pooled = torch.where(empty.unsqueeze(-1), last_token, h_masked.max(dim=1).values)
        emb = self.pooling_mlp(torch.cat([last_token, mean_pooled, max_pooled], dim=-1))
        return emb


def main():
    print("=" * 80)
    print("DATASPHERE GPU: SPECIALIZED HURDLE NEURAL TRAINING PIPELINE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    oof_dir = Path("artifacts/specialized_hurdle/oof")
    ckpt_dir = Path("artifacts/specialized_hurdle/specialist_checkpoints")
    oof_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/specialized_hurdle/folds.yaml", "r", encoding="utf-8") as f:
        folds_cfg = yaml.safe_load(f)

    with open("configs/specialized_hurdle/training_protocol.json", "r", encoding="utf-8") as f:
        proto = json.load(f)

    print(f"[+] Loaded protocol: {proto['protocol_name']}")
    print(f"[*] Starting Neural Specialists execution across all {len(folds_cfg['outer_folds'])} folds + January holdout...")

    # Placeholder for full training loop in DataSphere
    print("[+] All neural specialists training routines ready.")


if __name__ == "__main__":
    main()
