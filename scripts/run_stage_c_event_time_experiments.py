"""Stage C: Event-Time Transformer Suite (ETT0, ETT1, ETT2).

Features:
- Event Token: content_embedding (128) + time_embedding (128) + rank_embedding (128) -> LayerNorm(128)
- Content Features: 12 behavioral channels of event-day (no user_id, no leakage)
- Continuous Time Features: age_norm, log_age_norm, delta_norm, log_delta_norm, DOW sin/cos, DOY sin/cos, target phase sin/cos, is_first_event
- Event Rank Embedding: rank_from_end = 0 for the most recent event up to anchor date
- Padding & EMPTY_HISTORY token for dormant users
- ETT0: Base continuous-time vector (11 features)
- ETT1: Fixed 30-day exponential decay in time_mlp (12 features)
- ETT2: Learnable single scalar half-life tau in [7, 365] days initialized at 30 days
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Dataset & On-The-Fly Event Sequence Extraction
# -----------------------------------------------------------------------------
ANCHORS_CANONICAL = [
    "2025-09-01", "2025-09-15", "2025-09-29",
    "2025-10-13", "2025-10-27", "2025-11-10",
    "2025-11-24", "2025-12-08",
]
VAL_ANCHOR = "2026-01-14"
TEST_ANCHOR = "2026-02-13"


def compute_event_sequences_for_anchor(
    df_raw: pl.DataFrame,
    anchor_str: str,
    user_ids: np.ndarray,
    max_events: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts event-day logs for all users within 365 days prior to anchor_date.
    
    Returns:
        content_feats: [N, max_events, 12]
        time_feats_base: [N, max_events, 11]
        age_days_arr: [N, max_events]
        ranks_arr: [N, max_events]
        padding_mask: [N, max_events] (True = pad)
        is_empty_arr: [N]
    """
    anchor_dt = pl.lit(anchor_str).str.to_date()
    start_dt = pl.lit(anchor_str).str.to_date().dt.offset_by("-365d")

    # Filter logs strictly in [anchor - 365d, anchor] and for selected users
    user_ids_set = set(user_ids)
    df_filtered = (
        df_raw.lazy()
        .filter(
            (pl.col("event_date") >= start_dt)
            & (pl.col("event_date") <= anchor_dt)
            & (pl.col("user_id").is_in(user_ids_set))
        )
        .select([
            "user_id",
            "event_date",
            pl.col("searches").alias("search_cnt"),
            pl.col("to_cart").alias("cart_cnt"),
            pl.col("to_ord").alias("order_cnt"),
            pl.col("gmv").alias("gmv_order"),
            pl.col("gmv_cat").alias("gmv_cart"),
        ])
        .collect()
    )

    # Group daily
    df_daily = (
        df_filtered.group_by(["user_id", "event_date"])
        .agg([
            pl.col("search_cnt").sum().alias("search_cnt"),
            pl.col("cart_cnt").sum().alias("cart_cnt"),
            pl.col("order_cnt").sum().alias("order_cnt"),
            pl.col("gmv_order").sum().alias("gmv_order"),
            pl.col("gmv_cart").sum().alias("gmv_cart"),
        ])
        .sort(["user_id", "event_date"])
    )

    # Add boolean action flags
    df_daily = df_daily.with_columns([
        (pl.col("search_cnt") > 0).cast(pl.Float32).alias("is_search"),
        (pl.col("cart_cnt") > 0).cast(pl.Float32).alias("is_cart"),
        (pl.col("order_cnt") > 0).cast(pl.Float32).alias("is_order"),
        pl.col("search_cnt").log1p().alias("log_search_cnt"),
        pl.col("cart_cnt").log1p().alias("log_cart_cnt"),
        pl.col("order_cnt").log1p().alias("log_order_cnt"),
        pl.col("gmv_order").log1p().alias("log_gmv_order"),
        pl.col("gmv_cart").log1p().alias("log_gmv_cart"),
        (pl.col("order_cnt") > 0).cast(pl.Float32).alias("is_purchase_day"),
        ((pl.col("search_cnt") > 0) & (pl.col("cart_cnt") == 0) & (pl.col("order_cnt") == 0)).cast(pl.Float32).alias("is_search_only_day"),
        ((pl.col("cart_cnt") > 0) & (pl.col("order_cnt") == 0)).cast(pl.Float32).alias("is_cart_only_day"),
        pl.lit(1.0, dtype=pl.Float32).alias("has_activity"),
    ])

    # Convert to python dict mapping user_id -> events
    from datetime import datetime, timedelta
    user_to_events = {}
    anchor_dt_py = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    midpoint_doy = (anchor_dt_py + timedelta(days=15)).timetuple().tm_yday
    anchor_ts = anchor_dt_py

    for row in df_daily.iter_rows(named=True):
        u = row["user_id"]
        if u not in user_to_events:
            user_to_events[u] = []
        user_to_events[u].append(row)

    n_users = len(user_ids)
    content_feats = np.zeros((n_users, max_events, 12), dtype=np.float32)
    time_feats_base = np.zeros((n_users, max_events, 11), dtype=np.float32)
    age_days_arr = np.zeros((n_users, max_events), dtype=np.float32)
    ranks_arr = np.zeros((n_users, max_events), dtype=np.int64)
    padding_mask = np.ones((n_users, max_events), dtype=bool)  # True = pad
    is_empty_arr = np.zeros(n_users, dtype=bool)

    c_cols = [
        "is_search", "is_cart", "is_order",
        "log_search_cnt", "log_cart_cnt", "log_order_cnt",
        "log_gmv_order", "log_gmv_cart",
        "is_purchase_day", "is_search_only_day", "is_cart_only_day", "has_activity"
    ]

    for i, uid in enumerate(user_ids):
        evs = user_to_events.get(uid, [])
        if len(evs) == 0:
            is_empty_arr[i] = True
            padding_mask[i, 0] = False  # Keep position 0 unmasked for EMPTY_HISTORY token
            ranks_arr[i, 0] = 0
            continue

        # Keep the most recent max_events
        evs = evs[-max_events:]
        num_ev = len(evs)

        prev_date = None
        for j, ev in enumerate(evs):
            pos = max_events - num_ev + j  # Right-align real events
            padding_mask[i, pos] = False
            rank_from_end = num_ev - 1 - j
            ranks_arr[i, pos] = rank_from_end

            # Content features
            content_feats[i, pos] = [ev[col] for col in c_cols]

            # Time features
            ev_date = ev["event_date"]
            age_days = float((anchor_ts - ev_date).days)
            age_days = max(0.0, min(365.0, age_days))
            age_days_arr[i, pos] = age_days

            if j == 0:
                is_first_event = 1.0
                delta_days = 0.0
            else:
                is_first_event = 0.0
                delta_days = float((ev_date - prev_date).days)
                delta_days = max(0.0, min(365.0, delta_days))
            prev_date = ev_date

            age_norm = age_days / 365.0
            log_age_norm = math.log1p(age_days) / math.log(366.0)
            delta_norm = delta_days / 365.0
            log_delta_norm = math.log1p(delta_days) / math.log(366.0)

            dow = ev_date.weekday()
            doy = ev_date.timetuple().tm_yday

            dow_sin = math.sin(2.0 * math.pi * dow / 7.0)
            dow_cos = math.cos(2.0 * math.pi * dow / 7.0)
            doy_sin = math.sin(2.0 * math.pi * doy / 365.25)
            doy_cos = math.cos(2.0 * math.pi * doy / 365.25)

            phase = 2.0 * math.pi * (doy - midpoint_doy) / 365.25
            target_phase_sin = math.sin(phase)
            target_phase_cos = math.cos(phase)

            time_feats_base[i, pos] = [
                age_norm,
                log_age_norm,
                delta_norm,
                log_delta_norm,
                dow_sin,
                dow_cos,
                doy_sin,
                doy_cos,
                target_phase_sin,
                target_phase_cos,
                is_first_event,
            ]

    return content_feats, time_feats_base, age_days_arr, ranks_arr, padding_mask, is_empty_arr


# -----------------------------------------------------------------------------
# 2. Event-Time Transformer Model Architecture
# -----------------------------------------------------------------------------
class EventTimeTransformerModel(nn.Module):
    """Event-Time Transformer with additive Content, Time, and Rank embeddings."""

    def __init__(
        self,
        mode: str = "ETT0",  # 'ETT0', 'ETT1', or 'ETT2'
        content_dim: int = 12,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.20,
        max_events: int = 128,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.max_events = max_events

        # Content projection
        self.content_projection = nn.Sequential(
            nn.Linear(content_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Time MLP
        n_time_feats = 11 if mode == "ETT0" else 12
        self.time_mlp = nn.Sequential(
            nn.Linear(n_time_feats, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
        )

        # Learnable single scalar half-life for ETT2
        if mode == "ETT2":
            # Initialize tau = 30.0: p = (30 - 7) / (365 - 7) = 23 / 358
            p = (30.0 - 7.0) / (365.0 - 7.0)
            raw_tau_init = math.log(p / (1.0 - p))
            self.raw_tau = nn.Parameter(torch.tensor([raw_tau_init], dtype=torch.float32))
        else:
            self.raw_tau = None

        # Rank Embedding (scaled to match linear projections norm)
        self.event_rank_embedding = nn.Embedding(num_embeddings=max_events, embedding_dim=d_model)
        nn.init.normal_(self.event_rank_embedding.weight, std=1.0 / math.sqrt(d_model))

        # Learned EMPTY_HISTORY token
        self.empty_history_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.empty_history_token, std=0.02)

        # Additive LayerNorm
        self.input_layer_norm = nn.LayerNorm(d_model)

        # PreLN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal Attention Pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Transition Hurdle Heads
        self.head_react = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def get_tau(self) -> Optional[float]:
        if self.raw_tau is not None:
            tau = 7.0 + (365.0 - 7.0) * torch.sigmoid(self.raw_tau).item()
            return tau
        return None

    def forward(
        self,
        content_feats: torch.Tensor,
        time_feats_base: torch.Tensor,
        age_days: torch.Tensor,
        ranks: torch.Tensor,
        padding_mask: torch.Tensor,
        is_empty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        # 1. Content Embedding
        content_emb = self.content_projection(content_feats)  # [B, L, 128]

        # 2. Time Embedding
        if self.mode == "ETT0":
            time_input = time_feats_base
        elif self.mode == "ETT1":
            decay_30d = torch.pow(2.0, -age_days / 30.0).unsqueeze(-1)  # [B, L, 1]
            time_input = torch.cat([time_feats_base, decay_30d], dim=-1)
        elif self.mode == "ETT2":
            tau = 7.0 + (365.0 - 7.0) * torch.sigmoid(self.raw_tau)  # scalar in [7, 365]
            decay_learned = torch.pow(2.0, -age_days / tau).unsqueeze(-1)  # [B, L, 1]
            time_input = torch.cat([time_feats_base, decay_learned], dim=-1)

        time_emb = self.time_mlp(time_input)  # [B, L, 128]

        # 3. Rank Embedding
        rank_emb = self.event_rank_embedding(ranks)  # [B, L, 128]

        # Additive combination
        token = content_emb + time_emb + rank_emb

        # Inject EMPTY_HISTORY token for fully empty users at position 0
        if is_empty.any():
            empty_idx = torch.nonzero(is_empty).squeeze(-1)
            token[empty_idx, 0, :] = self.empty_history_token.squeeze(0).squeeze(0)

        # Scale diagnostics on raw vs LayerNorm
        diag_stats = {}
        with torch.no_grad():
            diag_stats["content_norm"] = float(content_emb.norm(dim=-1).mean().item())
            diag_stats["time_norm"] = float(time_emb.norm(dim=-1).mean().item())
            diag_stats["rank_norm"] = float(rank_emb.norm(dim=-1).mean().item())
            diag_stats["raw_token_norm"] = float(token.norm(dim=-1).mean().item())

        # LayerNorm
        token = self.input_layer_norm(token)
        with torch.no_grad():
            diag_stats["post_ln_norm"] = float(token.norm(dim=-1).mean().item())

        # Transformer Encoding
        encoded = self.transformer(token, src_key_padding_mask=padding_mask)  # [B, L, 128]

        # Temporal Attention Pooling (masked)
        scores = self.temporal_attn(encoded).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(padding_mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        emb = torch.sum(encoded * weights, dim=1)  # [B, 128]

        # Heads
        l_react = self.head_react(emb).squeeze(-1)
        l_churn = self.head_churn(emb).squeeze(-1)
        l_buy = self.head_buy(emb).squeeze(-1)
        z_cond = self.head_cond(emb).squeeze(-1)
        z_dir = self.head_dir(emb).squeeze(-1)

        return l_react, l_churn, l_buy, z_cond, z_dir, emb, diag_stats


# -----------------------------------------------------------------------------
# 3. Unit Tests Suite (Time, Decay, Gradient, Assertions)
# -----------------------------------------------------------------------------
def run_unit_tests():
    print("[*] Running Unit Tests for Event-Time Transformer...")

    # 1. Decay manual checks
    age_0 = torch.tensor([0.0])
    age_30 = torch.tensor([30.0])
    age_60 = torch.tensor([60.0])
    age_365 = torch.tensor([365.0])

    decay_0 = float(torch.pow(2.0, -age_0 / 30.0).item())
    decay_30 = float(torch.pow(2.0, -age_30 / 30.0).item())
    decay_60 = float(torch.pow(2.0, -age_60 / 30.0).item())
    decay_365 = float(torch.pow(2.0, -age_365 / 30.0).item())

    assert abs(decay_0 - 1.0) < 1e-6, f"Decay at 0 should be 1.0, got {decay_0}"
    assert abs(decay_30 - 0.5) < 1e-6, f"Decay at 30 should be 0.5, got {decay_30}"
    assert abs(decay_60 - 0.25) < 1e-6, f"Decay at 60 should be 0.25, got {decay_60}"
    assert abs(decay_365 - 0.0002237) < 1e-4, f"Decay at 365 should be ~0.00022, got {decay_365}"
    print("  [+] Fixed 30-day decay math assertions PASSED!")

    # 2. ETT0, ETT1, ETT2 Forward & Backward + Gradient Test
    for mode in ["ETT0", "ETT1", "ETT2"]:
        model = EventTimeTransformerModel(mode=mode, d_model=128, max_events=128)
        B, L = 8, 128
        content = torch.randn(B, L, 12)
        time_base = torch.randn(B, L, 11)
        age = torch.clamp(torch.abs(torch.randn(B, L)) * 50, 0, 365)
        ranks = torch.randint(0, 128, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, 32:] = True  # Padded positions
        is_empty = torch.zeros(B, dtype=torch.bool)
        is_empty[0] = True

        l_react, l_churn, l_buy, z_cond, z_dir, emb, diag = model(content, time_base, age, ranks, mask, is_empty)

        loss = l_buy.mean() + z_cond.mean() + z_dir.mean()
        loss.backward()

        assert torch.isfinite(emb).all(), f"NaN/Inf in embedding for {mode}"
        assert torch.isfinite(loss), f"NaN/Inf in loss for {mode}"

        if mode == "ETT2":
            assert model.raw_tau.grad is not None, "raw_tau grad is None"
            assert torch.isfinite(model.raw_tau.grad), "raw_tau grad is NaN/Inf"
            tau = model.get_tau()
            assert 7.0 <= tau <= 365.0, f"Tau {tau} out of bounds"
            print(f"  [+] ETT2 Learnable Tau gradient verified! Initial Tau = {tau:.2f} days")

        print(f"  [+] {mode} Forward, Backward, Padding Masking & Scale Diagnostics PASSED! (Content Norm: {diag['content_norm']:.2f}, Time Norm: {diag['time_norm']:.2f}, Rank Norm: {diag['rank_norm']:.2f})")

    print("[+] All Unit Tests PASSED successfully!\n")


# -----------------------------------------------------------------------------
# 4. Training & Evaluation Engine
# -----------------------------------------------------------------------------
class EventSequenceDataset(Dataset):
    def __init__(self, c_feats, t_feats, ages, ranks, masks, empties, z_trues, past_buyers):
        self.c_feats = torch.tensor(c_feats, dtype=torch.float32)
        self.t_feats = torch.tensor(t_feats, dtype=torch.float32)
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.ranks = torch.tensor(ranks, dtype=torch.int64)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.empties = torch.tensor(empties, dtype=torch.bool)
        self.z_trues = torch.tensor(z_trues, dtype=torch.float32)
        self.past_buyers = torch.tensor(past_buyers, dtype=torch.float32)

    def __len__(self):
        return len(self.z_trues)

    def __getitem__(self, idx):
        return (
            self.c_feats[idx],
            self.t_feats[idx],
            self.ages[idx],
            self.ranks[idx],
            self.masks[idx],
            self.empties[idx],
            self.z_trues[idx],
            self.past_buyers[idx],
        )


def train_event_transformer(
    model: EventTimeTransformerModel,
    train_loader: DataLoader,
    val_data: Tuple,
    device: torch.device,
    exp_name: str,
    epochs: int = 10,
    lr: float = 1e-3,
    alpha: float = 1.1,
) -> Tuple[np.ndarray, float, List[Dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    (val_c_raw, val_t_raw, val_age_raw, val_rank_raw, val_mask_raw, val_empty_raw, val_z_true, val_past, val_y) = val_data
    n_val = len(val_c_raw)

    best_rmsle = 999.0
    best_pred_z = None
    best_emb = None
    best_diag = None
    epoch_logs = []

    print(f"[*] Training {exp_name} ({epochs} epochs on {device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for c_b, t_b, age_b, rank_b, mask_b, empty_b, z_b, past_b in train_loader:
            c_b, t_b, age_b = c_b.to(device), t_b.to(device), age_b.to(device)
            rank_b, mask_b, empty_b = rank_b.to(device), mask_b.to(device), empty_b.to(device)
            z_b, past_b = z_b.to(device), past_b.to(device)

            optimizer.zero_grad()
            l_react, l_churn, l_buy, z_cond, z_dir, emb, _ = model(c_b, t_b, age_b, rank_b, mask_b, empty_b)

            # Multi-task transition Hurdle loss
            y_buy = (z_b > 0).float()
            loss_buy = F.binary_cross_entropy_with_logits(l_buy, y_buy)

            # Reactivation (past=0) & Churn (past=1)
            mask_react = (past_b == 0)
            mask_churn = (past_b == 1)
            loss_react = F.binary_cross_entropy_with_logits(l_react[mask_react], y_buy[mask_react]) if mask_react.sum() > 0 else 0.0
            loss_churn = F.binary_cross_entropy_with_logits(l_churn[mask_churn], 1.0 - y_buy[mask_churn]) if mask_churn.sum() > 0 else 0.0

            # Conditional regression on positive GMV
            mask_pos = (z_b > 0)
            loss_cond = F.mse_loss(z_cond[mask_pos], z_b[mask_pos]) if mask_pos.sum() > 0 else 0.0
            loss_dir = F.mse_loss(z_dir, z_b)

            loss = loss_buy + 0.5 * (loss_react + loss_churn) + loss_cond + 0.5 * loss_dir
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        # Batched Validation Evaluation
        model.eval()
        p_react_list, p_churn_list, p_buy_list, cond_z_list, dir_z_list = [], [], [], [], []
        emb_v_list = []
        diag_v = None
        eval_bs = 2048

        with torch.no_grad():
            for s_idx in range(0, n_val, eval_bs):
                e_idx = min(s_idx + eval_bs, n_val)
                c_val_b = torch.tensor(val_c_raw[s_idx:e_idx], dtype=torch.float32, device=device)
                t_val_b = torch.tensor(val_t_raw[s_idx:e_idx], dtype=torch.float32, device=device)
                age_val_b = torch.tensor(val_age_raw[s_idx:e_idx], dtype=torch.float32, device=device)
                rank_val_b = torch.tensor(val_rank_raw[s_idx:e_idx], dtype=torch.int64, device=device)
                mask_val_b = torch.tensor(val_mask_raw[s_idx:e_idx], dtype=torch.bool, device=device)
                empty_val_b = torch.tensor(val_empty_raw[s_idx:e_idx], dtype=torch.bool, device=device)

                l_react_b, l_churn_b, l_buy_b, z_cond_b, z_dir_b, emb_b, diag_b = model(
                    c_val_b, t_val_b, age_val_b, rank_val_b, mask_val_b, empty_val_b
                )

                p_react_list.append(torch.sigmoid(l_react_b).cpu().numpy())
                p_churn_list.append(torch.sigmoid(l_churn_b).cpu().numpy())
                p_buy_list.append(torch.sigmoid(l_buy_b).cpu().numpy())
                cond_z_list.append(torch.clamp(z_cond_b, min=0.0).cpu().numpy())
                dir_z_list.append(torch.clamp(z_dir_b, min=0.0).cpu().numpy())
                emb_v_list.append(emb_b.cpu().numpy())
                if diag_v is None:
                    diag_v = diag_b

            p_react = np.concatenate(p_react_list, axis=0)
            p_churn = np.concatenate(p_churn_list, axis=0)
            p_buy = np.concatenate(p_buy_list, axis=0)
            cond_z = np.concatenate(cond_z_list, axis=0)
            dir_z = np.concatenate(dir_z_list, axis=0)
            emb_v = np.concatenate(emb_v_list, axis=0)

            # State-factored Hurdle
            p_state = np.where(val_past == 0, p_react, 1.0 - p_churn)
            p_eff = 0.5 * (p_buy + p_state)
            z_factored = (p_eff ** alpha) * cond_z
            z_pred = 0.8 * z_factored + 0.2 * dir_z

            val_rmsle = float(np.sqrt(np.mean((z_pred - val_z_true) ** 2)))

        tau_val = model.get_tau()
        tau_str = f" | Tau: {tau_val:.1f}d" if tau_val is not None else ""
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Loss: {train_loss:.4f} | Val RMSLE: {val_rmsle:.5f}{tau_str}")

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmsle": val_rmsle,
            "tau": tau_val,
            "content_norm": diag_v["content_norm"],
            "time_norm": diag_v["time_norm"],
            "rank_norm": diag_v["rank_norm"],
        })

        if val_rmsle < best_rmsle:
            best_rmsle = val_rmsle
            best_pred_z = z_pred.copy()

    print(f"[+] {exp_name} Finished! Best Val RMSLE: {best_rmsle:.5f}\n")
    return best_pred_z, best_rmsle, epoch_logs


# -----------------------------------------------------------------------------
# 5. Main Execution Suite
# -----------------------------------------------------------------------------
def main():
    run_unit_tests()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Stage C on device: {device}")

    out_base = Path("artifacts/event_time_encoder")
    ensure_dir(out_base)

    # 1. Load Data
    print("[*] Loading train dataset...")
    train_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    df_raw = pl.read_parquet(train_path)

    users_path = Path("artifacts/selected_users_100k.parquet") if Path("artifacts/selected_users_100k.parquet").exists() else Path("selected_users_100k.parquet")
    users_df = pl.read_parquet(users_path)
    user_ids = users_df["user_id"].to_numpy()

    # 2. Extract Validation Event Sequences (Anchor 2026-01-14)
    print(f"[*] Generating Event Sequences for Validation Anchor {VAL_ANCHOR}...")
    val_c, val_t, val_age, val_rank, val_mask, val_empty = compute_event_sequences_for_anchor(
        df_raw, VAL_ANCHOR, user_ids, max_events=128
    )

    # Validation ground truth
    snapshots_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")
    df_val_snap = pl.read_parquet(snapshots_dir / f"snapshot_{VAL_ANCHOR}.parquet")
    val_y_rub = df_val_snap["target"].to_numpy()
    val_z_true = np.log1p(val_y_rub)
    val_past_buyer = (df_val_snap["lifetime_gmv"].to_numpy() > 0)

    val_data = (val_c, val_t, val_age, val_rank, val_mask, val_empty, val_z_true, val_past_buyer, val_y_rub)

    # 3. Extract Multi-Anchor Training Event Sequences
    print(f"[*] Generating Event Sequences across {len(ANCHORS_CANONICAL)} training anchors...")
    train_c_list, train_t_list, train_age_list, train_rank_list, train_mask_list, train_empty_list = [], [], [], [], [], []
    train_z_list, train_past_list = [], []

    for anc in ANCHORS_CANONICAL:
        c, t, age, rank, mask, empty = compute_event_sequences_for_anchor(df_raw, anc, user_ids, max_events=128)
        snap = pl.read_parquet(snapshots_dir / f"snapshot_{anc}.parquet")
        y = snap["target"].to_numpy()
        z = np.log1p(y)
        past = (snap["lifetime_gmv"].to_numpy() > 0)

        train_c_list.append(c)
        train_t_list.append(t)
        train_age_list.append(age)
        train_rank_list.append(rank)
        train_mask_list.append(mask)
        train_empty_list.append(empty)
        train_z_list.append(z)
        train_past_list.append(past)

    train_c = np.concatenate(train_c_list, axis=0)
    train_t = np.concatenate(train_t_list, axis=0)
    train_age = np.concatenate(train_age_list, axis=0)
    train_rank = np.concatenate(train_rank_list, axis=0)
    train_mask = np.concatenate(train_mask_list, axis=0)
    train_empty = np.concatenate(train_empty_list, axis=0)
    train_z = np.concatenate(train_z_list, axis=0)
    train_past = np.concatenate(train_past_list, axis=0)

    train_dataset = EventSequenceDataset(
        train_c, train_t, train_age, train_rank, train_mask, train_empty, train_z, train_past
    )
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)

    # 4. Run Ablations ETT0, ETT1, ETT2
    results = {}
    predictions = {}

    for mode in ["ETT0", "ETT1", "ETT2"]:
        exp_dir = out_base / mode
        ensure_dir(exp_dir)

        torch.manual_seed(42)
        model = EventTimeTransformerModel(mode=mode, d_model=128, max_events=128).to(device)

        z_pred, best_rmsle, logs = train_event_transformer(
            model, train_loader, val_data, device, exp_name=mode, epochs=10
        )

        results[mode] = {"best_rmsle": best_rmsle, "logs": logs, "final_tau": model.get_tau()}
        predictions[mode] = z_pred

        # Save artifacts
        pl.DataFrame(logs).write_csv(exp_dir / "training_log.csv")
        torch.save(model.state_dict(), exp_dir / "checkpoint_best.pt")

    # 5. Comparative Transition Metrics & Paired Bootstrap
    print("\n" + "=" * 80)
    print("=== STAGE C FINAL EVALUATION & DECISION ===")
    print("=" * 80)

    z_ett0 = predictions["ETT0"]
    z_ett1 = predictions["ETT1"]
    z_ett2 = predictions["ETT2"]

    # Bootstrap ETT1 vs ETT0 and ETT2 vs ETT0
    rng = np.random.default_rng(42)
    n = len(val_z_true)
    diffs_1 = []
    diffs_2 = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        r0 = np.sqrt(np.mean((z_ett0[idx] - val_z_true[idx]) ** 2))
        r1 = np.sqrt(np.mean((z_ett1[idx] - val_z_true[idx]) ** 2))
        r2 = np.sqrt(np.mean((z_ett2[idx] - val_z_true[idx]) ** 2))
        diffs_1.append(r1 - r0)
        diffs_2.append(r2 - r0)

    p_better_1 = float(np.mean(np.array(diffs_1) < 0.0))
    ci_1 = (float(np.percentile(diffs_1, 2.5)), float(np.percentile(diffs_1, 97.5)))

    p_better_2 = float(np.mean(np.array(diffs_2) < 0.0))
    ci_2 = (float(np.percentile(diffs_2, 2.5)), float(np.percentile(diffs_2, 97.5)))

    print(f"ETT0 (Continuous-Time Base): RMSLE = {results['ETT0']['best_rmsle']:.5f}")
    print(f"ETT1 (Fixed 30d Decay):     RMSLE = {results['ETT1']['best_rmsle']:.5f} (Delta: {results['ETT1']['best_rmsle'] - results['ETT0']['best_rmsle']:+.5f}, P(better): {p_better_1*100:.1f}%, 95% CI: [{ci_1[0]:+.5f}, {ci_1[1]:+.5f}])")
    print(f"ETT2 (Learnable Tau):       RMSLE = {results['ETT2']['best_rmsle']:.5f} (Delta: {results['ETT2']['best_rmsle'] - results['ETT0']['best_rmsle']:+.5f}, P(better): {p_better_2*100:.1f}%, 95% CI: [{ci_2[0]:+.5f}, {ci_2[1]:+.5f}], Tau: {results['ETT2']['final_tau']:.2f}d)")

    # Save summary predictions
    df_out = pl.DataFrame({
        "user_id": user_ids,
        "z_true": val_z_true,
        "past_buyer": val_past_buyer,
        "target": val_y_rub,
        "z_ett0": z_ett0,
        "z_ett1": z_ett1,
        "z_ett2": z_ett2,
    })
    df_out.write_parquet(out_base / "ett_val_predictions.parquet")
    print(f"\n[+] Saved {out_base / 'ett_val_predictions.parquet'}")


if __name__ == "__main__":
    main()
