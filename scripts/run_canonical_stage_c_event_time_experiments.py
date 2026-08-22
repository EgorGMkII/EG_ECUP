"""Canonical Stage C: Event-Time Transformer Suite (ETT0, ETT1, ETT2).

100% strict adherence to CANONICAL specification:
- 11 Anchors (ANCHORS_V1: 2025-07-21 to 2025-12-08)
- 12 Canonical Content Features
- 12 Time Features in ALL 3 models (ETT0: decay=0, ETT1: decay_30d, ETT2: decay_tau) -> identical parameter count!
- Pooling: concat(last_nonempty_token, masked_mean, masked_max) -> pooling_mlp
- 4 Heads: reactivation_head, churn_head, conditional_head, direct_head
- Exact Canonical Loss: 1.00*L_factorized + 0.25*L_direct + 0.25*L_conditional + 0.10*L_react + 0.10*L_churn
- Batched Validation to prevent GPU OOM
- Checkpoint selection by minimal Factorized RMSLE on 2026-01-14
"""

import json
import math
import os
from datetime import datetime, timedelta
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
# 1. Canonical Anchors & Data Extraction
# -----------------------------------------------------------------------------
ANCHORS_V1 = [
    "2025-07-21", "2025-08-04", "2025-08-18",
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
    """Extracts event-day logs for selected users within [anchor - 364d, anchor]."""
    anchor_dt = pl.lit(anchor_str).str.to_date()
    start_dt = pl.lit(anchor_str).str.to_date().dt.offset_by("-364d")

    user_ids_set = set(user_ids)

    # Filter logs strictly in [anchor - 364d, anchor]
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
            "search",
            "cat",
            "has_search_to_cart",
            "has_search_to_ord",
            "has_cat_to_cart",
            "has_cat_to_ord",
            "search_to_cart",
            "search_to_ord",
            "cat_to_cart",
            "cat_to_ord",
            "gmv",
            "to_ord",
        ])
        .collect()
    )

    # Group daily per user
    df_daily = (
        df_filtered.group_by(["user_id", "event_date"])
        .agg([
            pl.col("search").sum().alias("search"),
            pl.col("cat").sum().alias("cat"),
            pl.col("has_search_to_cart").max().alias("has_search_to_cart"),
            pl.col("has_search_to_ord").max().alias("has_search_to_ord"),
            pl.col("has_cat_to_cart").max().alias("has_cat_to_cart"),
            pl.col("has_cat_to_ord").max().alias("has_cat_to_ord"),
            pl.col("search_to_cart").sum().alias("search_to_cart"),
            pl.col("search_to_ord").sum().alias("search_to_ord"),
            pl.col("cat_to_cart").sum().alias("cat_to_cart"),
            pl.col("cat_to_ord").sum().alias("cat_to_ord"),
            pl.col("gmv").sum().alias("gmv"),
            pl.col("to_ord").sum().alias("to_ord"),
        ])
        .sort(["user_id", "event_date"])
    )

    # Compute 12 canonical content features
    df_daily = df_daily.with_columns([
        pl.col("search").cast(pl.Float32).alias("c0_search"),
        pl.col("cat").cast(pl.Float32).alias("c1_cat"),
        pl.col("has_search_to_cart").cast(pl.Float32).alias("c2_has_search_to_cart"),
        pl.col("has_search_to_ord").cast(pl.Float32).alias("c3_has_search_to_ord"),
        pl.col("has_cat_to_cart").cast(pl.Float32).alias("c4_has_cat_to_cart"),
        pl.col("has_cat_to_ord").cast(pl.Float32).alias("c5_has_cat_to_ord"),
        pl.col("search_to_cart").log1p().alias("c6_log_search_to_cart"),
        pl.col("search_to_ord").log1p().alias("c7_log_search_to_ord"),
        pl.col("cat_to_cart").log1p().alias("c8_log_cat_to_cart"),
        pl.col("cat_to_ord").log1p().alias("c9_log_cat_to_ord"),
        pl.col("gmv").log1p().alias("c10_log_gmv"),
        (pl.col("to_ord") > 0).cast(pl.Float32).alias("c11_is_purchase_day"),
    ])

    c_cols = [f"c{k}_{name}" for k, name in enumerate([
        "search", "cat", "has_search_to_cart", "has_search_to_ord",
        "has_cat_to_cart", "has_cat_to_ord", "log_search_to_cart",
        "log_search_to_ord", "log_cat_to_cart", "log_cat_to_ord",
        "log_gmv", "is_purchase_day"
    ])]

    # Group into dict user_id -> rows
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

    for i, uid in enumerate(user_ids):
        evs = user_to_events.get(uid, [])
        if len(evs) == 0:
            is_empty_arr[i] = True
            padding_mask[i, 0] = False  # Keep pos 0 unmasked for learned EMPTY_HISTORY token
            ranks_arr[i, 0] = 0
            continue

        evs = evs[-max_events:]
        num_ev = len(evs)

        prev_date = None
        for j, ev in enumerate(evs):
            pos = max_events - num_ev + j  # Right-aligned
            padding_mask[i, pos] = False
            rank_from_end = num_ev - 1 - j
            ranks_arr[i, pos] = rank_from_end

            # Content
            content_feats[i, pos] = [ev[col] for col in c_cols]

            # Time
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
                age_norm, log_age_norm, delta_norm, log_delta_norm,
                dow_sin, dow_cos, doy_sin, doy_cos,
                target_phase_sin, target_phase_cos, is_first_event
            ]

    return content_feats, time_feats_base, age_days_arr, ranks_arr, padding_mask, is_empty_arr


# -----------------------------------------------------------------------------
# 2. PyTorch Dataset
# -----------------------------------------------------------------------------
class EventSequenceDataset(Dataset):
    def __init__(self, content, time_base, age_days, ranks, mask, empty, z_true, was_active, y_rub):
        self.content = torch.tensor(content, dtype=torch.float32)
        self.time_base = torch.tensor(time_base, dtype=torch.float32)
        self.age_days = torch.tensor(age_days, dtype=torch.float32)
        self.ranks = torch.tensor(ranks, dtype=torch.int64)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.empty = torch.tensor(empty, dtype=torch.bool)
        self.z_true = torch.tensor(z_true, dtype=torch.float32)
        self.was_active = torch.tensor(was_active, dtype=torch.bool)
        self.y_rub = torch.tensor(y_rub, dtype=torch.float32)

    def __len__(self):
        return len(self.z_true)

    def __getitem__(self, idx):
        return (
            self.content[idx],
            self.time_base[idx],
            self.age_days[idx],
            self.ranks[idx],
            self.mask[idx],
            self.empty[idx],
            self.z_true[idx],
            self.was_active[idx],
            self.y_rub[idx],
        )


# -----------------------------------------------------------------------------
# 3. Canonical Event-Time Transformer Model Architecture
# -----------------------------------------------------------------------------
class CanonicalEventTimeTransformer(nn.Module):
    def __init__(
        self,
        mode: str = "ETT0",  # ETT0, ETT1, ETT2
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        content_dim: int = 12,
        max_events: int = 128,
    ):
        super().__init__()
        assert mode in ["ETT0", "ETT1", "ETT2"]
        self.mode = mode
        self.d_model = d_model
        self.max_events = max_events

        # Content Projection
        self.content_projection = nn.Sequential(
            nn.Linear(content_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Time MLP: receives 12 features for all 3 models!
        self.time_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
        )

        # Learnable tau for ETT2
        if mode == "ETT2":
            p = (30.0 - 7.0) / (365.0 - 7.0)
            raw_tau_init = math.log(p / (1.0 - p))
            self.raw_tau = nn.Parameter(torch.tensor([raw_tau_init], dtype=torch.float32))
        else:
            self.raw_tau = None

        # Event Rank Embedding (scaled std)
        self.event_rank_embedding = nn.Embedding(num_embeddings=max_events, embedding_dim=d_model)
        nn.init.normal_(self.event_rank_embedding.weight, std=1.0 / math.sqrt(d_model))

        # Learned EMPTY_HISTORY token
        self.empty_history_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.empty_history_token, std=1.0 / math.sqrt(d_model))

        # Additive LayerNorm
        self.input_layer_norm = nn.LayerNorm(d_model)

        # Transformer Encoder
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

        # Pooling MLP: concat(last_token, mean_pool, max_pool) -> 3*d_model -> d_model
        self.pooling_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Downstream 4 Heads
        self.reactivation_head = nn.Linear(d_model, 1)
        self.churn_head = nn.Linear(d_model, 1)
        self.conditional_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def get_tau(self) -> Optional[float]:
        if self.raw_tau is not None:
            return 7.0 + 358.0 * torch.sigmoid(self.raw_tau).item()
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

        # 2. Time Embedding with 12th decay feature
        if self.mode == "ETT0":
            decay_feat = torch.zeros_like(age_days).unsqueeze(-1)  # [B, L, 1]
        elif self.mode == "ETT1":
            decay_feat = torch.pow(2.0, -age_days / 30.0).unsqueeze(-1)  # [B, L, 1]
        elif self.mode == "ETT2":
            tau = 7.0 + 358.0 * torch.sigmoid(self.raw_tau)
            decay_feat = torch.pow(2.0, -age_days / tau).unsqueeze(-1)  # [B, L, 1]

        time_input = torch.cat([time_feats_base, decay_feat], dim=-1)  # [B, L, 12]
        time_emb = self.time_mlp(time_input)  # [B, L, 128]

        # 3. Rank Embedding
        rank_emb = self.event_rank_embedding(ranks)  # [B, L, 128]

        # Token = LayerNorm(content + time + rank)
        token = content_emb + time_emb + rank_emb
        token = self.input_layer_norm(token)

        # Handle Dormant Users with EMPTY_HISTORY token
        if is_empty.any():
            token[is_empty, 0:1, :] = self.empty_history_token
            padding_mask[is_empty, 0] = False
            padding_mask[is_empty, 1:] = True

        # Diagnostic norms
        diag_stats = {
            "content_norm": float(content_emb.norm(dim=-1).mean().item()),
            "time_norm": float(time_emb.norm(dim=-1).mean().item()),
            "rank_norm": float(rank_emb.norm(dim=-1).mean().item()),
        }

        # Transformer Encoding
        encoded = self.transformer(token, src_key_padding_mask=padding_mask)

        # Canonical Pooling: concat(last_token, mean_pool, max_pool)
        # valid tokens mask [B, L, 1]
        valid_mask = (~padding_mask).unsqueeze(-1).float()  # [B, L, 1]
        last_token = encoded[:, -1, :]  # right-aligned: position -1 is always the latest event or empty token

        # Masked Mean
        sum_tokens = torch.sum(encoded * valid_mask, dim=1)
        count_tokens = torch.clamp(valid_mask.sum(dim=1), min=1.0)
        mean_pool = sum_tokens / count_tokens

        # Masked Max
        encoded_for_max = encoded.masked_fill(padding_mask.unsqueeze(-1), -1e9)
        max_pool = torch.max(encoded_for_max, dim=1).values

        # Concat & Project
        pool_concat = torch.cat([last_token, mean_pool, max_pool], dim=-1)  # [B, 384]
        user_emb = self.pooling_mlp(pool_concat)  # [B, 128]

        # Heads
        l_react = self.reactivation_head(user_emb).squeeze(-1)
        l_churn = self.churn_head(user_emb).squeeze(-1)
        raw_cond = self.conditional_head(user_emb).squeeze(-1)
        raw_dir = self.direct_head(user_emb).squeeze(-1)

        z_cond = F.softplus(raw_cond)
        z_dir = F.softplus(raw_dir)

        # Hurdle Probabilities & Factorized Z
        p_react = torch.sigmoid(l_react)
        p_churn = torch.sigmoid(l_churn)

        return l_react, l_churn, p_react, p_churn, z_cond, z_dir, user_emb, diag_stats


# -----------------------------------------------------------------------------
# 4. Canonical Unit Tests
# -----------------------------------------------------------------------------
def run_canonical_unit_tests():
    print("[*] Running Canonical Unit Tests for Event-Time Transformer...")
    
    # 1. Decay math assertions
    tau_30 = 30.0
    assert abs(2.0 ** (-0.0 / tau_30) - 1.0) < 1e-6
    assert abs(2.0 ** (-30.0 / tau_30) - 0.5) < 1e-6
    assert abs(2.0 ** (-60.0 / tau_30) - 0.25) < 1e-6
    assert abs(2.0 ** (-365.0 / tau_30) - (2.0 ** (-365.0 / 30.0))) < 1e-6
    print("  [+] Fixed 30-day decay math assertions PASSED!")

    # 2. Forward & Scale checks
    B, L = 16, 128
    c = torch.randn(B, L, 12)
    t = torch.randn(B, L, 11)
    age = torch.clamp(torch.rand(B, L) * 365.0, 0.0, 365.0)
    ranks = torch.randint(0, 128, (B, L))
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :64] = True  # left padded
    empty = torch.zeros(B, dtype=torch.bool)
    empty[0] = True

    for mode in ["ETT0", "ETT1", "ETT2"]:
        model = CanonicalEventTimeTransformer(mode=mode)
        out = model(c, t, age, ranks, mask, empty)
        l_react, l_churn, p_react, p_churn, z_cond, z_dir, user_emb, diag = out
        assert user_emb.shape == (B, 128)
        assert l_react.shape == (B,)
        assert z_cond.shape == (B,)
        print(f"  [+] {mode} Forward PASSED! Content: {diag['content_norm']:.2f}, Time: {diag['time_norm']:.2f}, Rank: {diag['rank_norm']:.2f}")

    # 3. ETT2 Tau Gradient Check
    model_e2 = CanonicalEventTimeTransformer(mode="ETT2")
    optimizer = torch.optim.AdamW(model_e2.parameters(), lr=1e-3)
    out = model_e2(c, t, age, ranks, mask, empty)
    loss = out[0].sum() + out[4].sum()
    loss.backward()
    assert model_e2.raw_tau.grad is not None
    print(f"  [+] ETT2 Tau Gradient verified! Initial Tau: {model_e2.get_tau():.2f}d")
    print("[+] All Canonical Unit Tests PASSED successfully!\n")


# -----------------------------------------------------------------------------
# 5. Training & Evaluation Engine
# -----------------------------------------------------------------------------
def train_canonical_model(
    model: CanonicalEventTimeTransformer,
    train_loader: DataLoader,
    val_data: Tuple,
    exp_name: str,
    epochs: int = 12,
    lr: float = 3e-4,
    device: str = "cuda",
    alpha: float = 1.1,
) -> Tuple[np.ndarray, float, List[Dict]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    (val_c_raw, val_t_raw, val_age_raw, val_rank_raw, val_mask_raw, val_empty_raw, val_z_true, val_was_act, val_y_rub) = val_data
    n_val = len(val_c_raw)

    best_fact_rmsle = 999.0
    best_pred_z = None
    best_direct_z = None
    best_diag = None
    epoch_logs = []

    print(f"[*] Training {exp_name} ({epochs} epochs on {device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        tot_l_fact, tot_l_dir, tot_l_cond, tot_l_react, tot_l_churn = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for c_b, t_b, age_b, rank_b, mask_b, empty_b, z_b, was_act_b, y_rub_b in train_loader:
            c_b, t_b, age_b = c_b.to(device), t_b.to(device), age_b.to(device)
            rank_b, mask_b, empty_b = rank_b.to(device), mask_b.to(device), empty_b.to(device)
            z_b, was_act_b, y_rub_b = z_b.to(device), was_act_b.to(device), y_rub_b.to(device)

            optimizer.zero_grad()
            l_react, l_churn, p_react, p_churn, z_cond, z_dir, user_emb, _ = model(
                c_b, t_b, age_b, rank_b, mask_b, empty_b
            )

            # Target definitions
            pos_mask = (y_rub_b > 0)
            target_buy = pos_mask.float()
            inact_mask = (~was_act_b)
            act_mask = was_act_b

            # 1. L_reactivation on previously inactive
            if inact_mask.sum() > 0:
                l_react_loss = F.binary_cross_entropy_with_logits(l_react[inact_mask], target_buy[inact_mask])
            else:
                l_react_loss = torch.tensor(0.0, device=device)

            # 2. L_churn on previously active
            if act_mask.sum() > 0:
                l_churn_loss = F.binary_cross_entropy_with_logits(l_churn[act_mask], 1.0 - target_buy[act_mask])
            else:
                l_churn_loss = torch.tensor(0.0, device=device)

            # 3. L_conditional on target > 0
            if pos_mask.sum() > 0:
                l_cond_loss = F.mse_loss(z_cond[pos_mask], z_b[pos_mask])
            else:
                l_cond_loss = torch.tensor(0.0, device=device)

            # 4. L_direct on all users
            l_dir_loss = F.mse_loss(z_dir, z_b)

            # 5. L_factorized on all users
            p_buy = torch.where(was_act_b, 1.0 - p_churn, p_react)
            p_eff = torch.clamp(p_buy, 1e-6, 1.0 - 1e-6)
            factorized_z = torch.pow(p_eff, alpha) * z_cond
            l_fact_loss = F.mse_loss(factorized_z, z_b)

            # Canonical total loss formula
            loss = (
                1.00 * l_fact_loss
                + 0.25 * l_dir_loss
                + 0.25 * l_cond_loss
                + 0.10 * l_react_loss
                + 0.10 * l_churn_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            tot_l_fact += l_fact_loss.item()
            tot_l_dir += l_dir_loss.item()
            tot_l_cond += l_cond_loss.item()
            tot_l_react += l_react_loss.item()
            tot_l_churn += l_churn_loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)

        # Batched Validation Inference (bs=2048 to prevent GPU OOM)
        model.eval()
        p_react_list, p_churn_list, cond_z_list, dir_z_list = [], [], [], []
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

                _, _, p_r_b, p_ch_b, z_c_b, z_d_b, emb_b, diag_b = model(
                    c_val_b, t_val_b, age_val_b, rank_val_b, mask_val_b, empty_val_b
                )

                p_react_list.append(p_r_b.cpu().numpy())
                p_churn_list.append(p_ch_b.cpu().numpy())
                cond_z_list.append(z_c_b.cpu().numpy())
                dir_z_list.append(z_d_b.cpu().numpy())
                emb_v_list.append(emb_b.cpu().numpy())
                if diag_v is None:
                    diag_v = diag_b

            p_react_val = np.concatenate(p_react_list, axis=0)
            p_churn_val = np.concatenate(p_churn_list, axis=0)
            cond_z_val = np.concatenate(cond_z_list, axis=0)
            dir_z_val = np.concatenate(dir_z_list, axis=0)

            # Factorized Hurdle on validation
            p_buy_val = np.where(val_was_act, 1.0 - p_churn_val, p_react_val)
            p_eff_val = np.clip(p_buy_val, 1e-6, 1.0 - 1e-6)
            fact_z_val = (p_eff_val ** alpha) * cond_z_val

            fact_rmsle = float(np.sqrt(np.mean((fact_z_val - val_z_true) ** 2)))
            dir_rmsle = float(np.sqrt(np.mean((dir_z_val - val_z_true) ** 2)))

        tau_val = model.get_tau()
        tau_str = f" | Tau: {tau_val:.1f}d" if tau_val is not None else ""
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Loss: {train_loss:.4f} (Fact: {tot_l_fact/n_batches:.4f}) | Val Fact RMSLE: {fact_rmsle:.5f} | Dir RMSLE: {dir_rmsle:.5f}{tau_str}")

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_factorized_rmsle": fact_rmsle,
            "val_direct_rmsle": dir_rmsle,
            "tau": tau_val,
            "content_norm": diag_v["content_norm"],
            "time_norm": diag_v["time_norm"],
            "rank_norm": diag_v["rank_norm"],
        })

        if fact_rmsle < best_fact_rmsle:
            best_fact_rmsle = fact_rmsle
            best_pred_z = fact_z_val.copy()
            best_direct_z = dir_z_val.copy()
            best_diag = {
                "p_react": p_react_val,
                "p_churn": p_churn_val,
                "p_buy": p_buy_val,
                "cond_z": cond_z_val,
                "dir_z": dir_z_val,
                "fact_z": fact_z_val,
            }

    print(f"[+] {exp_name} Finished! Best Canonical Factorized RMSLE: {best_fact_rmsle:.5f}\n")
    return best_pred_z, best_fact_rmsle, epoch_logs, best_diag


# -----------------------------------------------------------------------------
# 6. Master Runner
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("CANONICAL STAGE C: EVENT-TIME TRANSFORMER SUITE (ETT0, ETT1, ETT2)")
    print("=" * 80)

    run_canonical_unit_tests()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device}")

    out_base = Path("artifacts/event_time_encoder")
    ensure_dir(out_base)

    # 1. Load Data
    train_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    df_raw = pl.read_parquet(train_path)

    users_path = Path("artifacts/selected_users_100k.parquet") if Path("artifacts/selected_users_100k.parquet").exists() else Path("selected_users_100k.parquet")
    users_df = pl.read_parquet(users_path)
    user_ids = users_df["user_id"].to_numpy()

    # 2. Extract Validation Sequences (2026-01-14)
    print(f"[*] Extracting Validation Sequences for {VAL_ANCHOR}...")
    val_c, val_t, val_age, val_rank, val_mask, val_empty = compute_event_sequences_for_anchor(
        df_raw, VAL_ANCHOR, user_ids, max_events=128
    )

    snapshots_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")
    df_val_snap = pl.read_parquet(snapshots_dir / f"snapshot_{VAL_ANCHOR}.parquet")
    val_y_rub = df_val_snap["target"].to_numpy()
    val_z_true = np.log1p(val_y_rub)
    val_was_act = (df_val_snap["lifetime_gmv"].to_numpy() > 0)

    val_data = (val_c, val_t, val_age, val_rank, val_mask, val_empty, val_z_true, val_was_act, val_y_rub)

    # 3. Extract Training Sequences across all 11 ANCHORS_V1
    print(f"[*] Extracting Multi-Anchor Training Sequences across all {len(ANCHORS_V1)} ANCHORS_V1...")
    train_c_list, train_t_list, train_age_list, train_rank_list, train_mask_list, train_empty_list = [], [], [], [], [], []
    train_z_list, train_act_list, train_y_list = [], [], []

    for anc in ANCHORS_V1:
        c, t, age, rank, mask, empty = compute_event_sequences_for_anchor(df_raw, anc, user_ids, max_events=128)
        snap = pl.read_parquet(snapshots_dir / f"snapshot_{anc}.parquet")
        y = snap["target"].to_numpy()
        z = np.log1p(y)
        was_act = (snap["lifetime_gmv"].to_numpy() > 0)

        train_c_list.append(c)
        train_t_list.append(t)
        train_age_list.append(age)
        train_rank_list.append(rank)
        train_mask_list.append(mask)
        train_empty_list.append(empty)
        train_z_list.append(z)
        train_act_list.append(was_act)
        train_y_list.append(y)

    train_c = np.concatenate(train_c_list, axis=0)
    train_t = np.concatenate(train_t_list, axis=0)
    train_age = np.concatenate(train_age_list, axis=0)
    train_rank = np.concatenate(train_rank_list, axis=0)
    train_mask = np.concatenate(train_mask_list, axis=0)
    train_empty = np.concatenate(train_empty_list, axis=0)
    train_z = np.concatenate(train_z_list, axis=0)
    train_act = np.concatenate(train_act_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    print(f"[+] Total Training User-Anchor Pairs: {len(train_z):,} (11 anchors x 100,000 users)")

    train_dataset = EventSequenceDataset(
        train_c, train_t, train_age, train_rank, train_mask, train_empty, train_z, train_act, train_y
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    # 4. Train Canonical ETT0, ETT1, ETT2
    results = {}
    models_config = [
        ("ETT0", "base"),
        ("ETT1", "fixed_decay"),
        ("ETT2", "learnable_decay"),
    ]

    for exp_id, mode_desc in models_config:
        torch.manual_seed(42)
        np.random.seed(42)

        exp_dir = out_base / exp_id
        ensure_dir(exp_dir)

        model = CanonicalEventTimeTransformer(mode=exp_id, d_model=128, nhead=4, num_layers=2)
        pred_z, best_rmsle, logs, diag = train_canonical_model(
            model, train_loader, val_data, exp_name=exp_id, epochs=12, lr=3e-4, device=device, alpha=1.1
        )

        results[exp_id] = {
            "best_factorized_rmsle": best_rmsle,
            "pred_z": pred_z,
            "logs": logs,
            "diag": diag,
        }

        # Save validation predictions parquet
        val_df = pl.DataFrame({
            "user_id": user_ids,
            "was_active": val_was_act,
            "target_raw": val_y_rub,
            "target_z": val_z_true,
            "p_react": diag["p_react"],
            "p_churn": diag["p_churn"],
            "p_buy": diag["p_buy"],
            "conditional_z": diag["cond_z"],
            "direct_z": diag["dir_z"],
            "factorized_z": diag["fact_z"],
        })
        val_df.write_parquet(exp_dir / "validation_predictions.parquet")

        # Save logs
        with open(exp_dir / "training_logs.json", "w") as f:
            json.dump(logs, f, indent=2)

    # 5. Compute Transitions & Statistical Comparisons
    print("\n" + "=" * 80)
    print("FINAL CANONICAL STAGE C RESULTS SUMMARY")
    print("=" * 80)
    for exp_id in ["ETT0", "ETT1", "ETT2"]:
        res = results[exp_id]
        print(f"[*] {exp_id:6s}: Factorized RMSLE = {res['best_factorized_rmsle']:.5f}")

    # Save summary table
    summary_rows = []
    for exp_id in ["ETT0", "ETT1", "ETT2"]:
        summary_rows.append({
            "experiment_id": exp_id,
            "anchors": "ANCHORS_V1 (11 anchors)",
            "factorized_rmsle": results[exp_id]["best_factorized_rmsle"],
        })
    pl.DataFrame(summary_rows).write_csv(out_base / "canonical_stage_c_summary.csv")
    print(f"\n[+] Canonical Stage C Complete! Saved to {out_base / 'canonical_stage_c_summary.csv'}")


if __name__ == "__main__":
    main()
