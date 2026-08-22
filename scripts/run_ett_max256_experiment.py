"""Controlled Sequence Length Experiment for Event-Time Transformer: OPT_MAX256.

Memory-Optimized Implementation:
- In-place preallocated contiguous numpy arrays (eliminates np.concatenate memory spike)
- int16 event ranks (saves 75% memory on rank indices)
- Lazy scanning and garbage collection per anchor
- micro_batch_size = 128, grad_accum_steps = 2 (effective batch_size = 256)
- max_lr = 3e-4 (warmup = 500, cosine decay to 10%)
- Step-level monitoring on 20k subset every 250 steps, full 100k val every 1000 steps
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ANCHORS_V1 = [
    "2025-07-21", "2025-08-04", "2025-08-18",
    "2025-09-01", "2025-09-15", "2025-09-29",
    "2025-10-13", "2025-10-27", "2025-11-10",
    "2025-11-24", "2025-12-08"
]
VAL_ANCHOR = "2026-01-14"


def extract_features_into_buffers(
    df_raw: pl.LazyFrame,
    anchor_str: str,
    user_ids: np.ndarray,
    out_c: np.ndarray,
    out_t: np.ndarray,
    out_r: np.ndarray,
    out_m: np.ndarray,
    out_emp: np.ndarray,
    offset: int = 0,
    max_events: int = 256,
    tau_days: float = 30.0,
):
    anchor_dt = pl.lit(anchor_str).str.to_date()
    start_dt = pl.lit(anchor_str).str.to_date().dt.offset_by("-364d")
    user_ids_set = set(user_ids)

    df_filtered = (
        df_raw
        .filter(
            (pl.col("event_date") >= start_dt)
            & (pl.col("event_date") <= anchor_dt)
            & (pl.col("user_id").is_in(user_ids_set))
        )
        .select([
            "user_id", "event_date", "search", "cat",
            "has_search_to_cart", "has_search_to_ord",
            "has_cat_to_cart", "has_cat_to_ord",
            "search_to_cart", "search_to_ord",
            "cat_to_cart", "cat_to_ord",
            "gmv", "to_ord"
        ])
        .collect()
    )

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

    user_to_events = {}
    anchor_dt_py = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    midpoint_doy = (anchor_dt_py + timedelta(days=15)).timetuple().tm_yday
    anchor_ts = anchor_dt_py

    for row in df_daily.iter_rows(named=True):
        u = row["user_id"]
        if u not in user_to_events:
            user_to_events[u] = []
        user_to_events[u].append(row)

    del df_filtered, df_daily
    gc.collect()

    for i, uid in enumerate(user_ids):
        idx = offset + i
        evs = user_to_events.get(uid, [])
        if len(evs) == 0:
            out_emp[idx] = True
            out_m[idx, 0] = False
            out_r[idx, 0] = 0
            continue

        evs = evs[-max_events:]
        num_ev = len(evs)

        prev_date = None
        for j, ev in enumerate(evs):
            pos = max_events - num_ev + j
            out_m[idx, pos] = False
            rank_from_end = num_ev - 1 - j
            out_r[idx, pos] = rank_from_end

            out_c[idx, pos] = [ev[col] for col in c_cols]

            ev_date = ev["event_date"]
            age_days = float((anchor_ts - ev_date).days)
            age_days = max(0.0, min(365.0, age_days))

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

            decay_val = math.exp(-age_days / tau_days)

            out_t[idx, pos] = [
                age_norm, log_age_norm, delta_norm, log_delta_norm,
                dow_sin, dow_cos, doy_sin, doy_cos,
                target_phase_sin, target_phase_cos, is_first_event,
                decay_val
            ]


class ZeroCopyDataset(Dataset):
    def __init__(self, content, time_feat, ranks, mask, empty, z_true, was_active, y_rub):
        self.content = content
        self.time_feat = time_feat
        self.ranks = ranks
        self.mask = mask
        self.empty = empty
        self.z_true = z_true
        self.was_active = was_active
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
            torch.tensor(self.y_rub[idx], dtype=torch.float32),
        )


class EventTimeTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        max_events: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_events = max_events

        self.content_projection = nn.Sequential(
            nn.Linear(12, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
        )
        self.event_rank_embedding = nn.Embedding(max_events, d_model)
        nn.init.normal_(self.event_rank_embedding.weight, std=1.0 / math.sqrt(d_model))

        self.input_layer_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.empty_history_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

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

    def forward(
        self,
        content: torch.Tensor,
        time_feat: torch.Tensor,
        ranks: torch.Tensor,
        padding_mask: torch.Tensor,
        is_empty: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, s, _ = content.shape

        content_emb = self.content_projection(content)
        time_emb = self.time_mlp(time_feat)
        rank_emb = self.event_rank_embedding(ranks)

        event_token = content_emb + time_emb + rank_emb
        event_token = self.input_layer_norm(event_token)
        event_token = self.input_dropout(event_token)

        empty_exp = self.empty_history_token.expand(b, s, -1)
        event_token = torch.where(is_empty.unsqueeze(1).unsqueeze(2), empty_exp, event_token)

        h = self.transformer_encoder(event_token, src_key_padding_mask=padding_mask)

        last_nonempty_token = h[:, -1, :]
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        sum_pooled = (h * valid_mask).sum(dim=1)
        cnt = valid_mask.sum(dim=1).clamp(min=1.0)
        mean_pooled = sum_pooled / cnt

        h_masked_for_max = h.masked_fill(padding_mask.unsqueeze(-1), -1e9)
        max_pooled = h_masked_for_max.max(dim=1).values
        max_pooled = torch.where(is_empty.unsqueeze(-1), last_nonempty_token, max_pooled)

        pooled_raw = torch.cat([last_nonempty_token, mean_pooled, max_pooled], dim=-1)
        pooled = self.pooling_mlp(pooled_raw)

        logit_react = self.reactivation_head(pooled).squeeze(-1)
        logit_churn = self.churn_head(pooled).squeeze(-1)
        p_react = torch.sigmoid(logit_react)
        p_churn = torch.sigmoid(logit_churn)
        cond_z = F.softplus(self.conditional_head(pooled).squeeze(-1))
        direct_z = F.softplus(self.direct_head(pooled).squeeze(-1))

        return {
            "p_react": p_react,
            "p_churn": p_churn,
            "cond_z": cond_z,
            "direct_z": direct_z,
            "logit_react": logit_react,
            "logit_churn": logit_churn,
        }


def compute_canonical_loss(
    preds: Dict[str, torch.Tensor],
    target_z: torch.Tensor,
    was_active: torch.Tensor,
    alpha: float = 1.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    p_react = preds["p_react"]
    p_churn = preds["p_churn"]
    cond_z = preds["cond_z"]
    direct_z = preds["direct_z"]
    logit_react = preds["logit_react"]
    logit_churn = preds["logit_churn"]

    is_act = was_active.bool()
    is_inact = ~is_act
    y_buy = (target_z > 0.0).float()

    p_buy = torch.where(is_act, 1.0 - p_churn, p_react)
    p_clamped = torch.clamp(p_buy, min=1e-6, max=1.0 - 1e-6)
    factorized_z = (p_clamped ** alpha) * cond_z

    loss_factorized = F.mse_loss(factorized_z, target_z)
    loss_direct = F.mse_loss(direct_z, target_z)

    pos_mask = target_z > 0.0
    if pos_mask.sum() > 0:
        loss_conditional = F.mse_loss(cond_z[pos_mask], target_z[pos_mask])
    else:
        loss_conditional = torch.tensor(0.0, device=target_z.device)

    if is_inact.sum() > 0:
        loss_react = F.binary_cross_entropy_with_logits(logit_react[is_inact], y_buy[is_inact])
    else:
        loss_react = torch.tensor(0.0, device=target_z.device)

    if is_act.sum() > 0:
        y_churn = 1.0 - y_buy[is_act]
        loss_churn = F.binary_cross_entropy_with_logits(logit_churn[is_act], y_churn)
    else:
        loss_churn = torch.tensor(0.0, device=target_z.device)

    total_loss = (
        1.00 * loss_factorized
        + 0.25 * loss_direct
        + 0.25 * loss_conditional
        + 0.10 * loss_react
        + 0.10 * loss_churn
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_fact": float(loss_factorized.item()),
        "loss_dir": float(loss_direct.item()),
    }
    return total_loss, metrics, factorized_z


def evaluate_dataset(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 1024,
    device: str = "cuda",
    alpha: float = 1.1,
) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fact_preds, dir_preds, targets, was_actives, y_rubs = [], [], [], [], []

    with torch.no_grad():
        for content, time_f, ranks, mask, empty, z, was_act, y_rub in loader:
            content = content.to(device)
            time_f = time_f.to(device)
            ranks = ranks.to(device)
            mask = mask.to(device)
            empty = empty.to(device)

            out = model(content, time_f, ranks, mask, empty)
            p_buy = torch.where(was_act.to(device).bool(), 1.0 - out["p_churn"], out["p_react"])
            p_clamped = torch.clamp(p_buy, 1e-6, 1.0 - 1e-6)
            fact_z = (p_clamped ** alpha) * out["cond_z"]

            fact_preds.append(fact_z.cpu().numpy())
            dir_preds.append(out["direct_z"].cpu().numpy())
            targets.append(z.numpy())
            was_actives.append(was_act.numpy())
            y_rubs.append(y_rub.numpy())

    fact_preds = np.concatenate(fact_preds)
    dir_preds = np.concatenate(dir_preds)
    targets = np.concatenate(targets)
    was_actives = np.concatenate(was_actives)
    y_rubs = np.concatenate(y_rubs)

    gmv_pred_fact = np.expm1(fact_preds).clip(min=0.0)
    gmv_pred_dir = np.expm1(dir_preds).clip(min=0.0)

    rmsle_fact = float(np.sqrt(np.mean((np.log1p(gmv_pred_fact) - np.log1p(y_rubs)) ** 2)))
    rmsle_dir = float(np.sqrt(np.mean((np.log1p(gmv_pred_dir) - np.log1p(y_rubs)) ** 2)))

    is_act = (was_actives > 0.0).astype(int)
    y_buy = (y_rubs > 0.0).astype(int)

    metrics = {
        "rmsle_fact": rmsle_fact,
        "rmsle_dir": rmsle_dir,
    }

    for act_state in [0, 1]:
        for buy_state in [0, 1]:
            mask_trans = (is_act == act_state) & (y_buy == buy_state)
            if mask_trans.sum() > 0:
                mse_fact_trans = float(np.mean((fact_preds[mask_trans] - targets[mask_trans]) ** 2))
                metrics[f"mse_fact_{act_state}->{buy_state}"] = mse_fact_trans

    return metrics, fact_preds, dir_preds


def get_lr_with_warmup(step: int, max_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.10) -> float:
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        alpha = step / max(1, warmup_steps)
        return min_lr + alpha * (max_lr - min_lr)
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine_decay * (max_lr - min_lr)


def main():
    print("=" * 80)
    print("CONTROLLED OPT_MAX256 SEQUENCE LENGTH EXPERIMENT FOR EVENT-TIME TRANSFORMER")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Execution device: {device}")

    out_dir = Path("artifacts/ett_optimization/OPT_MAX256")
    out_dir.mkdir(parents=True, exist_ok=True)

    users_path = Path("artifacts/selected_users_100k.parquet")
    if not users_path.exists():
        users_path = Path("selected_users_100k.parquet")
    
    snapshots_dir = Path("data/snapshots")
    if not snapshots_dir.exists():
        snapshots_dir = Path("snapshots")

    users_df = pl.read_parquet(users_path)
    user_ids = users_df["user_id"].to_numpy()
    n_users = len(user_ids)
    print(f"[+] Loaded {n_users} canonical users from {users_path}.")

    train_path = Path("data/train.parquet")
    if not train_path.exists():
        train_path = Path("train.parquet")
    print(f"[*] Scanning raw events lazily from {train_path}...")
    df_raw = pl.scan_parquet(train_path)

    # 1. Allocate Validation Buffers
    print("\n[*] Preallocating and Extracting Validation Sequences (2026-01-14, max_events=256)...")
    val_c = np.zeros((n_users, 256, 12), dtype=np.float32)
    val_t = np.zeros((n_users, 256, 12), dtype=np.float32)
    val_r = np.zeros((n_users, 256), dtype=np.int16)
    val_m = np.ones((n_users, 256), dtype=bool)
    val_emp = np.zeros(n_users, dtype=bool)

    extract_features_into_buffers(
        df_raw, VAL_ANCHOR, user_ids, val_c, val_t, val_r, val_m, val_emp,
        offset=0, max_events=256, tau_days=30.0
    )

    snap_val = pl.read_parquet(snapshots_dir / f"snapshot_{VAL_ANCHOR}.parquet")
    val_target = snap_val["target"].to_numpy().astype(np.float32)
    val_target_z = np.log1p(np.maximum(0.0, val_target)).astype(np.float32)
    val_was_act = (snap_val["lifetime_gmv"].to_numpy() > 0).astype(np.float32)

    val_dataset = ZeroCopyDataset(
        val_c, val_t, val_r, val_m, val_emp, val_target_z, val_was_act, val_target
    )
    print(f"[+] Validation dataset ready: {len(val_dataset)} samples.")

    # 2. Monitoring 20k Indices
    monitoring_path = Path("artifacts/ett_optimization/monitoring_users_20k.parquet")
    if not monitoring_path.exists():
        monitoring_path = Path("monitoring_users_20k.parquet")
    if not monitoring_path.exists():
        monitoring_path = Path("ett_optimization/monitoring_users_20k.parquet")

    if monitoring_path.exists():
        df_mon = pl.read_parquet(monitoring_path)
        monitoring_indices = df_mon["val_index"].to_numpy()
        print(f"[+] Loaded existing 20k monitoring indices from {monitoring_path}")
    else:
        print("[*] Creating stratified 20k monitoring subset...")
        rng = np.random.RandomState(42)
        y_buy_val = (val_target > 0).astype(int)
        mon_idx_list = []
        for act_state in [0, 1]:
            for buy_state in [0, 1]:
                state_idx = np.where((val_was_act == act_state) & (y_buy_val == buy_state))[0]
                n_sample = int(round(len(state_idx) * 0.20))
                sampled = rng.choice(state_idx, size=n_sample, replace=False)
                mon_idx_list.extend(sampled)
        monitoring_indices = np.array(mon_idx_list, dtype=np.int64)
        out_mon = out_dir / "monitoring_users_20k.parquet"
        pl.DataFrame({"val_index": monitoring_indices, "user_id": user_ids[monitoring_indices]}).write_parquet(out_mon)
        print(f"[+] Created and saved {len(monitoring_indices)} monitoring indices to {out_mon}")

    monitoring_dataset = torch.utils.data.Subset(val_dataset, monitoring_indices)

    # 3. Preallocate and Fill 11 Training Anchors Sequences (1.1M pairs) In-Place
    n_train_samples = len(ANCHORS_V1) * n_users
    print(f"\n[*] Preallocating Contiguous Training Buffers: {n_train_samples} samples (256 tokens)...")
    all_train_c = np.zeros((n_train_samples, 256, 12), dtype=np.float32)
    all_train_t = np.zeros((n_train_samples, 256, 12), dtype=np.float32)
    all_train_r = np.zeros((n_train_samples, 256), dtype=np.int16)
    all_train_m = np.ones((n_train_samples, 256), dtype=bool)
    all_train_emp = np.zeros(n_train_samples, dtype=bool)
    all_train_z = np.zeros(n_train_samples, dtype=np.float32)
    all_train_act = np.zeros(n_train_samples, dtype=np.float32)
    all_train_rub = np.zeros(n_train_samples, dtype=np.float32)

    for k, anchor in enumerate(ANCHORS_V1):
        offset = k * n_users
        print(f"  -> Extracting anchor {anchor} ({k+1}/{len(ANCHORS_V1)}) into preallocated buffer...")
        extract_features_into_buffers(
            df_raw, anchor, user_ids,
            all_train_c, all_train_t, all_train_r, all_train_m, all_train_emp,
            offset=offset, max_events=256, tau_days=30.0
        )
        snap = pl.read_parquet(snapshots_dir / f"snapshot_{anchor}.parquet")
        target = snap["target"].to_numpy().astype(np.float32)
        all_train_rub[offset : offset + n_users] = target
        all_train_z[offset : offset + n_users] = np.log1p(np.maximum(0.0, target))
        all_train_act[offset : offset + n_users] = (snap["lifetime_gmv"].to_numpy() > 0).astype(np.float32)

    train_dataset = ZeroCopyDataset(
        all_train_c, all_train_t, all_train_r, all_train_m, all_train_emp,
        all_train_z, all_train_act, all_train_rub
    )
    print(f"[+] Complete Zero-Copy Training Dataset Ready: {len(train_dataset)} sequences (1.1M pairs, 256 tokens).")

    # 4. Train Model with Gradient Accumulation
    torch.manual_seed(42)
    np.random.seed(42)

    model = EventTimeTransformer(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.10,
        max_events=256,
    ).to(device)

    max_lr = 3e-4
    warmup_steps = 500
    max_steps = 12000
    min_steps = 4000
    patience_steps = 2000
    micro_batch_size = 128
    grad_accum_steps = 2  # Effective batch size = 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True, num_workers=2)
    train_iter = iter(train_loader)

    step_logs = []
    best_monitoring_rmsle = 999.0
    best_monitoring_step = 0
    candidate_checkpoints = []
    steps_without_improvement = 0
    start_time = time.time()

    print(f"\n[*] Training OPT_MAX256 (effective bs=256, max_lr={max_lr:.1e}, max_events=256)...")

    for global_step in range(1, max_steps + 1):
        lr = get_lr_with_warmup(global_step, max_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            content, time_f, ranks, mask, empty, z, was_act, _ = batch
            content = content.to(device)
            time_f = time_f.to(device)
            ranks = ranks.to(device)
            mask = mask.to(device)
            empty = empty.to(device)
            target = z.to(device)
            was_act_dev = was_act.to(device)

            model.train()
            out = model(content, time_f, ranks, mask, empty)
            loss, loss_dict, _ = compute_canonical_loss(out, target, was_act_dev)
            (loss / grad_accum_steps).backward()
            accum_loss += loss.item() / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Step validation
        if global_step % 250 == 0:
            mon_metrics, _, _ = evaluate_dataset(model, monitoring_dataset, batch_size=1024, device=device)
            mon_fact = mon_metrics["rmsle_fact"]
            mon_dir = mon_metrics["rmsle_dir"]

            is_new_best = mon_fact < best_monitoring_rmsle
            if is_new_best:
                best_monitoring_rmsle = mon_fact
                best_monitoring_step = global_step
                steps_without_improvement = 0

                state_dict_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                candidate_checkpoints.append({
                    "step": global_step,
                    "monitoring_rmsle": mon_fact,
                    "state_dict": state_dict_cpu,
                })
                candidate_checkpoints = sorted(candidate_checkpoints, key=lambda x: x["monitoring_rmsle"])[:3]
            else:
                steps_without_improvement += 250

            log_entry = {
                "run": "OPT_MAX256",
                "global_step": global_step,
                "examples_seen": global_step * 256,
                "lr": lr,
                "train_loss": accum_loss,
                "mon_rmsle_fact": mon_fact,
                "mon_rmsle_dir": mon_dir,
                "is_best_mon": is_new_best,
            }

            if global_step % 1000 == 0:
                full_metrics, _, _ = evaluate_dataset(model, val_dataset, batch_size=1024, device=device)
                log_entry.update({
                    "full_rmsle_fact": full_metrics["rmsle_fact"],
                    "full_rmsle_dir": full_metrics["rmsle_dir"],
                    "mse_0_0": full_metrics.get("mse_fact_0->0", 0.0),
                    "mse_0_1": full_metrics.get("mse_fact_0->1", 0.0),
                    "mse_1_0": full_metrics.get("mse_fact_1->0", 0.0),
                    "mse_1_1": full_metrics.get("mse_fact_1->1", 0.0),
                })
                print(
                    f"  Step [{global_step:5d}/{max_steps}] | LR: {lr:.2e} | "
                    f"Loss: {accum_loss:.4f} | "
                    f"Mon Fact RMSLE: {mon_fact:.5f} | "
                    f"FULL Fact RMSLE: {full_metrics['rmsle_fact']:.5f} (Dir: {full_metrics['rmsle_dir']:.5f})"
                )
            else:
                print(
                    f"  Step [{global_step:5d}/{max_steps}] | LR: {lr:.2e} | "
                    f"Loss: {accum_loss:.4f} | "
                    f"Mon Fact RMSLE: {mon_fact:.5f}"
                )

            step_logs.append(log_entry)

            if global_step >= min_steps and steps_without_improvement >= patience_steps:
                print(f"  [!] Early stopping triggered at step {global_step} (patience={patience_steps} reached).")
                break

    elapsed = time.time() - start_time
    print(f"[*] OPT_MAX256 finished training in {elapsed/60:.1f} min. Re-evaluating candidate checkpoints...")

    best_final_checkpoint = None
    best_full_rmsle = 999.0
    best_full_preds_fact = None
    best_full_preds_dir = None
    best_full_metrics = None

    for cand in candidate_checkpoints:
        model.load_state_dict(cand["state_dict"])
        metrics, preds_fact, preds_dir = evaluate_dataset(model, val_dataset, batch_size=1024, device=device)
        print(f"  -> Candidate Step {cand['step']:5d} (Mon: {cand['monitoring_rmsle']:.5f}) -> FULL Fact RMSLE: {metrics['rmsle_fact']:.5f}")
        if metrics["rmsle_fact"] < best_full_rmsle:
            best_full_rmsle = metrics["rmsle_fact"]
            best_final_checkpoint = cand
            best_full_preds_fact = preds_fact
            best_full_preds_dir = preds_dir
            best_full_metrics = metrics

    print(f"\n[+] OPT_MAX256 WINNER CHECKPOINT: Step {best_final_checkpoint['step']} with FULL Fact RMSLE = {best_full_rmsle:.5f}\n")

    torch.save(best_final_checkpoint["state_dict"], out_dir / "best_model.pt")

    df_preds = pl.DataFrame({
        "pred_factorized_z": best_full_preds_fact,
        "pred_direct_z": best_full_preds_dir,
    })
    df_preds.write_parquet(out_dir / "validation_predictions.parquet")

    summary = [{
        "run_name": "OPT_MAX256",
        "max_events": 256,
        "max_lr": max_lr,
        "best_step": best_final_checkpoint["step"],
        "examples_seen": best_final_checkpoint["step"] * 256,
        "best_full_fact_rmsle": best_full_rmsle,
        "best_full_dir_rmsle": best_full_metrics["rmsle_dir"],
        "mse_0_0": best_full_metrics.get("mse_fact_0->0", 0.0),
        "mse_0_1": best_full_metrics.get("mse_fact_0->1", 0.0),
        "mse_1_0": best_full_metrics.get("mse_fact_1->0", 0.0),
        "mse_1_1": best_full_metrics.get("mse_fact_1->1", 0.0),
    }]
    pl.DataFrame(summary).write_csv(out_dir / "max256_summary.csv")
    pl.DataFrame(step_logs).write_csv(out_dir / "max256_training_curves.csv")
    print(f"[+] Saved summary to {out_dir / 'max256_summary.csv'}")


if __name__ == "__main__":
    main()
