"""RUN A — META_250K: Training Specialists up to cutoff 2025-12-15 & Joint RMSLE Meta-Optimization.

1. Cutoff: target_end(anchor) <= 2025-12-15 (17 anchors: 2025-03-31 .. 2025-11-10).
2. Meta-Anchor: 2025-12-15 (target: 2025-12-16 .. 2026-01-14).
3. Models trained from scratch:
   - CatBoost: CB_REACT_META, CB_CHURN_META, CB_AMOUNT_META.
   - S1 Masked GRU Specialists (Base + React, Churn, Amount).
   - S2 Dense GRU Specialists (Base + React, Churn, Amount).
   - ETT Specialists (Base + React, Churn, Amount, 180 tok, tau=30d).
4. Meta-Anchor Inference on ALL 250,000 users.
5. User-level 5-Fold Meta-CV & Joint RMSLE SLSQP Optimization (13 parameters).
6. Exports joint_meta_weights_250k.json and meta_anchor_predictions_250k.parquet.
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.snapshots import build_snapshot


# ==============================================================================
# 1. NEURAL ARCHITECTURES (S1, S2, ETT)
# ==============================================================================

class GRUEncoder(nn.Module):
    def __init__(self, d_in: int = 12, d_model: int = 128, n_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head_dir = nn.Linear(d_model, 1)
        self.head_cond = nn.Linear(d_model, 1)
        self.head_react = nn.Linear(d_model, 1)
        self.head_churn = nn.Linear(d_model, 1)

    def forward(self, content, time_feat, ranks, mask, empty):
        h_proj = F.gelu(self.proj(content))
        out, _ = self.gru(h_proj)
        h = out[:, -1, :]
        h = torch.where(empty.unsqueeze(-1), torch.zeros_like(h), h)
        dir_z = torch.relu(self.head_dir(h)).squeeze(-1)
        cond_z = torch.relu(self.head_cond(h)).squeeze(-1)
        r_logit = self.head_react(h).squeeze(-1)
        c_logit = self.head_churn(h).squeeze(-1)
        return dir_z, cond_z, r_logit, c_logit


class EventTimeTransformer(nn.Module):
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
        self.cont_proj = nn.Linear(n_content_feats, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(n_time_feats, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.rank_emb = nn.Embedding(max_events + 1, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_dir = nn.Linear(d_model, 1)
        self.head_cond = nn.Linear(d_model, 1)
        self.head_react = nn.Linear(d_model, 1)
        self.head_churn = nn.Linear(d_model, 1)

    def forward(self, content, time_feat, ranks, mask, empty):
        c_emb = self.cont_proj(content)
        t_emb = self.time_proj(time_feat)
        r_emb = self.rank_emb(ranks.clamp(0, 180))
        x = self.norm_in(c_emb + t_emb + r_emb)
        
        mask_safe = mask.clone()
        mask_safe[mask.all(dim=-1), 0] = False
        h_seq = self.transformer(x, src_key_padding_mask=mask_safe)
        h = h_seq[:, -1, :]
        h = torch.where(empty.unsqueeze(-1), torch.zeros_like(h), h)
        
        dir_z = torch.relu(self.head_dir(h)).squeeze(-1)
        cond_z = torch.relu(self.head_cond(h)).squeeze(-1)
        r_logit = self.head_react(h).squeeze(-1)
        c_logit = self.head_churn(h).squeeze(-1)
        return dir_z, cond_z, r_logit, c_logit


class MemmapDataset(Dataset):
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
            torch.from_numpy(np.asarray(self.content[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.time_feat[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.ranks[idx], dtype=np.int64)),
            torch.from_numpy(np.asarray(self.mask[idx], dtype=bool)),
            torch.tensor(bool(self.empty[idx]), dtype=torch.bool),
            torch.tensor(float(self.z_true[idx]), dtype=torch.float32),
            torch.tensor(float(self.was_active[idx]), dtype=torch.float32),
            torch.tensor(float(self.will_buy[idx]), dtype=torch.float32),
            torch.tensor(float(self.y_rub[idx]), dtype=torch.float32),
        )


# ==============================================================================
# 2. PROVEN POLARS DAILY SEQUENCE EXTRACTION
# ==============================================================================

def extract_event_time_sequences(
    df_raw: pl.DataFrame,
    user_ids: np.ndarray,
    anchor_str: str,
    max_events: int = 180,
    tau_days: float = 30.0,
    out_c: np.ndarray = None,
    out_t: np.ndarray = None,
    out_r: np.ndarray = None,
    out_m: np.ndarray = None,
    out_emp: np.ndarray = None,
    offset: int = 0,
):
    anchor_dt = datetime.strptime(anchor_str, "%Y-%m-%d").date()
    hist_start = anchor_dt - timedelta(days=364)

    df_filtered = df_raw.filter(
        (pl.col("event_date") >= hist_start) & (pl.col("event_date") <= anchor_dt)
    )

    cont_exprs = [
        pl.col("search").log1p().alias("c_search"),
        pl.col("cat").log1p().alias("c_cat"),
        pl.col("to_cart").log1p().alias("c_to_cart"),
        pl.col("to_ord").log1p().alias("c_to_ord"),
        pl.col("gmv").log1p().alias("c_gmv"),
        pl.col("gmv_search").log1p().alias("c_gmv_search"),
        pl.col("gmv_cat").log1p().alias("c_gmv_cat"),
        pl.col("searches").log1p().alias("c_searches"),
        pl.col("search_to_cart").alias("c_s2c"),
        pl.col("search_to_ord").alias("c_s2o"),
        pl.col("cat_to_cart").alias("c_c2c"),
        pl.col("cat_to_ord").alias("c_c2o"),
    ]

    time_diff_days = (pl.lit(anchor_dt) - pl.col("event_date")).dt.total_days().cast(pl.Float32)
    t_decay = (-time_diff_days / float(tau_days)).exp()
    t_doy = pl.col("event_date").dt.ordinal_day().cast(pl.Float32)
    t_sin = (2.0 * np.pi * t_doy / 365.25).sin()
    t_cos = (2.0 * np.pi * t_doy / 365.25).cos()
    delta_days = (time_diff_days - 15.0).abs()
    season_prox = (-delta_days / 30.0).exp()

    time_exprs = [
        (time_diff_days / 365.0).alias("t_norm"),
        t_decay.alias("t_decay"),
        t_sin.alias("t_sin"),
        t_cos.alias("t_cos"),
        season_prox.alias("t_season_prox"),
        (pl.col("event_date").dt.weekday() / 7.0).alias("t_dow"),
        (time_diff_days <= 7).cast(pl.Float32).alias("t_is_w1"),
        (time_diff_days <= 14).cast(pl.Float32).alias("t_is_w2"),
        (time_diff_days <= 30).cast(pl.Float32).alias("t_is_m1"),
        (time_diff_days <= 60).cast(pl.Float32).alias("t_is_m2"),
        (time_diff_days <= 90).cast(pl.Float32).alias("t_is_q1"),
        (time_diff_days <= 180).cast(pl.Float32).alias("t_is_h1"),
    ]

    df_prepared = df_filtered.select([
        pl.col("user_id"),
        pl.col("event_date"),
        *cont_exprs,
        *time_exprs,
    ]).sort(["user_id", "event_date"])

    user_groups = df_prepared.group_by("user_id", maintain_order=True).agg([
        pl.col("c_search"), pl.col("c_cat"), pl.col("c_to_cart"), pl.col("c_to_ord"),
        pl.col("c_gmv"), pl.col("c_gmv_search"), pl.col("c_gmv_cat"), pl.col("c_searches"),
        pl.col("c_s2c"), pl.col("c_s2o"), pl.col("c_c2c"), pl.col("c_c2o"),
        pl.col("t_norm"), pl.col("t_decay"), pl.col("t_sin"), pl.col("t_cos"),
        pl.col("t_season_prox"), pl.col("t_dow"), pl.col("t_is_w1"), pl.col("t_is_w2"),
        pl.col("t_is_m1"), pl.col("t_is_m2"), pl.col("t_is_q1"), pl.col("t_is_h1"),
    ])

    user_dict = {row["user_id"]: row for row in user_groups.iter_rows(named=True)}
    c_cols = ["c_search", "c_cat", "c_to_cart", "c_to_ord", "c_gmv", "c_gmv_search", "c_gmv_cat", "c_searches", "c_s2c", "c_s2o", "c_c2c", "c_c2o"]
    t_cols = ["t_norm", "t_decay", "t_sin", "t_cos", "t_season_prox", "t_dow", "t_is_w1", "t_is_w2", "t_is_m1", "t_is_m2", "t_is_q1", "t_is_h1"]

    for i, u in enumerate(user_ids):
        idx = offset + i
        if u not in user_dict:
            out_emp[idx] = True
            out_m[idx, :] = True
            continue

        row = user_dict[u]
        n_ev = len(row["c_search"])
        if n_ev == 0:
            out_emp[idx] = True
            out_m[idx, :] = True
            continue

        take = min(n_ev, max_events)
        start_pad = max_events - take

        c_mat = np.column_stack([row[col][-take:] for col in c_cols]).astype(np.float32)
        t_mat = np.column_stack([row[col][-take:] for col in t_cols]).astype(np.float32)

        out_c[idx, start_pad:, :] = c_mat
        out_t[idx, start_pad:, :] = t_mat
        out_r[idx, start_pad:] = np.arange(1, take + 1, dtype=np.int64)
        out_m[idx, start_pad:] = False
        out_emp[idx] = False


# ==============================================================================
# 3. JOINT RMSLE OPTIMIZATION HELPER
# ==============================================================================

def optimize_joint_meta_weights(
    X_react: np.ndarray,
    X_churn: np.ndarray,
    X_amount_scaled: np.ndarray,
    was_active: np.ndarray,
    y_true_rub: np.ndarray,
    init_weights: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    z_target = np.log1p(np.maximum(y_true_rub, 0.0))

    if init_weights is None:
        init_weights = np.array([
            0.25, 0.25, 0.25, 0.25,      # react
            0.25, 0.25, 0.25, 0.25,      # churn
            0.25, 0.25, 0.25, 0.25,      # amount
            0.0                          # intercept
        ], dtype=np.float64)

    def objective(w):
        w_r = w[0:4]
        w_c = w[4:8]
        w_a = w[8:12]
        b_a = w[12]

        p_r = expit(X_react @ w_r)
        p_c = expit(X_churn @ w_c)
        p_buy = np.where(was_active == 0, p_r, 1.0 - p_c)

        cond_z = np.clip(X_amount_scaled @ w_a + b_a, 0.0, None)
        z_pred = np.power(np.clip(p_buy, 0.0, 1.0), 1.1) * cond_z
        z_pred = np.clip(z_pred, 0.0, None)

        mse = np.mean((z_pred - z_target) ** 2)
        return mse

    bounds = [
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0),
        (-1.0, 1.0)
    ]

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w[0:4]) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.sum(w[4:8]) - 1.0},
    ]

    res = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    return res.x, res.fun


# ==============================================================================
# 4. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print("RUN A — META_250K: SPECIALISTS TRAINING & JOINT RMSLE OPTIMIZATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = Path("artifacts/specialized_hurdle_250k/meta_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("artifacts/specialized_hurdle_250k/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load User Cohort (250,000 users)
    df_sample = pl.read_csv("sample_submit.csv")
    all_users = df_sample["user_id"].to_numpy()
    n_users = len(all_users)
    print(f"[+] Loaded 250k user cohort: {n_users:,} unique users.")

    # 2. RUN A Temporal Anchors (17 anchors up to 2025-11-10)
    meta_anchor = "2025-12-15"
    run_a_cb_anchors = [
        "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
        "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
        "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
        "2025-10-27", "2025-11-10"
    ]
    run_a_neural_anchors = [
        "2025-03-31", "2025-04-28", "2025-05-26", "2025-06-23",
        "2025-07-21", "2025-08-18", "2025-09-15", "2025-10-27"
    ]
    print(f"[+] RUN A CatBoost training anchors ({len(run_a_cb_anchors)}): {run_a_cb_anchors[0]} .. {run_a_cb_anchors[-1]}")
    print(f"[+] RUN A Neural training anchors ({len(run_a_neural_anchors)}): {run_a_neural_anchors}")
    print(f"[+] Meta-Anchor for Joint Optimization: {meta_anchor}")

    # 3. Load Raw Events
    events_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    print(f"[*] Loading raw events from {events_path}...")
    df_events = pl.read_parquet(events_path)
    print(f"[+] Loaded raw events: {len(df_events):,} rows.")

    snap_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")

    # 4. STEP A: CATBOOST SPECIALISTS TRAINING ON ALL RUN A ANCHORS
    print("\n" + "=" * 80)
    print("STEP A: TRAINING CATBOOST SPECIALISTS (RUN A)")
    print("=" * 80)
    
    cb_dfs = []
    for anc in run_a_cb_anchors:
        anc_path = snap_dir / f"snapshot_{anc}.parquet"
        if not anc_path.exists():
            anc_path = Path(f"anchor_{anc}.parquet")
        df_snap = pl.read_parquet(anc_path)
        cb_dfs.append(df_snap)

    df_cb_train = pl.concat(cb_dfs)
    print(f"[+] Pooled CatBoost training data for RUN A: {len(df_cb_train):,} rows.")

    excluded = {"user_id", "target", "lifetime_gmv", "will_buy_30d", "anchor_date", "history_start", "history_end", "target_start", "target_end", "user_segment_id"}
    feat_cols = [c for c in df_cb_train.columns if c not in excluded]
    print(f"[*] Tabular features count: {len(feat_cols)}")

    y_train_cb_gmv = df_cb_train["target"].to_numpy().astype(np.float32)
    was_act_tr = (df_cb_train["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_tr = (y_train_cb_gmv > 0).astype(int)
    X_cb_tr = df_cb_train.select(feat_cols).to_numpy().astype(np.float32)

    # CB_REACT
    mask_react = (was_act_tr == 0)
    print(f"\n[*] Training CB_REACT_META on {mask_react.sum():,} rows...")
    cb_react = CatBoostClassifier(iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0, task_type="GPU", random_seed=42, verbose=500)
    cb_react.fit(X_cb_tr[mask_react], will_buy_tr[mask_react])

    # CB_CHURN
    mask_churn = (was_act_tr == 1)
    print(f"\n[*] Training CB_CHURN_META on {mask_churn.sum():,} rows...")
    cb_churn = CatBoostClassifier(iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0, task_type="GPU", random_seed=42, verbose=500)
    cb_churn.fit(X_cb_tr[mask_churn], 1 - will_buy_tr[mask_churn])

    # CB_AMOUNT
    mask_amt = (y_train_cb_gmv > 0)
    print(f"\n[*] Training CB_AMOUNT_META on {mask_amt.sum():,} rows...")
    cb_amount = CatBoostRegressor(iterations=3500, learning_rate=0.035, depth=7, l2_leaf_reg=6.0, loss_function="RMSE", eval_metric="RMSE", task_type="GPU", random_seed=42, verbose=500)
    cb_amount.fit(X_cb_tr[mask_amt], np.log1p(y_train_cb_gmv[mask_amt]))

    del df_cb_train, X_cb_tr, was_act_tr, will_buy_tr, y_train_cb_gmv, cb_dfs
    gc.collect()

    # 5. STEP B: NEURAL SPECIALISTS TRAINING (S1, S2, ETT)
    print("\n" + "=" * 80)
    print("STEP B: TRAINING NEURAL SPECIALISTS (RUN A)")
    print("=" * 80)

    n_neural_anchors = len(run_a_neural_anchors)
    total_seq_samples = n_neural_anchors * 100000

    train_c = np.zeros((total_seq_samples, 180, 12), dtype=np.float32)
    train_t = np.zeros((total_seq_samples, 180, 12), dtype=np.float32)
    train_r = np.zeros((total_seq_samples, 180), dtype=np.int64)
    train_m = np.ones((total_seq_samples, 180), dtype=bool)
    train_emp = np.ones(total_seq_samples, dtype=bool)
    train_z = np.zeros(total_seq_samples, dtype=np.float32)
    train_act = np.zeros(total_seq_samples, dtype=np.float32)
    train_wb = np.zeros(total_seq_samples, dtype=np.float32)
    train_rub = np.zeros(total_seq_samples, dtype=np.float32)

    for i, anc in enumerate(run_a_neural_anchors):
        print(f"[*] Extracting sequences for {anc} ({i+1}/{n_neural_anchors})...")
        anc_path = snap_dir / f"snapshot_{anc}.parquet"
        if not anc_path.exists():
            anc_path = Path(f"anchor_{anc}.parquet")
        df_snap_anc = pl.read_parquet(anc_path)
        snap_users = df_snap_anc["user_id"].to_numpy()
        
        offset = i * 100000
        extract_event_time_sequences(
            df_events, snap_users, anc, max_events=180, tau_days=30.0,
            out_c=train_c, out_t=train_t, out_r=train_r, out_m=train_m, out_emp=train_emp,
            offset=offset
        )
        
        rub = df_snap_anc["target"].to_numpy().astype(np.float32)
        train_rub[offset:offset+100000] = rub
        train_act[offset:offset+100000] = (df_snap_anc["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(np.float32)
        train_wb[offset:offset+100000] = (rub > 0).astype(np.float32)
        train_z[offset:offset+100000] = np.log1p(np.maximum(rub, 0.0))

    dataset = MemmapDataset(train_c, train_t, train_r, train_m, train_emp, train_z, train_act, train_wb, train_rub)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, num_workers=2)
    print(f"[+] Total neural training sequences: {len(dataset):,} samples.")

    # Train S1 Masked GRU
    print("\n[*] Training S1 Masked GRU Specialists...")
    s1_model = GRUEncoder(d_in=12, d_model=128, n_layers=2, dropout=0.10).to(device)
    opt_s1 = torch.optim.AdamW(s1_model.parameters(), lr=3e-4, weight_decay=1e-4)
    s1_iter = iter(loader)
    for step in range(1, 4501):
        try: batch = next(s1_iter)
        except StopIteration: s1_iter = iter(loader); batch = next(s1_iter)
        c, t, r, m, emp, z_t, act_t, wb_t, _ = [b.to(device) for b in batch]
        c_masked = torch.where(m.unsqueeze(-1), torch.zeros_like(c), c)
        dir_z, cond_z, r_log, c_log = s1_model(c_masked, t, r, m, emp)
        
        pos_mask = z_t > 0
        inact_mask = (act_t == 0)
        act_mask = (act_t == 1)
        p_buy = torch.where(inact_mask, torch.sigmoid(r_log), 1.0 - torch.sigmoid(c_log))
        fact_z = p_buy * cond_z
        
        loss = F.mse_loss(fact_z, z_t) + 0.25 * F.mse_loss(dir_z, z_t)
        if pos_mask.any(): loss += 0.25 * F.mse_loss(cond_z[pos_mask], z_t[pos_mask])
        if inact_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(r_log[inact_mask], wb_t[inact_mask])
        if act_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(c_log[act_mask], 1.0 - wb_t[act_mask])
            
        opt_s1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1_model.parameters(), 1.0); opt_s1.step()

    # Train S2 Dense GRU
    print("\n[*] Training S2 Dense GRU Specialists...")
    s2_model = GRUEncoder(d_in=12, d_model=128, n_layers=2, dropout=0.10).to(device)
    opt_s2 = torch.optim.AdamW(s2_model.parameters(), lr=3e-4, weight_decay=1e-4)
    s2_iter = iter(loader)
    for step in range(1, 4501):
        try: batch = next(s2_iter)
        except StopIteration: s2_iter = iter(loader); batch = next(s2_iter)
        c, t, r, m, emp, z_t, act_t, wb_t, _ = [b.to(device) for b in batch]
        dir_z, cond_z, r_log, c_log = s2_model(c, t, r, m, emp)
        
        pos_mask = z_t > 0
        inact_mask = (act_t == 0)
        act_mask = (act_t == 1)
        p_buy = torch.where(inact_mask, torch.sigmoid(r_log), 1.0 - torch.sigmoid(c_log))
        fact_z = p_buy * cond_z
        
        loss = F.mse_loss(fact_z, z_t) + 0.25 * F.mse_loss(dir_z, z_t)
        if pos_mask.any(): loss += 0.25 * F.mse_loss(cond_z[pos_mask], z_t[pos_mask])
        if inact_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(r_log[inact_mask], wb_t[inact_mask])
        if act_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(c_log[act_mask], 1.0 - wb_t[act_mask])
            
        opt_s2.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s2_model.parameters(), 1.0); opt_s2.step()

    # Train EventTimeTransformer
    print("\n[*] Training Event-Time Transformer Specialists (180 tok, tau=30d)...")
    ett_model = EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, max_events=180).to(device)
    opt_ett = torch.optim.AdamW(ett_model.parameters(), lr=3e-4, weight_decay=1e-4)
    ett_iter = iter(loader)
    for step in range(1, 4501):
        try: batch = next(ett_iter)
        except StopIteration: ett_iter = iter(loader); batch = next(ett_iter)
        c, t, r, m, emp, z_t, act_t, wb_t, _ = [b.to(device) for b in batch]
        dir_z, cond_z, r_log, c_log = ett_model(c, t, r, m, emp)
        
        pos_mask = z_t > 0
        inact_mask = (act_t == 0)
        act_mask = (act_t == 1)
        p_buy = torch.where(inact_mask, torch.sigmoid(r_log), 1.0 - torch.sigmoid(c_log))
        fact_z = p_buy * cond_z
        
        loss = F.mse_loss(fact_z, z_t) + 0.25 * F.mse_loss(dir_z, z_t)
        if pos_mask.any(): loss += 0.25 * F.mse_loss(cond_z[pos_mask], z_t[pos_mask])
        if inact_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(r_log[inact_mask], wb_t[inact_mask])
        if act_mask.any(): loss += 0.10 * F.binary_cross_entropy_with_logits(c_log[act_mask], 1.0 - wb_t[act_mask])
            
        opt_ett.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ett_model.parameters(), 1.0); opt_ett.step()

    del train_c, train_t, train_r, train_m, train_emp, train_act, train_wb, train_rub, train_z, loader, dataset
    gc.collect()

    # 6. STEP C: META-ANCHOR (2025-12-15) PREDICTIONS ON ALL 250,000 USERS
    print("\n" + "=" * 80)
    print("STEP C: INFERENCE ON META-ANCHOR 2025-12-15 (ALL 250,000 USERS)")
    print("=" * 80)

    # CatBoost Inference on Meta-Anchor
    print(f"[*] Building 250k Meta-Anchor feature snapshot for {meta_anchor}...")
    df_meta_snap = build_snapshot(
        data=df_events,
        user_ids=all_users.tolist(),
        anchor_date=pd.Timestamp(meta_anchor).date(),
        is_test=False,
    )
    X_meta = df_meta_snap.select(feat_cols).to_numpy().astype(np.float32)
    was_act_meta = (df_meta_snap["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    gmv_meta = df_meta_snap["target"].to_numpy().astype(np.float32)
    will_buy_meta = (gmv_meta > 0).astype(int)

    cb_r_logit = cb_react.predict(X_meta, prediction_type="RawFormulaVal")
    cb_c_logit = cb_churn.predict(X_meta, prediction_type="RawFormulaVal")
    cb_a_z = cb_amount.predict(X_meta)

    # Neural Inference on Meta-Anchor
    print(f"[*] Extracting Meta-Anchor sequences for {meta_anchor} (250,000 users)...")
    m_c = np.zeros((n_users, 180, 12), dtype=np.float32)
    m_t = np.zeros((n_users, 180, 12), dtype=np.float32)
    m_r = np.zeros((n_users, 180), dtype=np.int64)
    m_m = np.ones((n_users, 180), dtype=bool)
    m_emp = np.ones(n_users, dtype=bool)

    extract_event_time_sequences(
        df_events, all_users, meta_anchor, max_events=180, tau_days=30.0,
        out_c=m_c, out_t=m_t, out_r=m_r, out_m=m_m, out_emp=m_emp, offset=0
    )

    s1_r_list, s1_c_list, s1_a_list = [], [], []
    s2_r_list, s2_c_list, s2_a_list = [], [], []
    ett_r_list, ett_c_list, ett_a_list = [], [], []

    bs = 1024
    s1_model.eval(); s2_model.eval(); ett_model.eval()

    with torch.no_grad():
        for i in range(0, n_users, bs):
            end_i = min(i + bs, n_users)
            c_b = torch.from_numpy(m_c[i:end_i]).to(device)
            t_b = torch.from_numpy(m_t[i:end_i]).to(device)
            r_b = torch.from_numpy(m_r[i:end_i]).to(device)
            m_b = torch.from_numpy(m_m[i:end_i]).to(device)
            emp_b = torch.from_numpy(m_emp[i:end_i]).to(device)

            c_b_masked = torch.where(m_b.unsqueeze(-1), torch.zeros_like(c_b), c_b)
            _, s1_a, s1_r, s1_c = s1_model(c_b_masked, t_b, r_b, m_b, emp_b)
            s1_r_list.append(s1_r.cpu().numpy())
            s1_c_list.append(s1_c.cpu().numpy())
            s1_a_list.append(s1_a.cpu().numpy())

            _, s2_a, s2_r, s2_c = s2_model(c_b, t_b, r_b, m_b, emp_b)
            s2_r_list.append(s2_r.cpu().numpy())
            s2_c_list.append(s2_c.cpu().numpy())
            s2_a_list.append(s2_a.cpu().numpy())

            _, ett_a, ett_r, ett_c = ett_model(c_b, t_b, r_b, m_b, emp_b)
            ett_r_list.append(ett_r.cpu().numpy())
            ett_c_list.append(ett_c.cpu().numpy())
            ett_a_list.append(ett_a.cpu().numpy())

    s1_r_logit = np.concatenate(s1_r_list)
    s1_c_logit = np.concatenate(s1_c_list)
    s1_a_z = np.concatenate(s1_a_list)

    s2_r_logit = np.concatenate(s2_r_list)
    s2_c_logit = np.concatenate(s2_c_list)
    s2_a_z = np.concatenate(s2_a_list)

    ett_r_logit = np.concatenate(ett_r_list)
    ett_c_logit = np.concatenate(ett_c_list)
    ett_a_z = np.concatenate(ett_a_list)

    # Save Meta-Anchor Raw Predictions Table
    df_meta_preds = pl.DataFrame({
        "user_id": all_users,
        "anchor": [meta_anchor] * n_users,
        "was_active": was_act_meta,
        "will_buy": will_buy_meta,
        "future_gmv_30d": gmv_meta,
        "cb_react_logit": cb_r_logit,
        "s1_react_logit": s1_r_logit,
        "s2_react_logit": s2_r_logit,
        "ett_react_logit": ett_r_logit,
        "cb_churn_logit": cb_c_logit,
        "s1_churn_logit": s1_c_logit,
        "s2_churn_logit": s2_c_logit,
        "ett_churn_logit": ett_c_logit,
        "cb_amount_z": cb_a_z,
        "s1_amount_z": s1_a_z,
        "s2_amount_z": s2_a_z,
        "ett_amount_z": ett_a_z,
    })

    meta_preds_path = out_dir / "meta_anchor_predictions_250k.parquet"
    df_meta_preds.write_parquet(meta_preds_path)
    df_meta_preds.write_parquet(Path("meta_anchor_predictions_250k.parquet"))
    print(f"[+] Saved Meta-Anchor predictions table to {meta_preds_path} and root ({len(df_meta_preds):,} rows)")

    # 7. STEP D: JOINT RMSLE META-OPTIMIZATION (13 PARAMETERS)
    print("\n" + "=" * 80)
    print("STEP D: 5-FOLD META-CV & JOINT RMSLE OPTIMIZATION (250k USERS)")
    print("=" * 80)

    X_r = np.column_stack([cb_r_logit, s1_r_logit, s2_r_logit, ett_r_logit])
    X_c = np.column_stack([cb_c_logit, s1_c_logit, s2_c_logit, ett_c_logit])
    X_a = np.column_stack([cb_a_z, s1_a_z, s2_a_z, ett_a_z])

    # Fit new Amount Scaler on Meta-Anchor
    amount_means = np.mean(X_a, axis=0).tolist()
    amount_scales = np.std(X_a, axis=0).tolist()
    scaler_cfg = {
        "feature_order": ["cb_amount_z", "s1_amount_z", "s2_amount_z", "ett_amount_z"],
        "means": amount_means,
        "scales": amount_scales,
        "created_at": datetime.now().isoformat()
    }
    with open(out_dir / "amount_scaler_250k.json", "w", encoding="utf-8") as f:
        json.dump(scaler_cfg, f, indent=2)
    with open("amount_scaler_250k.json", "w", encoding="utf-8") as f:
        json.dump(scaler_cfg, f, indent=2)

    X_a_scaled = (X_a - np.array(amount_means)) / np.array(amount_scales)

    # 5-Fold User-Level Meta-CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(all_users), 1):
        w_f, loss_trn = optimize_joint_meta_weights(
            X_r[trn_idx], X_c[trn_idx], X_a_scaled[trn_idx],
            was_act_meta[trn_idx], gmv_meta[trn_idx]
        )
        
        p_r_val = expit(X_r[val_idx] @ w_f[0:4])
        p_c_val = expit(X_c[val_idx] @ w_f[4:8])
        p_buy_val = np.where(was_act_meta[val_idx] == 0, p_r_val, 1.0 - p_c_val)
        cond_z_val = np.clip(X_a_scaled[val_idx] @ w_f[8:12] + w_f[12], 0.0, None)
        z_pred_val = np.power(np.clip(p_buy_val, 0.0, 1.0), 1.1) * cond_z_val
        z_tgt_val = np.log1p(np.maximum(gmv_meta[val_idx], 0.0))
        
        val_mse = np.mean((z_pred_val - z_tgt_val) ** 2)
        val_rmsle = np.sqrt(val_mse)
        
        cv_records.append({
            "fold": fold,
            "val_mse": float(val_mse),
            "val_rmsle": float(val_rmsle),
            "weights": [float(x) for x in w_f]
        })
        print(f"  Fold {fold}: Val MSE = {val_mse:.6f} | Val RMSLE = {val_rmsle:.6f}")

    mean_cv_rmsle = np.mean([r["val_rmsle"] for r in cv_records])
    print(f"\n[+] 5-Fold User-Level Meta-CV Mean RMSLE: {mean_cv_rmsle:.6f}")

    # Fit on all 250k meta users
    final_w, final_loss = optimize_joint_meta_weights(
        X_r, X_c, X_a_scaled, was_act_meta, gmv_meta
    )
    final_rmsle = np.sqrt(final_loss)
    print(f"\n[+] Final Joint Optimization on ALL 250k users:")
    print(f"    Meta MSE:   {final_loss:.6f}")
    print(f"    Meta RMSLE: {final_rmsle:.6f}")

    meta_package = {
        "experiment_name": "SPECIALIZED_HURDLE_JOINT_250K_V2",
        "created_at": datetime.now().isoformat(),
        "meta_anchor": meta_anchor,
        "cohort_users": n_users,
        "meta_mse": float(final_loss),
        "meta_rmsle": float(final_rmsle),
        "cv_5fold_mean_rmsle": float(mean_cv_rmsle),
        "alpha_fixed": 1.1,
        "react_stack_weights": [float(x) for x in final_w[0:4]],
        "churn_stack_weights": [float(x) for x in final_w[4:8]],
        "amount_ridge_coefficients": [float(x) for x in final_w[8:12]],
        "amount_ridge_intercept": float(final_w[12]),
        "amount_scaler": scaler_cfg,
        "model_order": ["CatBoost", "S1_Masked_GRU", "S2_Dense_GRU", "Event_Time_Transformer"]
    }

    meta_weights_path = out_dir / "joint_meta_weights_250k.json"
    with open(meta_weights_path, "w", encoding="utf-8") as f:
        json.dump(meta_package, f, indent=2)
    with open("joint_meta_weights_250k.json", "w", encoding="utf-8") as f:
        json.dump(meta_package, f, indent=2)
    print(f"[+] Saved final Joint Meta Package to {meta_weights_path} and root")

    # Update run_status.json
    status_path = logs_dir / "run_status.json"
    status_data = {
        "experiment_name": "SPECIALIZED_HURDLE_JOINT_250K_V2",
        "updated_at": datetime.now().isoformat(),
        "current_stage": "STAGE_2_RUN_A_COMPLETED",
        "run_a_results": {
            "meta_anchor": meta_anchor,
            "meta_mse": float(final_loss),
            "meta_rmsle": float(final_rmsle),
            "cv_5fold_mean_rmsle": float(mean_cv_rmsle),
            "joint_weights": meta_package
        }
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    with open("run_status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)

    print("\n[+] RUN A — META_250K COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
