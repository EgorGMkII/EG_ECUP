"""RUN 1: Complete Meta-Weights Training on 100k Users.

Trains from scratch:
1. CatBoost Specialists: CB_REACT, CB_CHURN, CB_AMOUNT on 17 pooled training anchors.
2. S1 Masked GRU Specialists: S1_BASE + React, Churn, Amount heads.
3. S2 Dense GRU Specialists: S2_BASE + React, Churn, Amount heads.
4. Event-Time Transformer Specialists: ETT_BASE (180 tokens, fixed tau=30d) + React, Churn, Amount heads.
5. Generates real specialist predictions on meta-anchor 2025-12-15.
6. Fits and fixes meta-weights (React Stack, Churn Stack, Amount Ridge) with ALPHA = 1.1.
7. Saves artifacts/run1_meta_weights.json.
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

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CANONICAL_ANCHORS = [
    "2025-03-31", "2025-04-14", "2025-04-28", "2025-05-12", "2025-05-26",
    "2025-06-09", "2025-06-23", "2025-07-07", "2025-07-21", "2025-08-04",
    "2025-08-18", "2025-09-01", "2025-09-15", "2025-09-29", "2025-10-13",
    "2025-10-27", "2025-11-10", "2025-11-24", "2025-12-08", "2025-12-15",
    "2025-12-22", "2026-01-05", "2026-01-14"
]


# ----------------------------------------------------------------------
# 1. Dataset Definitions & Sequence Extraction
# ----------------------------------------------------------------------

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
    midpoint_doy = (anchor_dt + timedelta(days=15)).timetuple().tm_yday

    df_filtered = df_raw.filter(
        (pl.col("event_date") >= hist_start) & (pl.col("event_date") <= anchor_dt)
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
            age_days = float((anchor_dt - ev_date).days)
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


# ----------------------------------------------------------------------
# 2. Neural Models (Event-Time Transformer & GRU)
# ----------------------------------------------------------------------

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

        # Multitask Heads
        self.direct_head = nn.Linear(d_model, 1)
        self.conditional_head = nn.Linear(d_model, 1)
        self.reactivation_head = nn.Linear(d_model, 1)
        self.churn_head = nn.Linear(d_model, 1)

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

    def forward(self, content, time_feat, ranks, mask, empty):
        emb = self.extract_embedding(content, time_feat, ranks, mask, empty)
        direct_z = F.softplus(self.direct_head(emb)).squeeze(-1)
        conditional_z = F.softplus(self.conditional_head(emb)).squeeze(-1)
        reactivation_logit = self.reactivation_head(emb).squeeze(-1)
        churn_logit = self.churn_head(emb).squeeze(-1)
        return direct_z, conditional_z, reactivation_logit, churn_logit


class GRUEncoder(nn.Module):
    def __init__(self, d_in: int = 12, d_model: int = 128, n_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direct_head = nn.Linear(d_model, 1)
        self.conditional_head = nn.Linear(d_model, 1)
        self.reactivation_head = nn.Linear(d_model, 1)
        self.churn_head = nn.Linear(d_model, 1)

    def extract_embedding(self, content, time_feat, ranks, mask, empty):
        b, s, _ = content.shape
        x = self.proj(content)
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        emb = self.pooling(last_out)
        return emb

    def forward(self, content, time_feat, ranks, mask, empty):
        emb = self.extract_embedding(content, time_feat, ranks, mask, empty)
        direct_z = F.softplus(self.direct_head(emb)).squeeze(-1)
        conditional_z = F.softplus(self.conditional_head(emb)).squeeze(-1)
        reactivation_logit = self.reactivation_head(emb).squeeze(-1)
        churn_logit = self.churn_head(emb).squeeze(-1)
        return direct_z, conditional_z, reactivation_logit, churn_logit


# ----------------------------------------------------------------------
# 3. Main Execution Function
# ----------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RUN 1: TRAIN META-WEIGHTS & SPECIALIST MODELS (100k COHORT)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    out_dir = Path("artifacts/specialized_hurdle")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 100k cohort users
    users_path = Path("artifacts/selected_users_100k.parquet")
    if not users_path.exists():
        users_path = Path("selected_users_100k.parquet")
    users_100k = pl.read_parquet(users_path)["user_id"].to_numpy()
    n_users = len(users_100k)
    print(f"[+] Loaded {n_users:,} canonical cohort users.")

    # 2. Determine Training Anchors (target_end <= 2025-12-15)
    all_anchors = CANONICAL_ANCHORS

    cutoff = pd.Timestamp("2025-12-15")
    train_anchors = []
    for a in all_anchors:
        a_dt = pd.Timestamp(a)
        t_end = a_dt + pd.Timedelta(days=30)
        if t_end <= cutoff:
            train_anchors.append(a)

    meta_anchor = "2025-12-15"
    print(f"[+] Found {len(train_anchors)} legal training anchors: {train_anchors[0]} .. {train_anchors[-1]}")
    print(f"[+] Meta-Anchor for Stack Fitting: {meta_anchor}")

    # ------------------------------------------------------------------
    # Step A: Train CatBoost Specialists on Feature Store
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP A: PREPARING FEATURE STORE & TRAINING CATBOOST SPECIALISTS")
    print("=" * 80)

    fs_dir = Path("artifacts/specialized_hurdle/feature_store")
    fs_dir.mkdir(parents=True, exist_ok=True)

    needed_anchors = train_anchors + [meta_anchor]
    missing = [a for a in needed_anchors if not (fs_dir / f"anchor_{a}.parquet").exists()]
    if missing:
        print(f"[*] Building causal feature store on-the-fly on VM for {len(missing)} anchors...")
        from src.specialized_hurdle.feature_store import build_causal_feature_store
        snap_dir = Path("data/snapshots") if Path("data/snapshots").exists() else Path("snapshots")
        build_causal_feature_store(
            snapshots_dir=snap_dir,
            out_dir=fs_dir,
            anchors=needed_anchors,
            user_ids=users_100k,
        )

    train_dfs = [pl.read_parquet(fs_dir / f"anchor_{a}.parquet") for a in train_anchors if (fs_dir / f"anchor_{a}.parquet").exists()]
    df_cb_train = pl.concat(train_dfs)
    df_cb_meta = pl.read_parquet(fs_dir / f"anchor_{meta_anchor}.parquet")

    excluded = {"user_id", "target", "lifetime_gmv", "will_buy_30d"}
    feat_cols = [c for c in df_cb_train.columns if c not in excluded]

    X_train_cb = df_cb_train.select(feat_cols).to_numpy().astype(np.float32)
    y_train_cb_gmv = df_cb_train["target"].to_numpy().astype(np.float32)
    was_act_train_cb = (df_cb_train["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_train_cb = (y_train_cb_gmv > 0).astype(int)

    X_meta_cb = df_cb_meta.select(feat_cols).to_numpy().astype(np.float32)
    y_meta_cb_gmv = df_cb_meta["target"].to_numpy().astype(np.float32)
    was_act_meta_cb = (df_cb_meta["lifetime_gmv"].to_numpy().astype(np.float32) > 0).astype(int)
    will_buy_meta_cb = (y_meta_cb_gmv > 0).astype(int)

    print(f"[*] Training CatBoost Specialists on {len(df_cb_train):,} pooled rows x {len(feat_cols)} features...")

    # CB_REACT
    m_react_tr = was_act_train_cb == 0
    m_react_va = was_act_meta_cb == 0
    cb_react = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_react.fit(X_train_cb[m_react_tr], will_buy_train_cb[m_react_tr], eval_set=(X_meta_cb[m_react_va], will_buy_meta_cb[m_react_va]), early_stopping_rounds=80, verbose=False)
    cb_react_logits_meta = cb_react.predict(X_meta_cb, prediction_type="RawFormulaVal")
    print(f"   [+] CB_REACT trained. Meta React AUC: {cb_react.get_best_score().get('validation', {}).get('Logloss', 0.0):.4f}")

    # CB_CHURN
    m_churn_tr = was_act_train_cb == 1
    m_churn_va = was_act_meta_cb == 1
    cb_churn = CatBoostClassifier(iterations=1200, learning_rate=0.04, depth=6, loss_function="Logloss", random_seed=42, verbose=False)
    cb_churn.fit(X_train_cb[m_churn_tr], (1 - will_buy_train_cb[m_churn_tr]), eval_set=(X_meta_cb[m_churn_va], (1 - will_buy_meta_cb[m_churn_va])), early_stopping_rounds=80, verbose=False)
    cb_churn_logits_meta = cb_churn.predict(X_meta_cb, prediction_type="RawFormulaVal")
    print(f"   [+] CB_CHURN trained. Meta Churn Logloss: {cb_churn.get_best_score().get('validation', {}).get('Logloss', 0.0):.4f}")

    # CB_AMOUNT
    m_amt_tr = y_train_cb_gmv > 0
    m_amt_va = y_meta_cb_gmv > 0
    cb_amount = CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=42, verbose=False)
    cb_amount.fit(X_train_cb[m_amt_tr], np.log1p(y_train_cb_gmv[m_amt_tr]), eval_set=(X_meta_cb[m_amt_va], np.log1p(y_meta_cb_gmv[m_amt_va])), early_stopping_rounds=80, verbose=False)
    cb_amount_z_meta = np.maximum(0.0, cb_amount.predict(X_meta_cb))
    print(f"   [+] CB_AMOUNT trained. Meta Amount RMSE: {cb_amount.get_best_score().get('validation', {}).get('RMSE', 0.0):.4f}")

    del df_cb_train, X_train_cb
    gc.collect()

    # ------------------------------------------------------------------
    # Step B: Extract Sequential Data & Train Neural Models
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP B: EXTRACTING EVENT SEQUENCES & TRAINING NEURAL SPECIALISTS")
    print("=" * 80)

    raw_path = Path("data/train.parquet") if Path("data/train.parquet").exists() else Path("train.parquet")
    df_raw = pl.read_parquet(raw_path) if raw_path.exists() else pl.DataFrame()

    max_ev = 180
    n_train_samples = len(train_anchors) * n_users

    train_c = np.zeros((n_train_samples, max_ev, 12), dtype=np.float32)
    train_t = np.zeros((n_train_samples, max_ev, 12), dtype=np.float32)
    train_r = np.zeros((n_train_samples, max_ev), dtype=np.int16)
    train_m = np.ones((n_train_samples, max_ev), dtype=bool)
    train_emp = np.zeros(n_train_samples, dtype=bool)
    train_z = np.zeros(n_train_samples, dtype=np.float32)
    train_act = np.zeros(n_train_samples, dtype=np.float32)
    train_buy = np.zeros(n_train_samples, dtype=np.float32)
    train_rub = np.zeros(n_train_samples, dtype=np.float32)

    print(f"[*] Preallocating training buffers ({n_train_samples:,} samples)...")
    for a_idx, a_str in enumerate(train_anchors):
        off = a_idx * n_users
        snap_a = pl.read_parquet(fs_dir / f"anchor_{a_str}.parquet")
        u_map = {u: i for i, u in enumerate(snap_a["user_id"].to_list())}
        order = [u_map[u] for u in users_100k if u in u_map]

        y_r = snap_a["target"].to_numpy()[order].astype(np.float32)
        past_g = snap_a["lifetime_gmv"].to_numpy()[order].astype(np.float32)

        train_z[off : off + n_users] = np.log1p(np.maximum(0.0, y_r))
        train_act[off : off + n_users] = (past_g > 0).astype(np.float32)
        train_buy[off : off + n_users] = (y_r > 0).astype(np.float32)
        train_rub[off : off + n_users] = y_r

        extract_event_time_sequences(
            df_raw, users_100k, a_str, max_events=max_ev,
            out_c=train_c, out_t=train_t, out_r=train_r, out_m=train_m, out_emp=train_emp,
            offset=off
        )
        print(f"   -> Extracted training anchor {a_str} ({a_idx+1}/{len(train_anchors)})")

    # Meta-anchor buffers
    meta_c = np.zeros((n_users, max_ev, 12), dtype=np.float32)
    meta_t = np.zeros((n_users, max_ev, 12), dtype=np.float32)
    meta_r = np.zeros((n_users, max_ev), dtype=np.int16)
    meta_m = np.ones((n_users, max_ev), dtype=bool)
    meta_emp = np.zeros(n_users, dtype=bool)
    meta_z = np.log1p(np.maximum(0.0, y_meta_cb_gmv)).astype(np.float32)
    meta_act = was_act_meta_cb.astype(np.float32)
    meta_buy = will_buy_meta_cb.astype(np.float32)
    meta_rub = y_meta_cb_gmv.astype(np.float32)

    extract_event_time_sequences(
        df_raw, users_100k, meta_anchor, max_events=max_ev,
        out_c=meta_c, out_t=meta_t, out_r=meta_r, out_m=meta_m, out_emp=meta_emp,
        offset=0
    )

    del df_raw
    gc.collect()

    train_ds = ZeroCopyDataset(train_c, train_t, train_r, train_m, train_emp, train_z, train_act, train_buy, train_rub)
    meta_ds = ZeroCopyDataset(meta_c, meta_t, meta_r, meta_m, meta_emp, meta_z, meta_act, meta_buy, meta_rub)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
    meta_loader = DataLoader(meta_ds, batch_size=512, shuffle=False, pin_memory=torch.cuda.is_available())

    # Training helper
    def train_neural_model(model: nn.Module, n_steps: int = 3500, lr: float = 3e-4):
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        model.train()
        step = 0
        train_iter = iter(train_loader)

        while step < n_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            c, t, r, m, emp, z_t, act, buy, _ = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            optimizer.zero_grad()

            dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)

            p_react = torch.sigmoid(r_logit)
            p_churn = torch.sigmoid(c_logit)
            p_buy = torch.where(act > 0.5, 1.0 - p_churn, p_react)
            fact_z = p_buy * cond_z

            l_fact = F.mse_loss(fact_z, z_t)
            l_dir = F.mse_loss(dir_z, z_t)

            pos_mask = buy > 0.5
            l_cond = F.mse_loss(cond_z[pos_mask], z_t[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=device)

            inact_mask = act <= 0.5
            l_react = F.binary_cross_entropy_with_logits(r_logit[inact_mask], buy[inact_mask]) if inact_mask.sum() > 0 else torch.tensor(0.0, device=device)

            act_mask = act > 0.5
            l_churn = F.binary_cross_entropy_with_logits(c_logit[act_mask], 1.0 - buy[act_mask]) if act_mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = 1.00 * l_fact + 0.25 * l_dir + 0.25 * l_cond + 0.10 * l_react + 0.10 * l_churn
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

        # Predict on meta anchor
        model.eval()
        r_logits_list, c_logits_list, cond_z_list = [], [], []
        with torch.no_grad():
            for batch in meta_loader:
                c, t, r, m, emp, _, _, _, _ = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                dir_z, cond_z, r_logit, c_logit = model(c, t, r, m, emp)
                r_logits_list.append(r_logit.cpu().numpy())
                c_logits_list.append(c_logit.cpu().numpy())
                cond_z_list.append(cond_z.cpu().numpy())

        return (
            np.concatenate(r_logits_list),
            np.concatenate(c_logits_list),
            np.concatenate(cond_z_list),
        )

    # Train S1
    print("\n[*] Training S1 Masked GRU Specialists...")
    s1_model = GRUEncoder(d_in=12, d_model=128, n_layers=2)
    s1_r_logit, s1_c_logit, s1_cond_z = train_neural_model(s1_model, n_steps=2500, lr=5e-4)

    # Train S2
    print("\n[*] Training S2 Dense GRU Specialists...")
    s2_model = GRUEncoder(d_in=12, d_model=128, n_layers=2)
    s2_r_logit, s2_c_logit, s2_cond_z = train_neural_model(s2_model, n_steps=2500, lr=5e-4)

    # Train ETT
    print("\n[*] Training Event-Time Transformer Specialists (180 tokens, fixed tau=30d)...")
    ett_model = EventTimeTransformer(d_model=128, n_heads=4, n_layers=2, max_events=max_ev)
    ett_r_logit, ett_c_logit, ett_cond_z = train_neural_model(ett_model, n_steps=3500, lr=3e-4)

    # Save specialists predictions on Meta-Anchor
    df_meta_preds = pl.DataFrame({
        "user_id": users_100k,
        "was_active": was_act_meta_cb,
        "will_buy": will_buy_meta_cb,
        "future_gmv_30d": y_meta_cb_gmv,
        "cb_react_logit": cb_react_logits_meta,
        "s1_react_logit": s1_r_logit,
        "s2_react_logit": s2_r_logit,
        "ett_react_logit": ett_r_logit,
        "cb_churn_logit": cb_churn_logits_meta,
        "s1_churn_logit": s1_c_logit,
        "s2_churn_logit": s2_c_logit,
        "ett_churn_logit": ett_c_logit,
        "cb_amount_z": cb_amount_z_meta,
        "s1_amount_z": s1_cond_z,
        "s2_amount_z": s2_cond_z,
        "ett_amount_z": ett_cond_z,
    })
    df_meta_preds.write_parquet(out_dir / "run1_meta_predictions.parquet")
    print(f"\n[+] Saved real specialist predictions to {out_dir / 'run1_meta_predictions.parquet'}")

    # ------------------------------------------------------------------
    # Step C: Fit Meta-Weights on Meta-Anchor
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP C: FITTING META-WEIGHTS (REACT, CHURN, AMOUNT RIDGE)")
    print("=" * 80)

    inact_mask = was_act_meta_cb == 0
    act_mask = was_act_meta_cb == 1
    pos_mask = y_meta_cb_gmv > 0

    # React Stack Optimization
    X_react = np.column_stack([cb_react_logits_meta[inact_mask], s1_r_logit[inact_mask], s2_r_logit[inact_mask], ett_r_logit[inact_mask]])
    y_react = will_buy_meta_cb[inact_mask]

    def react_loss(w):
        w_norm = np.maximum(0.0, w) / max(1e-6, np.sum(np.maximum(0.0, w)))
        comb = np.dot(X_react, w_norm)
        p = expit(comb)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -np.mean(y_react * np.log(p) + (1.0 - y_react) * np.log(1.0 - p))

    res_react = minimize(react_loss, [0.25, 0.25, 0.25, 0.25], bounds=[(0, 1)] * 4)
    w_react = np.maximum(0.0, res_react.x) / np.sum(np.maximum(0.0, res_react.x))
    print(f"[+] React Stack Weights (CB, S1, S2, ETT): {w_react.round(4).tolist()}")

    # Churn Stack Optimization
    X_churn = np.column_stack([cb_churn_logits_meta[act_mask], s1_c_logit[act_mask], s2_c_logit[act_mask], ett_c_logit[act_mask]])
    y_churn = (1 - will_buy_meta_cb[act_mask])

    def churn_loss(w):
        w_norm = np.maximum(0.0, w) / max(1e-6, np.sum(np.maximum(0.0, w)))
        comb = np.dot(X_churn, w_norm)
        p = expit(comb)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -np.mean(y_churn * np.log(p) + (1.0 - y_churn) * np.log(1.0 - p))

    res_churn = minimize(churn_loss, [0.25, 0.25, 0.25, 0.25], bounds=[(0, 1)] * 4)
    w_churn = np.maximum(0.0, res_churn.x) / np.sum(np.maximum(0.0, res_churn.x))
    print(f"[+] Churn Stack Weights (CB, S1, S2, ETT): {w_churn.round(4).tolist()}")

    # Amount Ridge Stack
    X_amount = np.column_stack([cb_amount_z_meta[pos_mask], s1_cond_z[pos_mask], s2_cond_z[pos_mask], ett_cond_z[pos_mask]])
    y_amount = np.log1p(y_meta_cb_gmv[pos_mask])

    ridge = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    ridge.fit(X_amount, y_amount)
    print(f"[+] Amount Ridge Coefficients: {ridge.coef_.round(4).tolist()} | Intercept: {ridge.intercept_:.4f}")

    # Final Meta-Weights Structure
    meta_weights = {
        "model_order": ["CatBoost", "S1_GRU", "S2_GRU", "ETT"],
        "react_stack_weights": w_react.tolist(),
        "churn_stack_weights": w_churn.tolist(),
        "amount_ridge_coefficients": ridge.coef_.tolist(),
        "amount_ridge_intercept": float(ridge.intercept_),
        "ALPHA": 1.1,
        "meta_anchor": meta_anchor,
        "train_anchors_count": len(train_anchors),
        "created_at": datetime.now().isoformat(),
    }

    with open(out_dir / "run1_meta_weights.json", "w", encoding="utf-8") as f:
        json.dump(meta_weights, f, indent=2)
    print(f"\n[+] FIXED META-WEIGHTS SAVED TO {out_dir / 'run1_meta_weights.json'}")

    # End-to-End Evaluation on Meta-Anchor
    p_react_all = expit(np.dot(np.column_stack([cb_react_logits_meta, s1_r_logit, s2_r_logit, ett_r_logit]), w_react))
    p_churn_all = expit(np.dot(np.column_stack([cb_churn_logits_meta, s1_c_logit, s2_c_logit, ett_c_logit]), w_churn))
    p_buy_all = np.where(was_act_meta_cb == 0, p_react_all, 1.0 - p_churn_all)

    cond_z_all = np.maximum(0.0, ridge.predict(np.column_stack([cb_amount_z_meta, s1_cond_z, s2_cond_z, ett_cond_z])))
    z_pred = (p_buy_all ** 1.1) * cond_z_all
    gmv_pred = np.expm1(np.maximum(0.0, z_pred))

    rmsle_meta = float(np.sqrt(np.mean((np.log1p(gmv_pred) - np.log1p(y_meta_cb_gmv)) ** 2)))
    print(f"\n[*] RUN 1 Meta-Anchor Fit RMSLE: {rmsle_meta:.5f}")


if __name__ == "__main__":
    main()
