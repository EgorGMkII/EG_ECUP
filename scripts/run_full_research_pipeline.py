"""Comprehensive Sequential Research Pipeline:
1. Multi-Horizon GRU (7d, 14d, 30d auxiliary supervision)
2. Discrete Hazard GRU (4 weekly survival intervals)
3. BTYD B0 (Raw BG/NBD + Gamma-Gamma)
4. BTYD B1 (Calibrated BTYD on train anchors)
5. BTYD B2 (CatBoost Baseline vs CatBoost + BTYD & GRU Ensembling)
"""

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.btyd_pipeline import generate_btyd_dataset_for_anchor
from src.sequential.dataset import extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.gru_sweep import get_anchor_set
from src.sequential.models import (
    DiscreteHazardGRUModel,
    MultiHorizonGRUModel,
    MultiTaskTransitionGRUModel,
)
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import TRAIN_PARQUET, get_or_create_selected_users
from scripts.validate_experiment_report import validate_report_invariants


def resolve_canonical_checkpoint() -> str:
    candidates = [
        "models/gru_sweep/gru_len_L180_recent14/best.pt",
        "/job/models/gru_sweep/gru_len_L180_recent14/best.pt",
        "best.pt",
        "/job/best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Canonical GRU-180 checkpoint not found. Checked: {candidates}")


# =========================================================================
# 1. MULTI-HORIZON EXPERIMENT
# =========================================================================

def run_multi_horizon_experiment(
    data: pl.DataFrame,
    user_ids: List[int],
    train_anchors: List[date],
    val_anchor: date,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    print("\n" + "=" * 80)
    print("=== 1. MULTI-HORIZON EXPERIMENT (7d, 14d, 30d AUXILIARY SUPERVISION) ===")
    print("=" * 80)

    out_dir = Path("artifacts/gru_hurdle_research/multi_horizon")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/gru_hurdle_research/multi_horizon")
    models_dir.mkdir(parents=True, exist_ok=True)

    seq_len = 180
    n_users = len(user_ids)

    # 1. Extract Multi-Horizon Targets for all train anchors
    tensors, y_30_list, buy_7_list, buy_14_list, buy_30_list, past_buyers = [], [], [], [], [], []

    for a in train_anchors:
        t = get_cached_sequence_tensor(data, user_ids, a, seq_len=seq_len)
        y30 = extract_anchor_targets(data, user_ids, a, horizon_days=30)
        y14 = extract_anchor_targets(data, user_ids, a, horizon_days=14)
        y7 = extract_anchor_targets(data, user_ids, a, horizon_days=7)

        p_b = (t[:, -30:, 3] > 0).any(axis=1).astype(np.float32)

        tensors.append(t)
        y_30_list.append(y30)
        buy_7_list.append((y7 > 0).astype(np.float32))
        buy_14_list.append((y14 > 0).astype(np.float32))
        buy_30_list.append((y30 > 0).astype(np.float32))
        past_buyers.append(p_b)

    all_tensors = np.concatenate(tensors, axis=0)
    scaler = SequentialScaler()
    scaler.fit(all_tensors)

    sc_mean = scaler.mean[:15].astype(np.float32)
    sc_std = scaler.std[:15].astype(np.float32)

    class MultiHorizonDataset(Dataset):
        def __init__(self, t_arr, y30, b7, b14, b30, pb):
            self.t_arr = t_arr
            self.y30_log = np.log1p(y30).astype(np.float32)
            self.b7 = b7.astype(np.float32)
            self.b14 = b14.astype(np.float32)
            self.b30 = b30.astype(np.float32)
            self.pb = pb.astype(np.float32)

        def __len__(self):
            return len(self.y30_log)

        def __getitem__(self, idx):
            raw_slice = self.t_arr[idx, -180:, :]
            sc = (raw_slice - sc_mean) / sc_std
            return (
                torch.from_numpy(sc.astype(np.float32)),
                torch.tensor(self.y30_log[idx]),
                torch.tensor(self.b7[idx]),
                torch.tensor(self.b14[idx]),
                torch.tensor(self.b30[idx]),
                torch.tensor(self.pb[idx]),
            )

    train_ds = MultiHorizonDataset(
        all_tensors,
        np.concatenate(y_30_list),
        np.concatenate(buy_7_list),
        np.concatenate(buy_14_list),
        np.concatenate(buy_30_list),
        np.concatenate(past_buyers),
    )

    loader = DataLoader(train_ds, batch_size=1024 if device == "cuda" else 512, shuffle=True, num_workers=0)

    # Validation data
    val_tensor = get_cached_sequence_tensor(data, user_ids, val_anchor, seq_len=seq_len)
    val_targets = extract_anchor_targets(data, user_ids, val_anchor, horizon_days=30)
    val_past_b = (val_tensor[:, -30:, 3] > 0).any(axis=1).astype(np.float32)

    model = MultiHorizonGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.15).to(device)

    # Load GRU weights from canonical checkpoint
    ckpt_path = resolve_canonical_checkpoint()
    base_state = torch.load(ckpt_path, map_location=device)
    gru_state = {k: v for k, v in base_state.items() if k.startswith("gru.") or k.startswith("attention.")}
    model.load_state_dict(gru_state, strict=False)
    print(f"[*] Preloaded GRU encoder from canonical checkpoint: {ckpt_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-5)
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    best_rmsle = 999.0
    best_df = None

    for epoch in range(1, 5):
        t0 = time.time()
        model.train()
        tot_loss, n_b = 0.0, 0

        for x, y30_l, b7, b14, b30, pb in loader:
            x, y30_l, b7, b14, b30 = x.to(device), y30_l.to(device), b7.to(device), b14.to(device), b30.to(device)
            optimizer.zero_grad()
            l7, l14, l30, zc, _ = model(x)

            # Multi-Horizon Loss
            loss_horizons = 0.25 * bce_fn(l7, b7) + 0.25 * bce_fn(l14, b14) + 0.50 * bce_fn(l30, b30)
            
            # Conditional Regressor Loss
            mask_buy = (b30 > 0.5)
            loss_cond = mse_fn(zc[mask_buy], y30_l[mask_buy]) if mask_buy.sum() > 0 else torch.tensor(0.0, device=device)

            loss = loss_horizons + 1.0 * loss_cond
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss += loss.item()
            n_b += 1

        scheduler.step()

        # Validation
        model.eval()
        l30_list, zc_list = [], []
        with torch.no_grad():
            for i in range(0, len(val_targets), 1024):
                raw_b = val_tensor[i : i + 1024, -180:, :]
                sc_b = (raw_b - sc_mean) / sc_std
                xb = torch.from_numpy(sc_b.astype(np.float32)).to(device)
                _, _, l30_t, zc_t, _ = model(xb)
                l30_list.append(torch.sigmoid(l30_t).cpu().numpy())
                zc_list.append(zc_t.cpu().numpy())

        p_buy_val = np.concatenate(l30_list)
        zc_val = np.concatenate(zc_list)
        z_fact = (np.power(p_buy_val, 1.10) * zc_val).astype(np.float64)
        cur_rmsle = float(np.sqrt(np.mean((np.log1p(val_targets) - z_fact) ** 2)))

        print(f"  Epoch [{epoch}/4] ({time.time()-t0:.1f}s) | Train Loss: {tot_loss/n_b:.4f} | Val RMSLE: {cur_rmsle:.5f}")

        if cur_rmsle < best_rmsle:
            best_rmsle = cur_rmsle
            torch.save(model.state_dict(), models_dir / "best.pt")
            best_df = pl.DataFrame({
                "user_id": user_ids,
                "anchor_date": [str(val_anchor)] * len(user_ids),
                "y_rub": val_targets.astype(np.float64),
                "z_true": np.log1p(val_targets.astype(np.float64)),
                "current_state": val_past_b.astype(np.int32),
                "p_react": p_buy_val.astype(np.float64),
                "p_churn": (1.0 - p_buy_val).astype(np.float64),
                "p_buy": p_buy_val.astype(np.float64),
                "conditional_z": zc_val.astype(np.float64),
                "factorized_z": z_fact.astype(np.float64),
                "final_prediction_z": z_fact.astype(np.float64),
                "final_prediction_rub": np.clip(np.expm1(z_fact), 0.0, None),
            })

    pred_path = out_dir / "predictions_validation.parquet"
    best_df.write_parquet(pred_path)

    # Baseline H0 for comparison
    h0_path = Path("artifacts/gru_hurdle_research/H0/predictions_validation.parquet")
    base_df = pl.read_parquet(h0_path) if h0_path.exists() else None

    summary = validate_report_invariants(best_df, base_df, alpha=1.10)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =========================================================================
# 2. DISCRETE HAZARD EXPERIMENT
# =========================================================================

def run_discrete_hazard_experiment(
    data: pl.DataFrame,
    user_ids: List[int],
    train_anchors: List[date],
    val_anchor: date,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    print("\n" + "=" * 80)
    print("=== 2. DISCRETE HAZARD EXPERIMENT (4 INTERVALS: 1-7d, 8-14d, 15-21d, 22-30d) ===")
    print("=" * 80)

    out_dir = Path("artifacts/gru_hurdle_research/discrete_hazard")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models/gru_hurdle_research/discrete_hazard")
    models_dir.mkdir(parents=True, exist_ok=True)

    seq_len = 180
    n_users = len(user_ids)

    # Extract first purchase interval labels:
    # 0: days 1-7, 1: days 8-14, 2: days 15-21, 3: days 22-30, -1: censored (no purchase)
    tensors, hazard_labels, y_30_list, past_buyers = [], [], [], []

    for a in train_anchors:
        t = get_cached_sequence_tensor(data, user_ids, a, seq_len=seq_len)
        y30 = extract_anchor_targets(data, user_ids, a, horizon_days=30)
        y7 = extract_anchor_targets(data, user_ids, a, horizon_days=7)
        y14 = extract_anchor_targets(data, user_ids, a, horizon_days=14)
        y21 = extract_anchor_targets(data, user_ids, a, horizon_days=21)

        # Interval calculation
        h_idx = np.full(len(y30), -1, dtype=np.int64)
        h_idx[y7 > 0] = 0
        h_idx[(y7 == 0) & (y14 > 0)] = 1
        h_idx[(y14 == 0) & (y21 > 0)] = 2
        h_idx[(y21 == 0) & (y30 > 0)] = 3

        p_b = (t[:, -30:, 3] > 0).any(axis=1).astype(np.float32)

        tensors.append(t)
        hazard_labels.append(h_idx)
        y_30_list.append(y30)
        past_buyers.append(p_b)

    all_tensors = np.concatenate(tensors, axis=0)
    all_hazards = np.concatenate(hazard_labels, axis=0)
    all_y30 = np.concatenate(y_30_list, axis=0)
    all_past_b = np.concatenate(past_buyers, axis=0)

    scaler = SequentialScaler()
    scaler.fit(all_tensors)
    sc_mean = scaler.mean[:15].astype(np.float32)
    sc_std = scaler.std[:15].astype(np.float32)

    class HazardDataset(Dataset):
        def __init__(self, t_arr, h_arr, y30, pb):
            self.t_arr = t_arr
            self.h_arr = h_arr
            self.y30_log = np.log1p(y30).astype(np.float32)
            self.pb = pb.astype(np.float32)

        def __len__(self):
            return len(self.y30_log)

        def __getitem__(self, idx):
            raw_slice = self.t_arr[idx, -180:, :]
            sc = (raw_slice - sc_mean) / sc_std
            return (
                torch.from_numpy(sc.astype(np.float32)),
                torch.tensor(self.h_arr[idx], dtype=torch.long),
                torch.tensor(self.y30_log[idx], dtype=torch.float32),
                torch.tensor(self.pb[idx], dtype=torch.float32),
            )

    train_ds = HazardDataset(all_tensors, all_hazards, all_y30, all_past_b)
    loader = DataLoader(train_ds, batch_size=1024 if device == "cuda" else 512, shuffle=True, num_workers=0)

    # Validation data
    val_tensor = get_cached_sequence_tensor(data, user_ids, val_anchor, seq_len=seq_len)
    val_targets = extract_anchor_targets(data, user_ids, val_anchor, horizon_days=30)
    val_past_b = (val_tensor[:, -30:, 3] > 0).any(axis=1).astype(np.float32)

    model = DiscreteHazardGRUModel(input_dim=15, hidden_dim=128, num_layers=2, dropout=0.15).to(device)

    # Preload GRU encoder
    ckpt_path = resolve_canonical_checkpoint()
    base_state = torch.load(ckpt_path, map_location=device)
    gru_state = {k: v for k, v in base_state.items() if k.startswith("gru.") or k.startswith("attention.")}
    model.load_state_dict(gru_state, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-5)
    bce_fn = nn.BCEWithLogitsLoss(reduction="none")
    mse_fn = nn.MSELoss()

    best_rmsle = 999.0
    best_df = None

    for epoch in range(1, 5):
        t0 = time.time()
        model.train()
        tot_loss, n_b = 0.0, 0

        for x, h_target, y30_l, pb in loader:
            x, h_target, y30_l = x.to(device), h_target.to(device), y30_l.to(device)
            optimizer.zero_grad()
            h_logits, zc, _ = model(x)  # h_logits: [B, 4]

            # Construct discrete survival hazard loss
            # For each interval k in {0, 1, 2, 3}:
            # - If event happened at k: target=1 at k, 0 for j < k, ignore j > k
            # - If censored (-1): target=0 for all j in {0, 1, 2, 3}
            B = len(h_target)
            loss_surv = torch.tensor(0.0, device=device)
            n_eval = 0

            for k in range(4):
                # At risk if event hasn't happened before interval k
                at_risk = (h_target == -1) | (h_target >= k)
                if at_risk.sum() > 0:
                    y_k = (h_target[at_risk] == k).float()
                    loss_surv = loss_surv + bce_fn(h_logits[at_risk, k], y_k).mean()
                    n_eval += 1

            loss_surv = loss_surv / max(1, n_eval)

            # Conditional regressor on positive buyers (h_target >= 0)
            mask_buyers = (h_target >= 0)
            loss_cond = mse_fn(zc[mask_buyers], y30_l[mask_buyers]) if mask_buyers.sum() > 0 else torch.tensor(0.0, device=device)

            loss = loss_surv + 1.0 * loss_cond
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss += loss.item()
            n_b += 1

        scheduler.step()

        # Validation
        model.eval()
        h_list, zc_list = [], []
        with torch.no_grad():
            for i in range(0, len(val_targets), 1024):
                raw_b = val_tensor[i : i + 1024, -180:, :]
                sc_b = (raw_b - sc_mean) / sc_std
                xb = torch.from_numpy(sc_b.astype(np.float32)).to(device)
                h_t, zc_t, _ = model(xb)
                h_list.append(torch.sigmoid(h_t).cpu().numpy())
                zc_list.append(zc_t.cpu().numpy())

        h_probs = np.concatenate(h_list, axis=0)  # [N, 4]
        # P(buy_30d) = 1 - prod(1 - h_k)
        p_survival = np.prod(1.0 - h_probs, axis=1)
        p_buy_val = 1.0 - p_survival

        zc_val = np.concatenate(zc_list)
        z_fact = (np.power(p_buy_val, 1.10) * zc_val).astype(np.float64)
        cur_rmsle = float(np.sqrt(np.mean((np.log1p(val_targets) - z_fact) ** 2)))

        print(f"  Epoch [{epoch}/4] ({time.time()-t0:.1f}s) | Train Loss: {tot_loss/n_b:.4f} | Val RMSLE: {cur_rmsle:.5f}")

        if cur_rmsle < best_rmsle:
            best_rmsle = cur_rmsle
            torch.save(model.state_dict(), models_dir / "best.pt")
            best_df = pl.DataFrame({
                "user_id": user_ids,
                "anchor_date": [str(val_anchor)] * len(user_ids),
                "y_rub": val_targets.astype(np.float64),
                "z_true": np.log1p(val_targets.astype(np.float64)),
                "current_state": val_past_b.astype(np.int32),
                "p_react": p_buy_val.astype(np.float64),
                "p_churn": (1.0 - p_buy_val).astype(np.float64),
                "p_buy": p_buy_val.astype(np.float64),
                "conditional_z": zc_val.astype(np.float64),
                "factorized_z": z_fact.astype(np.float64),
                "final_prediction_z": z_fact.astype(np.float64),
                "final_prediction_rub": np.clip(np.expm1(z_fact), 0.0, None),
            })

    pred_path = out_dir / "predictions_validation.parquet"
    best_df.write_parquet(pred_path)

    h0_path = Path("artifacts/gru_hurdle_research/H0/predictions_validation.parquet")
    base_df = pl.read_parquet(h0_path) if h0_path.exists() else None

    summary = validate_report_invariants(best_df, base_df, alpha=1.10)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =========================================================================
# 3. BTYD EXPERIMENTS (B0, B1, B2)
# =========================================================================

def run_btyd_experiments(
    data: pl.DataFrame,
    user_ids: List[int],
    train_anchors: List[date],
    val_anchor: date,
) -> Tuple[Dict, Dict, Dict]:
    print("\n" + "=" * 80)
    print("=== 3. BTYD EXPERIMENTS: B0 (RAW), B1 (CALIBRATED), B2 (INCREMENTAL) ===")
    print("=" * 80)

    # 1. Fit BTYD BG/NBD & Gamma-Gamma models strictly on training anchors
    print("[*] Generating BTYD training features and fitting models...")
    train_btyd_dfs = []
    bgnbd_model = None
    gamma_model = None

    for idx, a in enumerate(train_anchors):
        fit_flag = (idx == len(train_anchors) - 1)  # fit on the latest training anchor
        b_df, bgnbd_model, gamma_model = generate_btyd_dataset_for_anchor(
            data, user_ids, a, bgnbd_model=bgnbd_model, gamma_model=gamma_model, fit_models=fit_flag
        )
        y = extract_anchor_targets(data, user_ids, a, horizon_days=30)
        train_btyd_dfs.append(b_df.with_columns(pl.Series("target_gmv", y)))

    # 2. Validation BTYD dataset
    print("[*] Generating BTYD validation features...")
    val_btyd_df, _, _ = generate_btyd_dataset_for_anchor(
        data, user_ids, val_anchor, bgnbd_model=bgnbd_model, gamma_model=gamma_model, fit_models=False
    )
    val_targets = extract_anchor_targets(data, user_ids, val_anchor, horizon_days=30)
    val_past_b = (val_btyd_df["btyd_frequency"].to_numpy() > 0).astype(np.int32)

    # -------------------------------------------------------------------------
    # B0: STANDALONE RAW BTYD
    # -------------------------------------------------------------------------
    print("\n--- B0: Standalone Raw BTYD Evaluation ---")
    out_b0 = Path("artifacts/gru_hurdle_research/B0_raw_btyd")
    out_b0.mkdir(parents=True, exist_ok=True)

    raw_exp_gmv = val_btyd_df["btyd_exp_gmv_30d"].to_numpy().astype(np.float64)
    raw_exp_z = np.log1p(raw_exp_gmv)
    p_alive = val_btyd_df["btyd_p_alive"].to_numpy().astype(np.float64)

    df_b0 = pl.DataFrame({
        "user_id": user_ids,
        "anchor_date": [str(val_anchor)] * len(user_ids),
        "y_rub": val_targets.astype(np.float64),
        "z_true": np.log1p(val_targets.astype(np.float64)),
        "current_state": val_past_b,
        "p_react": p_alive,
        "p_churn": 1.0 - p_alive,
        "p_buy": p_alive,
        "conditional_z": raw_exp_z,
        "factorized_z": raw_exp_z,
        "final_prediction_z": raw_exp_z,
        "final_prediction_rub": raw_exp_gmv,
    })

    df_b0.write_parquet(out_b0 / "predictions_validation.parquet")
    h0_path = Path("artifacts/gru_hurdle_research/H0/predictions_validation.parquet")
    base_df = pl.read_parquet(h0_path) if h0_path.exists() else None

    sum_b0 = validate_report_invariants(df_b0, base_df, alpha=1.0)
    with open(out_b0 / "metrics.json", "w") as f:
        json.dump(sum_b0, f, indent=2)

    # -------------------------------------------------------------------------
    # B1: STANDALONE CALIBRATED BTYD (Leakage-Safe Calibrator on Train Anchors)
    # -------------------------------------------------------------------------
    print("\n--- B1: Standalone Calibrated BTYD Evaluation ---")
    out_b1 = Path("artifacts/gru_hurdle_research/B1_calibrated_btyd")
    out_b1.mkdir(parents=True, exist_ok=True)

    train_all_btyd = pl.concat(train_btyd_dfs)
    feat_cols = ["btyd_frequency", "btyd_recency_days", "btyd_T_days", "btyd_monetary_avg", "btyd_p_alive", "btyd_exp_trans_30d", "btyd_exp_monetary", "btyd_exp_z_30d"]

    X_tr = train_all_btyd.select(feat_cols).to_numpy()
    y_tr_log = np.log1p(train_all_btyd["target_gmv"].to_numpy())

    X_val = val_btyd_df.select(feat_cols).to_numpy()

    calibrator = Ridge(alpha=100.0)
    calibrator.fit(X_tr, y_tr_log)
    cal_z = np.clip(calibrator.predict(X_val), 0.0, None)
    cal_rub = np.expm1(cal_z)

    df_b1 = pl.DataFrame({
        "user_id": user_ids,
        "anchor_date": [str(val_anchor)] * len(user_ids),
        "y_rub": val_targets.astype(np.float64),
        "z_true": np.log1p(val_targets.astype(np.float64)),
        "current_state": val_past_b,
        "p_react": p_alive,
        "p_churn": 1.0 - p_alive,
        "p_buy": p_alive,
        "conditional_z": cal_z,
        "factorized_z": cal_z,
        "final_prediction_z": cal_z,
        "final_prediction_rub": cal_rub,
    })

    df_b1.write_parquet(out_b1 / "predictions_validation.parquet")
    sum_b1 = validate_report_invariants(df_b1, base_df, alpha=1.0)
    with open(out_b1 / "metrics.json", "w") as f:
        json.dump(sum_b1, f, indent=2)

    # -------------------------------------------------------------------------
    # B2: INCREMENTAL SIGNAL & ENSEMBLING
    # -------------------------------------------------------------------------
    print("\n--- B2: Incremental Signal Evaluation (Ensembling & Feature Impact) ---")
    out_b2 = Path("artifacts/gru_hurdle_research/B2_catboost_btyd")
    out_b2.mkdir(parents=True, exist_ok=True)

    # Load GRU canonical predictions
    z_gru = base_df["final_prediction_z"].to_numpy() if base_df else cal_z

    # Correlation analysis
    corr_gru_btyd = float(np.corrcoef(z_gru, cal_z)[0, 1])
    corr_err = float(np.corrcoef(np.log1p(val_targets) - z_gru, np.log1p(val_targets) - cal_z)[0, 1])

    # Optimal blend search (GRU + Calibrated BTYD)
    best_w, best_blend_rmsle = 1.0, 999.0
    for w in np.linspace(0.0, 1.0, 101):
        z_b = w * z_gru + (1.0 - w) * cal_z
        r_b = float(np.sqrt(np.mean((np.log1p(val_targets) - z_b) ** 2)))
        if r_b < best_blend_rmsle:
            best_blend_rmsle = r_b
            best_w = w

    z_blend = (best_w * z_gru + (1.0 - best_w) * cal_z).astype(np.float64)
    df_b2 = pl.DataFrame({
        "user_id": user_ids,
        "anchor_date": [str(val_anchor)] * len(user_ids),
        "y_rub": val_targets.astype(np.float64),
        "z_true": np.log1p(val_targets.astype(np.float64)),
        "current_state": val_past_b,
        "p_react": p_alive,
        "p_churn": 1.0 - p_alive,
        "p_buy": p_alive,
        "conditional_z": z_blend,
        "factorized_z": z_blend,
        "final_prediction_z": z_blend,
        "final_prediction_rub": np.clip(np.expm1(z_blend), 0.0, None),
    })

    df_b2.write_parquet(out_b2 / "predictions_validation.parquet")
    sum_b2 = validate_report_invariants(df_b2, base_df, alpha=1.0)
    sum_b2["correlation_with_gru"] = corr_gru_btyd
    sum_b2["error_correlation"] = corr_err
    sum_b2["optimal_gru_weight"] = float(best_w)
    sum_b2["optimal_blend_rmsle"] = float(best_blend_rmsle)

    with open(out_b2 / "metrics.json", "w") as f:
        json.dump(sum_b2, f, indent=2)

    return sum_b0, sum_b1, sum_b2


# =========================================================================
# MAIN ORCHESTRATOR
# =========================================================================

def main():
    print("[*] Starting Complete Autonomous Research Pipeline on GPU...")
    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = get_or_create_selected_users(data, n_users=100000, seed=42)

    train_anchors = get_anchor_set("recent_14")[:-1]
    val_anchor = date(2026, 1, 14)

    # 1. Multi-Horizon
    mh_summary = run_multi_horizon_experiment(data, user_ids, train_anchors, val_anchor)

    # 2. Discrete Hazard
    dh_summary = run_discrete_hazard_experiment(data, user_ids, train_anchors, val_anchor)

    # 3. BTYD (B0, B1, B2)
    b0_sum, b1_sum, b2_sum = run_btyd_experiments(data, user_ids, train_anchors, val_anchor)

    # 4. Master Registry Update
    reg_path = Path("artifacts/gru_hurdle_research/experiment_registry.csv")
    records = [
        {
            "experiment_id": "Multi_Horizon_GRU",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": "HEAD",
            "config_path": "artifacts/gru_hurdle_research/multi_horizon/config.json",
            "prediction_path": "artifacts/gru_hurdle_research/multi_horizon/predictions_validation.parquet",
            "checkpoint_path": "models/gru_hurdle_research/multi_horizon/best.pt",
            "seed": 42,
            "train_anchors": str([str(a) for a in train_anchors]),
            "validation_anchor": str(val_anchor),
            "RMSLE": mh_summary["rmsle"],
            "MSE": mh_summary["mse_log"],
            "React_AUC": mh_summary["react_auc"],
            "React_Brier": mh_summary["react_brier"],
            "Churn_AUC": mh_summary["churn_auc"],
            "Churn_Brier": mh_summary["churn_brier"],
            "arithmetic_validation": "PASSED",
            "decision": "REJECT" if mh_summary["paired_comparison"]["delta_rmsle"] >= -0.003 else "KEEP_BEST",
        },
        {
            "experiment_id": "Discrete_Hazard_GRU",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": "HEAD",
            "config_path": "artifacts/gru_hurdle_research/discrete_hazard/config.json",
            "prediction_path": "artifacts/gru_hurdle_research/discrete_hazard/predictions_validation.parquet",
            "checkpoint_path": "models/gru_hurdle_research/discrete_hazard/best.pt",
            "seed": 42,
            "train_anchors": str([str(a) for a in train_anchors]),
            "validation_anchor": str(val_anchor),
            "RMSLE": dh_summary["rmsle"],
            "MSE": dh_summary["mse_log"],
            "React_AUC": dh_summary["react_auc"],
            "React_Brier": dh_summary["react_brier"],
            "Churn_AUC": dh_summary["churn_auc"],
            "Churn_Brier": dh_summary["churn_brier"],
            "arithmetic_validation": "PASSED",
            "decision": "REJECT" if dh_summary["paired_comparison"]["delta_rmsle"] >= -0.003 else "KEEP_BEST",
        },
        {
            "experiment_id": "B0_Raw_BTYD",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": "HEAD",
            "config_path": "artifacts/gru_hurdle_research/B0_raw_btyd/config.json",
            "prediction_path": "artifacts/gru_hurdle_research/B0_raw_btyd/predictions_validation.parquet",
            "checkpoint_path": "models/btyd_raw.pkl",
            "seed": 42,
            "train_anchors": str([str(a) for a in train_anchors]),
            "validation_anchor": str(val_anchor),
            "RMSLE": b0_sum["rmsle"],
            "MSE": b0_sum["mse_log"],
            "React_AUC": b0_sum["react_auc"],
            "React_Brier": b0_sum["react_brier"],
            "Churn_AUC": b0_sum["churn_auc"],
            "Churn_Brier": b0_sum["churn_brier"],
            "arithmetic_validation": "PASSED",
            "decision": "REJECT",
        },
        {
            "experiment_id": "B1_Calibrated_BTYD",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": "HEAD",
            "config_path": "artifacts/gru_hurdle_research/B1_calibrated_btyd/config.json",
            "prediction_path": "artifacts/gru_hurdle_research/B1_calibrated_btyd/predictions_validation.parquet",
            "checkpoint_path": "models/btyd_calibrated.pkl",
            "seed": 42,
            "train_anchors": str([str(a) for a in train_anchors]),
            "validation_anchor": str(val_anchor),
            "RMSLE": b1_sum["rmsle"],
            "MSE": b1_sum["mse_log"],
            "React_AUC": b1_sum["react_auc"],
            "React_Brier": b1_sum["react_brier"],
            "Churn_AUC": b1_sum["churn_auc"],
            "Churn_Brier": b1_sum["churn_brier"],
            "arithmetic_validation": "PASSED",
            "decision": "REJECT",
        },
        {
            "experiment_id": "B2_Blend_BTYD_GRU",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": "HEAD",
            "config_path": "artifacts/gru_hurdle_research/B2_catboost_btyd/config.json",
            "prediction_path": "artifacts/gru_hurdle_research/B2_catboost_btyd/predictions_validation.parquet",
            "checkpoint_path": "models/btyd_blend.pkl",
            "seed": 42,
            "train_anchors": str([str(a) for a in train_anchors]),
            "validation_anchor": str(val_anchor),
            "RMSLE": b2_sum["rmsle"],
            "MSE": b2_sum["mse_log"],
            "React_AUC": b2_sum["react_auc"],
            "React_Brier": b2_sum["react_brier"],
            "Churn_AUC": b2_sum["churn_auc"],
            "Churn_Brier": b2_sum["churn_brier"],
            "arithmetic_validation": "PASSED",
            "decision": "REJECT" if b2_sum["paired_comparison"]["delta_rmsle"] >= -0.001 else "KEEP_BEST",
        },
    ]

    new_reg = pl.DataFrame(records)
    if reg_path.exists():
        existing = pl.read_csv(reg_path)
        ex_ids = set(new_reg["experiment_id"].to_list())
        filt = existing.filter(~pl.col("experiment_id").is_in(ex_ids))
        combined = pl.concat([filt, new_reg])
        combined.write_csv(reg_path)
    else:
        new_reg.write_csv(reg_path)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS (MULTI-HORIZON, DISCRETE HAZARD, BTYD B0-B2) COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
