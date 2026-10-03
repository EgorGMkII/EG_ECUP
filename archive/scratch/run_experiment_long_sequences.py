"""Experiment: Long-Sequence Encoders (GRU-90 vs GRU-365 vs Hierarchical GRU vs Patch Transformer-365)."""

import gc
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.models import (
    HierarchicalGRUModel,
    MultiTaskTransitionGRUModel,
    PatchTransformer365Model,
)
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import generate_panel_anchors, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.inference import compute_factorized_gmv
from src.transitions.metrics import decompose_mse_by_transitions, evaluate_classifier_metrics
from src.validation import get_snapshot_path

TRANSITIONS_ARTIFACTS = Path("artifacts/transitions")
TRANSITIONS_ARTIFACTS.mkdir(parents=True, exist_ok=True)
VAL_ANCHOR = date(2026, 1, 14)


class MemmapTransitionSequenceDataset(Dataset):
    """Zero-RAM PyTorch Dataset streaming 365-day (or 90-day) sequences directly from disk via mmap."""

    def __init__(
        self,
        tensor_paths: List[Path],
        targets_list: List[np.ndarray],
        past_buyers_list: List[np.ndarray],
        scaler: Optional[SequentialScaler] = None,
        seq_len: int = 365,
    ):
        self.tensor_paths = tensor_paths
        self.scaler = scaler
        self.seq_len = seq_len

        # Open memmaps
        self.mmaps = [np.load(p, mmap_mode="r") for p in tensor_paths]
        self.lengths = [len(m) for m in self.mmaps]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

        # Targets in RAM (few MBs total)
        y_all = np.concatenate(targets_list).astype(np.float32)
        self.y_true = torch.from_numpy(y_all).float()
        self.y_log = torch.log1p(torch.clamp(self.y_true, min=0.0))
        self.past_buyer = torch.from_numpy(np.concatenate(past_buyers_list)).float()
        self.fut_buyer = (self.y_true > 0).float()
        self.y_churn = (1.0 - self.fut_buyer)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int):
        # Locate which anchor tensor this index belongs to
        tensor_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[tensor_idx]

        # Extract sequence from memmap
        raw_seq = self.mmaps[tensor_idx][local_idx] # (365, 15)
        if self.seq_len < raw_seq.shape[0]:
            raw_seq = raw_seq[-self.seq_len :, :]

        if self.scaler is not None:
            scaled_seq = (raw_seq - self.scaler.mean) / self.scaler.std
        else:
            scaled_seq = raw_seq

        x_tensor = torch.from_numpy(scaled_seq.astype(np.float32)).float()

        return (
            x_tensor,
            self.y_log[idx],
            self.fut_buyer[idx],
            self.past_buyer[idx],
            self.y_churn[idx],
        )



def train_transition_seq_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 6,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    name: str = "Model",
) -> None:
    """Trains a sequence model with masked transition and multi-task regression losses."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    bce_fn = nn.BCEWithLogitsLoss(reduction="none")
    mse_fn = nn.MSELoss(reduction="none")

    print(f"[*] Training {name} on {device} ({epochs} epochs)...")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y_log, fut_buy, past_buy, y_churn in train_loader:
            x, y_log = x.to(device), y_log.to(device)
            fut_buy, past_buy, y_churn = fut_buy.to(device), past_buy.to(device), y_churn.to(device)

            optimizer.zero_grad()
            l_react, l_churn, l_buy, z_cond, z_dir, _ = model(x)

            # Masked losses
            mask_dormant = (past_buy == 0)
            loss_react = bce_fn(l_react[mask_dormant], fut_buy[mask_dormant]).mean() if mask_dormant.sum() > 0 else torch.tensor(0.0, device=device)

            mask_active = (past_buy == 1)
            loss_churn = bce_fn(l_churn[mask_active], y_churn[mask_active]).mean() if mask_active.sum() > 0 else torch.tensor(0.0, device=device)

            loss_buy = bce_fn(l_buy, fut_buy).mean()

            mask_buyers = (fut_buy == 1)
            loss_cond = mse_fn(z_cond[mask_buyers], y_log[mask_buyers]).mean() if mask_buyers.sum() > 0 else torch.tensor(0.0, device=device)
            loss_dir = mse_fn(z_dir, y_log).mean()

            loss = 1.0 * loss_dir + 0.5 * loss_cond + 0.5 * loss_react + 0.5 * loss_churn + 0.2 * loss_buy
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        print(f"  {name} Epoch [{ep:02d}/{epochs:02d}] | Loss: {total_loss/max(n_batches, 1):.4f}")


def evaluate_seq_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    past_buyer_val: np.ndarray,
    device: torch.device,
) -> dict:
    """Evaluates classification and factorized regression metrics on validation."""
    model.eval()
    mask_dormant_val = (past_buyer_val == 0)
    mask_active_val = (past_buyer_val == 1)
    fut_buyer_val = (y_val > 0).astype(np.int32)
    y_react_val = fut_buyer_val[mask_dormant_val]
    y_churn_val = (1 - fut_buyer_val)[mask_active_val]

    l_react_l, l_churn_l, l_buy_l, z_cond_l, z_dir_l = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X_val), 1024):
            xb = torch.from_numpy(X_val[i : i + 1024]).float().to(device)
            lr, lc, lb, zc, zd, _ = model(xb)
            l_react_l.append(lr.cpu().numpy())
            l_churn_l.append(lc.cpu().numpy())
            l_buy_l.append(lb.cpu().numpy())
            z_cond_l.append(zc.cpu().numpy())
            z_dir_l.append(zd.cpu().numpy())

    l_react = np.concatenate(l_react_l)
    l_churn = np.concatenate(l_churn_l)
    z_cond = np.concatenate(z_cond_l)
    z_dir = np.concatenate(z_dir_l)

    p_react = 1.0 / (1.0 + np.exp(-l_react[mask_dormant_val]))
    p_churn = 1.0 / (1.0 + np.exp(-l_churn[mask_active_val]))

    p_buy_fact = np.zeros(len(y_val), dtype=np.float32)
    p_buy_fact[mask_dormant_val] = p_react
    p_buy_fact[mask_active_val] = 1.0 - p_churn

    z_fact, y_pred_fact = compute_factorized_gmv(p_buy_fact, z_cond, power_p=1.1)
    y_pred_dir = np.clip(np.expm1(z_dir), 0.0, None)

    m_react = evaluate_classifier_metrics(y_react_val, p_react, "Reactivation")
    m_churn = evaluate_classifier_metrics(y_churn_val, p_churn, "Churn")
    decomp_fact = decompose_mse_by_transitions(y_val, y_pred_fact, past_buyer_val)
    decomp_dir = decompose_mse_by_transitions(y_val, y_pred_dir, past_buyer_val)

    return {
        "p_react": p_react,
        "p_churn": p_churn,
        "p_buy_fact": p_buy_fact,
        "z_fact": z_fact,
        "z_dir": z_dir,
        "y_pred_fact": y_pred_fact,
        "y_pred_dir": y_pred_dir,
        "react_auc": m_react["roc_auc"],
        "react_brier": m_react["brier_score"],
        "churn_auc": m_churn["roc_auc"],
        "churn_brier": m_churn["brier_score"],
        "rmsle_fact": decomp_fact["total_rmsle"],
        "rmsle_dir": decomp_dir["total_rmsle"],
        "decomp_fact": decomp_fact,
    }


def main():
    print("===================================================================")
    print("=== EXPERIMENTS C & D: LONG-SEQUENCE ENCODERS BENCHMARK ===")
    print("===================================================================")

    data = pl.read_parquet(TRAIN_PARQUET)
    anchors = generate_panel_anchors()
    purge_cutoff = VAL_ANCHOR - timedelta(days=30)
    train_anchors = [a for a in anchors if a <= purge_cutoff][-6:] # 6 recent training anchors

    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    user_ids = val_snap["user_id"].to_list()
    y_val = val_snap["target"].to_numpy().astype(np.float32)
    past_buyer_val = (val_snap["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32)

    # 1. Prepare 365-day Sequence Tensors
    print(f"[*] Ensuring 365-day daily sequence tensors exist for Validation and {len(train_anchors)} Train anchors...")
    val_tensor_path = CACHE_DIR / f"seq_tensor_2026-01-14_u{len(user_ids)}_t365.npy"
    if not val_tensor_path.exists():
        _ = get_cached_sequence_tensor(data, user_ids, VAL_ANCHOR, seq_len=365)

    X_val_365_raw = np.load(val_tensor_path)
    scaler_365 = SequentialScaler().fit(X_val_365_raw[:20000])
    X_val_365 = scaler_365.transform(X_val_365_raw)
    X_val_90 = X_val_365[:, -90:, :]

    # Check/create training tensor paths
    train_paths_365, y_tr_list, past_b_tr_list = [], [], []
    for a in train_anchors:
        t_path = CACHE_DIR / f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(user_ids)}_t365.npy"
        if not t_path.exists():
            _ = get_cached_sequence_tensor(data, user_ids, a, seq_len=365)
        train_paths_365.append(t_path)

        y_a = extract_anchor_targets(data, user_ids, a)
        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        pb_a = (snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32)

        y_tr_list.append(y_a)
        past_b_tr_list.append(pb_a)
        del snap_a

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader for 365d (Zero-RAM Memmap streaming)
    train_ds_365 = MemmapTransitionSequenceDataset(train_paths_365, y_tr_list, past_b_tr_list, scaler=scaler_365, seq_len=365)
    train_loader_365 = DataLoader(train_ds_365, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

    # DataLoader for 90d (Zero-RAM Memmap streaming)
    train_ds_90 = MemmapTransitionSequenceDataset(train_paths_365, y_tr_list, past_b_tr_list, scaler=scaler_365, seq_len=90)
    train_loader_90 = DataLoader(train_ds_90, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)


    # -------------------------------------------------------------------------
    # 1. Model 1: GRU-90 (Baseline Control)
    # -------------------------------------------------------------------------
    gru_90 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2).to(device)
    train_transition_seq_model(gru_90, train_loader_90, epochs=6, lr=1e-3, device=device, name="GRU-90")
    res_gru90 = evaluate_seq_model(gru_90, X_val_90, y_val, past_buyer_val, device)

    # -------------------------------------------------------------------------
    # 2. Model 2: GRU-365 (Full 365-day Daily History)
    # -------------------------------------------------------------------------
    gru_365 = MultiTaskTransitionGRUModel(input_dim=15, hidden_dim=128, num_layers=2).to(device)
    train_transition_seq_model(gru_365, train_loader_365, epochs=6, lr=1e-3, device=device, name="GRU-365")
    res_gru365 = evaluate_seq_model(gru_365, X_val_365, y_val, past_buyer_val, device)

    # -------------------------------------------------------------------------
    # 3. Model 3: Patch Transformer-365 (52 Weekly Patches)
    # -------------------------------------------------------------------------
    tf_365 = PatchTransformer365Model(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3).to(device)
    train_transition_seq_model(tf_365, train_loader_365, epochs=6, lr=1e-3, device=device, name="PatchTransformer-365")
    res_tf365 = evaluate_seq_model(tf_365, X_val_365, y_val, past_buyer_val, device)

    # -------------------------------------------------------------------------
    # 4. Model 4: Hierarchical GRU (90d Daily + 275d Weekly)
    # -------------------------------------------------------------------------
    hier_gru = HierarchicalGRUModel(input_dim=15, hidden_daily=96, hidden_weekly=64).to(device)
    train_transition_seq_model(hier_gru, train_loader_365, epochs=6, lr=1e-3, device=device, name="Hierarchical-GRU")
    res_hier = evaluate_seq_model(hier_gru, X_val_365, y_val, past_buyer_val, device)

    # -------------------------------------------------------------------------
    # Comparison Table
    # -------------------------------------------------------------------------
    comp_df = pl.DataFrame({
        "Model": [
            "1. GRU-90 (Control Baseline)",
            "2. GRU-365 (Full Year Daily)",
            "3. Patch Transformer-365 (52 Weekly Patches)",
            "4. Hierarchical GRU (90d Daily + 275d Weekly)",
        ],
        "RMSLE_Direct": [
            res_gru90["rmsle_dir"],
            res_gru365["rmsle_dir"],
            res_tf365["rmsle_dir"],
            res_hier["rmsle_dir"],
        ],
        "RMSLE_Factorized": [
            res_gru90["rmsle_fact"],
            res_gru365["rmsle_fact"],
            res_tf365["rmsle_fact"],
            res_hier["rmsle_fact"],
        ],
        "Reactivation_AUC": [
            res_gru90["react_auc"],
            res_gru365["react_auc"],
            res_tf365["react_auc"],
            res_hier["react_auc"],
        ],
        "Reactivation_Brier": [
            res_gru90["react_brier"],
            res_gru365["react_brier"],
            res_tf365["react_brier"],
            res_hier["react_brier"],
        ],
        "Churn_AUC": [
            res_gru90["churn_auc"],
            res_gru365["churn_auc"],
            res_tf365["churn_auc"],
            res_hier["churn_auc"],
        ],
        "Churn_Brier": [
            res_gru90["churn_brier"],
            res_gru365["churn_brier"],
            res_tf365["churn_brier"],
            res_hier["churn_brier"],
        ],
    })

    print("\n===================================================================")
    print("=== LONG-SEQUENCE ARCHITECTURES BENCHMARK RESULTS ===")
    print("===================================================================")
    print(comp_df)

    # Save Predictions
    seq_pred_df = pl.DataFrame({
        "user_id": user_ids,
        "past_buyer_30d": past_buyer_val,
        "target": y_val,
        "z_gru90_dir": res_gru90["z_dir"],
        "z_gru365_dir": res_gru365["z_dir"],
        "z_tf365_dir": res_tf365["z_dir"],
        "z_hier_dir": res_hier["z_dir"],
        "z_gru90_fact": res_gru90["z_fact"],
        "z_gru365_fact": res_gru365["z_fact"],
        "z_tf365_fact": res_tf365["z_fact"],
        "z_hier_fact": res_hier["z_fact"],
    })
    seq_pred_df.write_parquet(TRANSITIONS_ARTIFACTS / "experiment_long_seq_predictions.parquet")
    print(f"[+] Saved predictions to {TRANSITIONS_ARTIFACTS / 'experiment_long_seq_predictions.parquet'}")


if __name__ == "__main__":
    main()
