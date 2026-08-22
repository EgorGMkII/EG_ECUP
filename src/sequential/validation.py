"""Purged Time-CV backtesting and transition state error analysis for Sequential Models."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score


from src.sequential.dataset import (
    CACHE_DIR,
    extract_anchor_targets,
    get_cached_sequence_tensor,
    MemmapMultiAnchorDataset,
    OzonSequenceDataset,
)
from src.sequential.models import DirectGRUModel, MultiTaskGRUModel
from src.sequential.preprocessing import SequentialScaler
from src.sequential.trainer import train_direct_gru, train_multitask_gru
from src.snapshots import generate_panel_anchors, TRAIN_PARQUET

BACKTEST_ANCHORS = [
    date(2025, 10, 13), # Autumn
    date(2025, 12, 8),  # Pre-NewYear
    date(2025, 12, 22), # NY-Transition
    date(2026, 1, 14),  # Post-NewYear
]


def evaluate_state_transitions(
    val_history_tensor: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Evaluates RMSLE breakdown across 4 buyer states: 0->0, 0->>0, >0->0, >0->>0."""
    # Last 30 days of history: index 3 is gmv in CHANNELS
    # If sum of gmv in last 30 days > 0 -> past active
    past_gmv = np.sum(val_history_tensor[:, -30:, 3], axis=1)
    past_active = past_gmv > 0
    fut_active = y_true > 0

    states = {
        "0 -> 0 (Stable Inactive)": (~past_active) & (~fut_active),
        "0 -> >0 (Reactivated / New)": (~past_active) & (fut_active),
        ">0 -> 0 (Churned)": (past_active) & (~fut_active),
        ">0 -> >0 (Stable Active)": (past_active) & (fut_active),
    }

    results = {}
    z_true = np.log1p(np.maximum(y_true, 0.0))
    z_pred = np.log1p(np.maximum(y_pred, 0.0))

    for state_name, mask in states.items():
        count = int(np.sum(mask))
        if count > 0:
            rmsle = float(np.sqrt(np.mean((z_pred[mask] - z_true[mask]) ** 2)))
            pct = float(count / len(y_true) * 100.0)
            results[state_name] = {
                "count": count,
                "share_pct": pct,
                "rmsle": rmsle,
                "mean_pred": float(np.mean(y_pred[mask])),
                "mean_true": float(np.mean(y_true[mask])),
            }
    return results


def run_purged_sequential_backtest(
    val_anchor: date,
    user_ids: List[int],
    data: Optional[pl.DataFrame] = None,
    model_type: str = "direct", # "direct" or "multitask"
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 12,
    batch_size: int = 512,
    seq_len: int = 90,
    recent_k_anchors: int = 12,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """Runs a strict purged backtest for a single anchor date."""
    if data is None:
        data = pl.read_parquet(TRAIN_PARQUET)

    anchors = generate_panel_anchors()
    # Purge: only anchors whose target window (anchor + 30d) strictly ends before val_anchor
    purge_cutoff = val_anchor - timedelta(days=30)
    train_anchors = [a for a in anchors if a <= purge_cutoff]

    if recent_k_anchors and len(train_anchors) > recent_k_anchors:
        train_anchors = train_anchors[-recent_k_anchors:]

    print(f"\n[*] Running {model_type.upper()} GRU Backtest for {val_anchor} (Training on {len(train_anchors)} purged anchors)...")

    # 1. Ensure all training tensors are saved on disk and collect file paths
    train_paths = []
    train_targets = []
    for a in train_anchors:
        filename = f"seq_tensor_{a.strftime('%Y-%m-%d')}_u{len(user_ids)}_t{seq_len}.npy"
        p = CACHE_DIR / filename
        if not p.exists():
            _ = get_cached_sequence_tensor(data, user_ids, a, seq_len=seq_len)
        train_paths.append(p)
        train_targets.append(extract_anchor_targets(data, user_ids, a))

    # 2. Fit Scaler from 1-2 training files without holding full dataset in RAM
    sample_tensor = np.load(train_paths[-1])
    scaler = SequentialScaler().fit(sample_tensor)
    del sample_tensor

    # 3. Load val tensor and target
    val_p = CACHE_DIR / f"seq_tensor_{val_anchor.strftime('%Y-%m-%d')}_u{len(user_ids)}_t{seq_len}.npy"
    if not val_p.exists():
        _ = get_cached_sequence_tensor(data, user_ids, val_anchor, seq_len=seq_len)

    X_val_raw = np.load(val_p)
    y_val = extract_anchor_targets(data, user_ids, val_anchor)
    X_val_scaled = scaler.transform(X_val_raw)

    train_ds = MemmapMultiAnchorDataset(train_paths, train_targets, scaler=scaler)
    val_ds = OzonSequenceDataset(X_val_scaled, y_val, user_ids=user_ids)

    # 4. Train Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_val_scaled.shape[-1]

    if model_type == "direct":
        model = DirectGRUModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        train_direct_gru(model, train_ds, val_ds, epochs=epochs, batch_size=batch_size, verbose=True)

        # Predict in batches
        model = model.to(device)
        model.eval()
        z_pred_list, embs_list = [], []
        inf_bs = 1024
        with torch.no_grad():
            for i in range(0, len(X_val_scaled), inf_bs):
                x_b = torch.from_numpy(X_val_scaled[i : i + inf_bs]).float().to(device)
                z_pred_t, emb_t = model(x_b)
                z_pred_list.append(torch.clamp(z_pred_t, min=0.0).cpu().numpy())
                embs_list.append(emb_t.cpu().numpy())

        z_pred = np.concatenate(z_pred_list)
        embs = np.vstack(embs_list)
        pred_gmv = np.expm1(z_pred)
        p_buy = (z_pred > 0.05).astype(float)

    else:
        model = MultiTaskGRUModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        train_multitask_gru(model, train_ds, val_ds, epochs=epochs, batch_size=batch_size, verbose=True)


        # Predict in batches
        model = model.to(device)
        model.eval()
        p_list, z_cond_list, embs_list = [], [], []
        inf_bs = 1024
        with torch.no_grad():
            for i in range(0, len(X_val_scaled), inf_bs):
                x_b = torch.from_numpy(X_val_scaled[i : i + inf_bs]).float().to(device)
                p_logits_t, z_cond_t, emb_t = model(x_b)
                p_list.append(torch.sigmoid(p_logits_t).cpu().numpy())
                z_cond_list.append(torch.clamp(z_cond_t, min=0.0).cpu().numpy())
                embs_list.append(emb_t.cpu().numpy())

        p_buy = np.concatenate(p_list)
        z_cond = np.concatenate(z_cond_list)
        embs = np.vstack(embs_list)
        z_pred = np.power(p_buy, 1.1) * z_cond
        pred_gmv = np.expm1(z_pred)


    z_true = np.log1p(np.maximum(y_val, 0.0))
    rmsle = float(np.sqrt(np.mean((z_pred - z_true) ** 2)))
    brier = float(np.mean((p_buy - (y_val > 0).astype(float)) ** 2))
    auc = float(roc_auc_score((y_val > 0).astype(int), p_buy))

    # Evaluate states
    state_eval = evaluate_state_transitions(X_val_raw, y_val, pred_gmv)


    print(f"[+] Backtest Result for {val_anchor}: RMSLE = {rmsle:.5f} | ROC-AUC = {auc:.4f} | Brier = {brier:.4f}")

    return {
        "val_anchor": str(val_anchor),
        "rmsle": rmsle,
        "auc": auc,
        "brier": brier,
        "pred_gmv": pred_gmv,
        "z_pred": z_pred,
        "y_true": y_val,
        "embeddings": embs,
        "state_eval": state_eval,
        "scaler": scaler,
    }
