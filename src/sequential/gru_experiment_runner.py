"""GPU Training, Evaluation, and Transition-State Profiling Engine for MultiTask GRU."""

import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, auc, roc_auc_score
from torch.utils.data import DataLoader

from src.sequential.losses import MultiTaskHurdleLoss
from src.sequential.models import MultiTaskTransitionGRUModel
from src.sequential.preprocessing import SequentialScaler
from src.sequential.gru_logging import GRUExperimentLogger, append_registry_record
from src.sequential.gru_sweep import SlicedMemmapDataset


def calculate_quantiles_dict(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(r, p))


class ConfigurableMultiTaskGRU(nn.Module):
    """Configurable MultiTask GRU supporting arbitrary layers, hidden sizes, and input dims."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        # Attention pooling
        self.attn_linear = nn.Linear(hidden_dim, 1)

        # Multi-task heads with configurable dropout
        self.head_react = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):
        out, _ = self.gru(x)  # (B, T, hidden_dim)

        # Attention pooling across sequence steps
        attn_weights = torch.softmax(self.attn_linear(out), dim=1)
        emb = torch.sum(attn_weights * out, dim=1)

        lr = self.head_react(emb).squeeze(-1)
        lc = self.head_churn(emb).squeeze(-1)
        zc = self.head_cond(emb).squeeze(-1)
        zd = self.head_dir(emb).squeeze(-1)
        return lr, lc, zc, zd, emb


def run_gru_training_and_eval(
    config: Dict[str, Any],
    train_dataset: SlicedMemmapDataset,
    val_raw_tensor: np.ndarray,
    val_targets: np.ndarray,
    val_past_buyer: np.ndarray,
    val_user_ids: List[int],
    val_catboost_z: np.ndarray,
    scaler: SequentialScaler,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Runs full training, validation evaluation, logging, and registry registration for one config."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = config["run_id"]
    logger = GRUExperimentLogger(run_id)
    logger.save_config(config)
    logger.log(f"[*] Starting Run {run_id} | seq_len={config['sequence_length']} | anchors={config['anchor_set']} | seed={config.get('seed', 42)}")

    # Deterministic Seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Instantiate Model
    input_dim = config.get("input_dim", 15)
    hidden_dim = config.get("hidden_size", 128)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.15)
    use_attention = config.get("use_attention", True)

    model = MultiTaskTransitionGRUModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # DataLoader - Optimized for High-End GPU (A100) & Multi-core CPU
    batch_size = config.get("batch_size", 2048 if torch.cuda.is_available() else 512)
    num_workers = 4 if (torch.cuda.is_available() and os.name != "nt") else 0
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )

    # Optimizer & Scheduler
    lr = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 10)
    lambda_cls = config.get("lambda_cls", 0.5)
    lambda_reg = config.get("lambda_reg", 1.0)
    alpha = config.get("alpha", 1.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "loss_react": [],
        "loss_churn": [],
        "loss_cond": [],
        "epoch_time_sec": [],
    }

    t0_train = time.time()
    best_loss = float("inf")
    best_epoch = 1
    checkpoint_path = logger.get_checkpoint_path()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        tot_loss, tot_lr, tot_lc, tot_zc, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        for x, y_log, past_b, fut_b in loader:
            x, y_log, past_b, fut_b = x.to(device), y_log.to(device), past_b.to(device), fut_b.to(device)
            optimizer.zero_grad()
            lr_t, lc_t, _, zc_t, _, _ = model(x)

            # Reactivation BCE (on dormant past_b == 0)
            mask_dormant = past_b == 0
            loss_r = bce_fn(lr_t[mask_dormant], fut_b[mask_dormant]) if mask_dormant.sum() > 0 else torch.tensor(0.0, device=device)

            # Churn BCE (on active past_b == 1)
            mask_active = past_b == 1
            loss_c = bce_fn(lc_t[mask_active], 1.0 - fut_b[mask_active]) if mask_active.sum() > 0 else torch.tensor(0.0, device=device)

            # Conditional Regression MSE (strictly on active buyers fut_b > 0.5)
            mask_buyers = fut_b > 0.5
            loss_reg = mse_fn(zc_t[mask_buyers], y_log[mask_buyers]) if mask_buyers.sum() > 0 else torch.tensor(0.0, device=device)

            loss = lambda_cls * (loss_r + loss_c) + lambda_reg * loss_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tot_loss += loss.item()
            tot_lr += loss_r.item()
            tot_lc += loss_c.item()
            tot_zc += loss_reg.item()
            n_batches += 1

        scheduler.step()
        ep_time = time.time() - t_epoch_start
        mean_loss = tot_loss / max(1, n_batches)

        history["epoch"].append(epoch)
        history["train_loss"].append(mean_loss)
        history["loss_react"].append(tot_lr / max(1, n_batches))
        history["loss_churn"].append(tot_lc / max(1, n_batches))
        history["loss_cond"].append(tot_zc / max(1, n_batches))
        history["epoch_time_sec"].append(ep_time)

        logger.log(f"  Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | Loss: {mean_loss:.4f} (React: {tot_lr/max(1,n_batches):.4f}, Churn: {tot_lc/max(1,n_batches):.4f}, Reg: {tot_zc/max(1,n_batches):.4f})")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    total_train_time = time.time() - t0_train
    logger.save_training_history(history)
    peak_gpu_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    # =========================================================================
    # VALIDATION EVALUATION
    # =========================================================================
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    seq_len = config["sequence_length"]
    n_val = len(val_targets)
    val_targets_log = np.log1p(val_targets)
    fut_buyer_val = (val_targets > 0).astype(np.int32)

    lr_list, lc_list, zc_list, zd_list = [], [], [], []
    inf_bs = 1024
    with torch.no_grad():
        for i in range(0, n_val, inf_bs):
            raw_b = val_raw_tensor[i : i + inf_bs]
            sliced_b = raw_b[:, -seq_len:, :]
            sc_mean = scaler.mean[: sliced_b.shape[-1]]
            sc_std = scaler.std[: sliced_b.shape[-1]]
            sc_b = (sliced_b - sc_mean) / sc_std

            xb = torch.from_numpy(sc_b.astype(np.float32)).to(device)
            lr_t, lc_t, _, zc_t, zd_t, _ = model(xb)

            lr_list.append(torch.sigmoid(lr_t).cpu().numpy())
            lc_list.append(torch.sigmoid(lc_t).cpu().numpy())
            zc_list.append(zc_t.cpu().numpy())
            zd_list.append(zd_t.cpu().numpy())

    p_react = np.concatenate(lr_list)
    p_churn = np.concatenate(lc_list)
    z_cond = np.concatenate(zc_list)
    z_dir = np.concatenate(zd_list)

    # Factorized prediction
    p_buy = np.where(val_past_buyer == 0, p_react, 1.0 - p_churn)
    p_buy_alpha = np.power(p_buy, alpha)
    z_fact = (p_buy_alpha * z_cond).astype(np.float32)

    pred_fact_rub = np.clip(np.expm1(z_fact), 0.0, None)
    pred_dir_rub = np.clip(np.expm1(z_dir), 0.0, None)

    # RMSLE metrics
    rmsle_fact = float(np.sqrt(np.mean((np.log1p(pred_fact_rub) - val_targets_log) ** 2)))
    rmsle_dir = float(np.sqrt(np.mean((np.log1p(pred_dir_rub) - val_targets_log) ** 2)))

    # Blend with CatBoost Transitions (50/50)
    z_blend = (0.50 * val_catboost_z + 0.50 * z_fact).astype(np.float32)
    pred_blend_rub = np.clip(np.expm1(z_blend), 0.0, None)
    rmsle_blend = float(np.sqrt(np.mean((np.log1p(pred_blend_rub) - val_targets_log) ** 2)))

    # Transition Classification Metrics
    mask_dormant = val_past_buyer == 0
    mask_active = val_past_buyer == 1

    react_auc = float(roc_auc_score(fut_buyer_val[mask_dormant], p_react[mask_dormant]))
    react_brier = float(brier_score_loss(fut_buyer_val[mask_dormant], p_react[mask_dormant]))
    churn_auc = float(roc_auc_score((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))
    churn_brier = float(brier_score_loss((1 - fut_buyer_val)[mask_active], p_churn[mask_active]))

    # 4 Transition States MSE
    mask_00 = (val_past_buyer == 0) & (fut_buyer_val == 0)
    mask_01 = (val_past_buyer == 0) & (fut_buyer_val == 1)
    mask_10 = (val_past_buyer == 1) & (fut_buyer_val == 0)
    mask_11 = (val_past_buyer == 1) & (fut_buyer_val == 1)

    diff_sq = (np.log1p(pred_fact_rub) - val_targets_log) ** 2
    mse_00 = float(np.mean(diff_sq[mask_00]))
    mse_01 = float(np.mean(diff_sq[mask_01]))
    mse_10 = float(np.mean(diff_sq[mask_10]))
    mse_11 = float(np.mean(diff_sq[mask_11]))

    # Transition metrics table
    df_trans = pl.DataFrame({
        "transition_state": ["00_stable_sleep", "01_reactivation", "10_churn", "11_retention", "overall"],
        "count": [int(mask_00.sum()), int(mask_01.sum()), int(mask_10.sum()), int(mask_11.sum()), len(val_targets)],
        "mse": [mse_00, mse_01, mse_10, mse_11, float(np.mean(diff_sq))],
        "rmsle": [float(np.sqrt(mse_00)), float(np.sqrt(mse_01)), float(np.sqrt(mse_10)), float(np.sqrt(mse_11)), rmsle_fact],
    })
    logger.save_transition_metrics(df_trans)

    # Error correlation with CatBoost
    err_cb = val_catboost_z - val_targets_log
    err_gru = z_fact - val_targets_log
    corr_cb = float(np.corrcoef(err_cb, err_gru)[0, 1])

    # Save Predictions Parquet
    df_preds = pl.DataFrame({
        "user_id": val_user_ids,
        "past_buyer_30d": val_past_buyer,
        "target": val_targets,
        "p_react": p_react,
        "p_churn": p_churn,
        "z_cond": z_cond,
        "z_dir": z_dir,
        "z_gru_fact": z_fact,
        "z_catboost": val_catboost_z,
        "z_blend_5050": z_blend,
        "pred_gru_rub": pred_fact_rub,
        "pred_blend_rub": pred_blend_rub,
    })
    pred_path = logger.save_predictions_parquet(df_preds)

    # Prediction Distribution
    dist_dict = {
        "pred_fact_rub": calculate_quantiles_dict(pred_fact_rub),
        "pred_blend_rub": calculate_quantiles_dict(pred_blend_rub),
    }
    logger.save_prediction_distribution(dist_dict)

    # Compiled Metrics Record
    metrics_summary = {
        "run_id": run_id,
        "status": "completed",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sequence_length": seq_len,
        "anchor_set": config["anchor_set"],
        "n_anchors": config.get("n_anchors", len(train_dataset.mmaps)),
        "n_users": len(val_user_ids),
        "n_samples": len(train_dataset),
        "hidden_size": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "lambda_cls": lambda_cls,
        "lambda_reg": lambda_reg,
        "alpha": alpha,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_time_sec": round(total_train_time, 1),
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "rmsle_direct": round(rmsle_dir, 5),
        "rmsle_factorized": round(rmsle_fact, 5),
        "rmsle_blend_cb": round(rmsle_blend, 5),
        "reactivation_auc": round(react_auc, 4),
        "reactivation_brier": round(react_brier, 4),
        "churn_auc": round(churn_auc, 4),
        "churn_brier": round(churn_brier, 4),
        "mse_00_sleep": round(mse_00, 5),
        "mse_01_react": round(mse_01, 5),
        "mse_10_churn": round(mse_10, 5),
        "mse_11_retention": round(mse_11, 5),
        "cb_error_correlation": round(corr_cb, 4),
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(pred_path),
        "notes": config.get("notes", ""),
    }

    logger.save_metrics(metrics_summary)
    append_registry_record(metrics_summary)

    logger.log(f"[+] Run {run_id} FINISHED in {total_train_time:.1f}s | Factorized RMSLE: {rmsle_fact:.5f} | Blend RMSLE: {rmsle_blend:.5f} | React AUC: {react_auc:.4f} | Churn AUC: {churn_auc:.4f}")

    del model, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return metrics_summary
