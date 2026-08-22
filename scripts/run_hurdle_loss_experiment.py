"""Fine-tuning and evaluation for Hybrid End-to-End Hurdle-Loss Experiments (H0, H1, H2, H3)."""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.gru_sweep import SlicedMemmapDataset, get_anchor_set
from src.sequential.models import MultiTaskTransitionGRUModel
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import TRAIN_PARQUET, get_or_create_selected_users
from scripts.validate_experiment_report import validate_report_invariants


class HurdleDataset(Dataset):
    def __init__(self, raw_tensor: np.ndarray, y_log: np.ndarray, past_buyer: np.ndarray, fut_buyer: np.ndarray, seq_len: int, scaler: SequentialScaler):
        self.raw_tensor = raw_tensor
        self.y_log = y_log.astype(np.float32)
        self.past_buyer = past_buyer.astype(np.float32)
        self.fut_buyer = fut_buyer.astype(np.float32)
        self.seq_len = seq_len
        self.scaler_mean = scaler.mean[:raw_tensor.shape[-1]].astype(np.float32)
        self.scaler_std = scaler.std[:raw_tensor.shape[-1]].astype(np.float32)

    def __len__(self) -> int:
        return len(self.y_log)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_slice = self.raw_tensor[idx, -self.seq_len:, :]
        scaled = (raw_slice - self.scaler_mean) / self.scaler_std
        return (
            torch.from_numpy(scaled.astype(np.float32)),
            torch.tensor(self.y_log[idx], dtype=torch.float32),
            torch.tensor(self.past_buyer[idx], dtype=torch.float32),
            torch.tensor(self.fut_buyer[idx], dtype=torch.float32),
        )


def run_hurdle_experiment(
    variant_name: str,
    weight_final: float,
    checkpoint_base: str,
    train_anchors: List[date],
    val_anchor: date,
    data: pl.DataFrame,
    user_ids: List[int],
    epochs: int = 3,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    print(f"\n===================================================================")
    print(f"=== RUNNING HURDLE EXPERIMENT: {variant_name} (weight_final = {weight_final}) ===")
    print(f"===================================================================")

    out_dir = Path(f"artifacts/gru_hurdle_research/{variant_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(f"models/gru_hurdle_research/{variant_name}")
    models_dir.mkdir(parents=True, exist_ok=True)

    seq_len = 180
    n_users = len(user_ids)

    # 1. Prepare Scaler & Datasets
    tensors, targets_list, past_buyers, fut_buyers = [], [], [], []
    for a in train_anchors:
        t = get_cached_sequence_tensor(data, user_ids, a, seq_len=seq_len)
        y = extract_anchor_targets(data, user_ids, a)
        p_b = (t[:, -30:, 3] > 0).any(axis=1).astype(np.float32)
        f_b = (y > 0).astype(np.float32)

        tensors.append(t)
        targets_list.append(y)
        past_buyers.append(p_b)
        fut_buyers.append(f_b)

    all_tensors = np.concatenate(tensors, axis=0)
    scaler = SequentialScaler()
    scaler.fit(all_tensors)

    train_dataset = HurdleDataset(
        raw_tensor=all_tensors,
        y_log=np.log1p(np.concatenate(targets_list, axis=0)),
        past_buyer=np.concatenate(past_buyers, axis=0),
        fut_buyer=np.concatenate(fut_buyers, axis=0),
        seq_len=seq_len,
        scaler=scaler,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=1024 if device == "cuda" else 512,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=0,
    )

    # Validation Dataset
    val_tensor = get_cached_sequence_tensor(data, user_ids, val_anchor, seq_len=seq_len)
    val_targets = extract_anchor_targets(data, user_ids, val_anchor)
    val_past_buyer = (val_tensor[:, -30:, 3] > 0).any(axis=1).astype(np.float32)

    # 2. Instantiate and Load Canonical Pretrained Weights
    model = MultiTaskTransitionGRUModel(
        input_dim=15,
        hidden_dim=128,
        num_layers=2,
        dropout=0.15,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_base, map_location=device))
    print(f"[*] Loaded canonical base weights from: {checkpoint_base}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    best_val_rmsle = 999.0
    best_checkpoint = models_dir / "best.pt"

    # Evaluate Epoch 0 (Baseline Before Fine-tuning)
    def evaluate_model(mod: nn.Module) -> Tuple[pl.DataFrame, float]:
        mod.eval()
        n_val = len(val_targets)
        inf_bs = 1024
        lr_l, lc_l, zc_l, zd_l = [], [], [], []
        sc_m = scaler.mean[:15]
        sc_s = scaler.std[:15]

        with torch.no_grad():
            for i in range(0, n_val, inf_bs):
                raw_b = val_tensor[i : i + inf_bs, -seq_len:, :]
                sc_b = (raw_b - sc_m) / sc_s
                xb = torch.from_numpy(sc_b.astype(np.float32)).to(device)
                lr_t, lc_t, _, zc_t, zd_t, _ = mod(xb)

                lr_l.append(torch.sigmoid(lr_t).cpu().numpy())
                lc_l.append(torch.sigmoid(lc_t).cpu().numpy())
                zc_l.append(zc_t.cpu().numpy())
                zd_l.append(zd_t.cpu().numpy())

        p_r = np.concatenate(lr_l)
        p_c = np.concatenate(lc_l)
        z_c = np.concatenate(zc_l)
        z_d = np.concatenate(zd_l)

        p_buy = np.where(val_past_buyer == 0, p_r, 1.0 - p_c)
        z_fact = np.power(p_buy, 1.10) * z_c
        pred_rub = np.clip(np.expm1(z_fact), 0.0, None)

        val_df = pl.DataFrame({
            "user_id": user_ids,
            "anchor_date": [str(val_anchor)] * n_val,
            "y_rub": val_targets.astype(np.float64),
            "z_true": np.log1p(val_targets.astype(np.float64)),
            "current_state": val_past_buyer.astype(np.int32),
            "p_react": p_r.astype(np.float64),
            "p_churn": p_c.astype(np.float64),
            "p_buy": p_buy.astype(np.float64),
            "conditional_z": z_c.astype(np.float64),
            "factorized_z": z_fact.astype(np.float64),
            "final_prediction_z": z_fact.astype(np.float64),
            "final_prediction_rub": pred_rub.astype(np.float64),
        })

        rmsle = float(np.sqrt(np.mean((val_df["z_true"].to_numpy() - val_df["final_prediction_z"].to_numpy()) ** 2)))
        return val_df, rmsle

    init_df, init_rmsle = evaluate_model(model)
    print(f"[*] Initial Zero-Shot Validation RMSLE: {init_rmsle:.5f}")

    if weight_final == 0.0:
        # H0 is canonical benchmark (0 fine-tuning or identity check)
        best_df = init_df
        best_val_rmsle = init_rmsle
        torch.save(model.state_dict(), best_checkpoint)
    else:
        # Fine-tuning loop
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            tot_loss, tot_cls, tot_cond, tot_fin, n_b = 0.0, 0.0, 0.0, 0.0, 0

            for x, y_log, past_b, fut_b in loader:
                x, y_log, past_b, fut_b = x.to(device), y_log.to(device), past_b.to(device), fut_b.to(device)
                optimizer.zero_grad()
                lr_t, lc_t, _, zc_t, _, _ = model(x)

                # 1. Classification Losses
                mask_dormant = (past_b == 0)
                loss_r = bce_fn(lr_t[mask_dormant], fut_b[mask_dormant]) if mask_dormant.sum() > 0 else torch.tensor(0.0, device=device)

                mask_active = (past_b == 1)
                loss_c = bce_fn(lc_t[mask_active], 1.0 - fut_b[mask_active]) if mask_active.sum() > 0 else torch.tensor(0.0, device=device)
                loss_cls = loss_r + loss_c

                # 2. Conditional Regression Loss (strictly positive buyers)
                mask_buyers = (fut_b > 0.5)
                loss_cond = mse_fn(zc_t[mask_buyers], y_log[mask_buyers]) if mask_buyers.sum() > 0 else torch.tensor(0.0, device=device)

                # 3. Hybrid End-to-End Factorized Loss
                p_buy_t = torch.where(past_b == 0, torch.sigmoid(lr_t), 1.0 - torch.sigmoid(lc_t))
                z_fact_t = p_buy_t * zc_t
                loss_final = mse_fn(z_fact_t, y_log)

                loss = 0.50 * loss_cls + 1.00 * loss_cond + weight_final * loss_final
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                tot_loss += loss.item()
                tot_cls += loss_cls.item()
                tot_cond += loss_cond.item()
                tot_fin += loss_final.item()
                n_b += 1

            scheduler.step()
            ep_time = time.time() - t0
            cur_df, cur_rmsle = evaluate_model(model)
            print(f"  Epoch [{epoch}/{epochs}] ({ep_time:.1f}s) | Loss: {tot_loss/n_b:.4f} (Cls: {tot_cls/n_b:.4f}, Cond: {tot_cond/n_b:.4f}, Fin: {tot_fin/n_b:.4f}) | Val RMSLE: {cur_rmsle:.5f}")

            if cur_rmsle < best_val_rmsle:
                best_val_rmsle = cur_rmsle
                best_df = cur_df
                torch.save(model.state_dict(), best_checkpoint)

    # Save final artifacts
    pred_path = out_dir / "predictions_validation.parquet"
    best_df.write_parquet(pred_path)

    # Validate against H0 baseline
    h0_pred_path = Path("artifacts/gru_hurdle_research/H0/predictions_validation.parquet")
    base_df = pl.read_parquet(h0_pred_path) if (h0_pred_path.exists() and variant_name != "H0") else None

    val_summary = validate_report_invariants(best_df, base_df, alpha=1.10)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(val_summary, f, indent=2)

    # Record in Registry
    reg_path = Path("artifacts/gru_hurdle_research/experiment_registry.csv")
    reg_exists = reg_path.exists()

    decision = "KEEP_BEST" if (variant_name == "H0" or (val_summary["paired_comparison"] and val_summary["paired_comparison"]["delta_rmsle"] < -0.003)) else "REJECT"

    rec = {
        "experiment_id": variant_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": "HEAD",
        "config_path": f"artifacts/gru_hurdle_research/{variant_name}/config.json",
        "prediction_path": str(pred_path),
        "checkpoint_path": str(best_checkpoint),
        "seed": 42,
        "train_anchors": str([str(a) for a in train_anchors]),
        "validation_anchor": str(val_anchor),
        "RMSLE": val_summary["rmsle"],
        "MSE": val_summary["mse_log"],
        "React_AUC": val_summary["react_auc"],
        "React_Brier": val_summary["react_brier"],
        "Churn_AUC": val_summary["churn_auc"],
        "Churn_Brier": val_summary["churn_brier"],
        "arithmetic_validation": "PASSED",
        "decision": decision,
    }

    rec_df = pl.DataFrame([rec])
    if reg_exists:
        old_df = pl.read_parquet(reg_path) if reg_path.suffix == ".parquet" else pl.read_csv(reg_path)
        old_filtered = old_df.filter(pl.col("experiment_id") != variant_name)
        combined = pl.concat([old_filtered, rec_df])
        combined.write_csv(reg_path)
    else:
        rec_df.write_csv(reg_path)

    return val_summary


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


def main():
    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = get_or_create_selected_users(data, n_users=100000, seed=42)

    train_anchors = get_anchor_set("recent_14")[:-1]  # 13 train anchors
    val_anchor = date(2026, 1, 14)
    checkpoint_canonical = resolve_canonical_checkpoint()
    print(f"[+] Using canonical GRU-180 checkpoint: {checkpoint_canonical}")

    # 1. H0: Canonical Reference (0.00 final loss weight)
    h0_res = run_hurdle_experiment("H0", 0.00, checkpoint_canonical, train_anchors, val_anchor, data, user_ids)

    # 2. H1: Hybrid Loss with weight 0.10
    h1_res = run_hurdle_experiment("H1", 0.10, checkpoint_canonical, train_anchors, val_anchor, data, user_ids)

    # 3. H2: Hybrid Loss with weight 0.25
    h2_res = run_hurdle_experiment("H2", 0.25, checkpoint_canonical, train_anchors, val_anchor, data, user_ids)

    # 4. H3: Hybrid Loss with weight 0.50
    h3_res = run_hurdle_experiment("H3", 0.50, checkpoint_canonical, train_anchors, val_anchor, data, user_ids)

    print("\n" + "=" * 80)
    print("ALL HURDLE EXPERIMENTS H0-H3 COMPLETED AND VERIFIED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
