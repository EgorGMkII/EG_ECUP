"""Experiment B: Multi-Task MLP with Masked Lifecycle Losses vs CatBoost Transitions."""

import gc
import json
import time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader

from src.hurdle import get_feature_columns
from src.snapshots import generate_panel_anchors, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.transitions.features import compute_all_transition_features
from src.transitions.inference import compute_factorized_gmv
from src.transitions.metrics import decompose_mse_by_transitions, evaluate_classifier_metrics
from src.transitions.mlp import MultiTaskDataset, MultiTaskMLP, train_multitask_mlp
from src.validation import get_snapshot_path

TRANSITIONS_ARTIFACTS = Path("artifacts/transitions")
TRANSITIONS_ARTIFACTS.mkdir(parents=True, exist_ok=True)
VAL_ANCHOR = date(2026, 1, 14)


def main():
    print("===================================================================")
    print("=== EXPERIMENT B: PYTORCH MULTI-TASK LIFECYCLE MLP ===")
    print("===================================================================")

    data = pl.read_parquet(TRAIN_PARQUET)
    anchors = generate_panel_anchors()
    purge_cutoff = VAL_ANCHOR - timedelta(days=30)
    train_anchors = [a for a in anchors if a <= purge_cutoff][-8:]

    # 1. Load validation snapshot
    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    user_ids = val_snap["user_id"].to_list()
    y_val = val_snap["target"].to_numpy().astype(np.float32)
    past_gmv_val = val_snap["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_val = (past_gmv_val > 0).astype(np.int32)
    fut_buyer_val = (y_val > 0).astype(np.int32)

    mask_dormant_val = (past_buyer_val == 0)
    mask_active_val = (past_buyer_val == 1)
    y_react_val = fut_buyer_val[mask_dormant_val]
    y_churn_val = (1 - fut_buyer_val)[mask_active_val]

    # Feature columns
    all_old_cols = get_feature_columns(val_snap)
    noisy_cols = [c for c in all_old_cols if "global_dau" in c or "global_gmv_per_active" in c or "global_buyer_rate" in c or "vs_global" in c]
    old_feat_cols = [c for c in all_old_cols if c not in noisy_cols]

    # Compute NEW Transition Features for validation
    val_trans_feats_df = compute_all_transition_features(data, user_ids, VAL_ANCHOR)
    val_trans_cols = [c for c in val_trans_feats_df.columns if c != "user_id"]

    # Compute Transition Features for training anchors
    train_trans_dfs = {}
    for a in train_anchors:
        train_trans_dfs[a] = compute_all_transition_features(data, user_ids, a)

    # 2. Build Training Matrices
    X_tr_list, y_tr_list, past_buyer_tr_list = [], [], []
    for a in train_anchors:
        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        X_old = snap_a.select(old_feat_cols).to_numpy().astype(np.float32)
        X_trans = train_trans_dfs[a].select(val_trans_cols).to_numpy().astype(np.float32)
        X_all = np.hstack([X_old, X_trans])

        X_tr_list.append(X_all)
        y_tr_list.append(snap_a["target"].to_numpy().astype(np.float32))
        past_buyer_tr_list.append((snap_a["gmv_sum_30d"].to_numpy().astype(np.float32) > 0).astype(np.int32))
        del snap_a

    X_tr = np.vstack(X_tr_list)
    y_tr = np.concatenate(y_tr_list)
    past_buyer_tr = np.concatenate(past_buyer_tr_list)
    del X_tr_list, y_tr_list, past_buyer_tr_list
    gc.collect()

    X_val_old = val_snap.select(old_feat_cols).to_numpy().astype(np.float32)
    X_val_trans = val_trans_feats_df.select(val_trans_cols).to_numpy().astype(np.float32)
    X_val = np.hstack([X_val_old, X_val_trans])

    # 3. Robust Scaling on Tabular Features for Neural Network
    print(f"[*] Fitting RobustScaler on {X_tr.shape[1]} combined tabular features...")
    # Clean infinities or extreme outliers
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1000.0, neginf=-1000.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1000.0, neginf=-1000.0)

    scaler = RobustScaler(unit_variance=True).fit(X_tr[-100000:])
    X_tr_scaled = scaler.transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # 4. PyTorch Multi-Task DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training PyTorch Multi-Task MLP on {device}...")

    train_ds = MultiTaskDataset(X_tr_scaled, y_tr, past_buyer_tr)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=True, num_workers=0)

    mlp_model = MultiTaskMLP(input_dim=X_tr_scaled.shape[1], hidden_dim=512, embed_dim=256, dropout=0.2).to(device)

    train_multitask_mlp(
        mlp_model,
        train_loader,
        epochs=12,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        verbose=True,
    )

    # 5. Validation Inference
    mlp_model.eval()
    logit_react_list, logit_churn_list, logit_buy_list, pred_cond_list, pred_dir_list = [], [], [], [], []

    with torch.no_grad():
        for i in range(0, len(X_val_scaled), 2048):
            xb = torch.from_numpy(X_val_scaled[i : i + 2048]).float().to(device)
            l_r, l_c, l_b, p_c, p_d = mlp_model(xb)
            logit_react_list.append(l_r.cpu().numpy())
            logit_churn_list.append(l_c.cpu().numpy())
            logit_buy_list.append(l_b.cpu().numpy())
            pred_cond_list.append(p_c.cpu().numpy())
            pred_dir_list.append(p_d.cpu().numpy())

    logit_react = np.concatenate(logit_react_list)
    logit_churn = np.concatenate(logit_churn_list)
    logit_buy = np.concatenate(logit_buy_list)
    pred_cond = np.concatenate(pred_cond_list)
    pred_dir = np.concatenate(pred_dir_list)

    # Convert logits to probabilities via sigmoid
    p_react_mlp = 1.0 / (1.0 + np.exp(-logit_react[mask_dormant_val]))
    p_churn_mlp = 1.0 / (1.0 + np.exp(-logit_churn[mask_active_val]))
    p_buy_direct_mlp = 1.0 / (1.0 + np.exp(-logit_buy))

    # Assemble factorized probability
    p_buy_factorized_mlp = np.zeros(len(user_ids), dtype=np.float32)
    p_buy_factorized_mlp[mask_dormant_val] = p_react_mlp
    p_buy_factorized_mlp[mask_active_val] = 1.0 - p_churn_mlp

    # Multi-Task Direct and Factorized GMV
    z_mlp_direct = pred_dir
    y_pred_mlp_direct = np.clip(np.expm1(z_mlp_direct), 0.0, None)

    z_mlp_fact, y_pred_mlp_fact = compute_factorized_gmv(p_buy_factorized_mlp, pred_cond, power_p=1.1)

    # 6. Evaluation Metrics
    react_m_mlp = evaluate_classifier_metrics(y_react_val, p_react_mlp, "MLP Reactivation")
    churn_m_mlp = evaluate_classifier_metrics(y_churn_val, p_churn_mlp, "MLP Churn")
    decomp_mlp_dir = decompose_mse_by_transitions(y_val, y_pred_mlp_direct, past_buyer_val)
    decomp_mlp_fact = decompose_mse_by_transitions(y_val, y_pred_mlp_fact, past_buyer_val)

    # 7. Probability Blending: CatBoost A3 + MLP
    exp_a_df = pl.read_parquet(TRANSITIONS_ARTIFACTS / "experiment_A_predictions.parquet")
    p_buy_cb = exp_a_df["p_buy_factorized_a3"].to_numpy().astype(np.float32)
    p_react_cb = exp_a_df["p_reactivation_a3"].to_numpy().astype(np.float32)[mask_dormant_val]
    p_churn_cb = exp_a_df["p_churn_a3"].to_numpy().astype(np.float32)[mask_active_val]

    # Blend 50/50
    p_react_blend = 0.50 * p_react_cb + 0.50 * p_react_mlp
    p_churn_blend = 0.50 * p_churn_cb + 0.50 * p_churn_mlp
    p_buy_blend = 0.50 * p_buy_cb + 0.50 * p_buy_factorized_mlp

    react_m_blend = evaluate_classifier_metrics(y_react_val, p_react_blend, "Ensemble Reactivation")
    churn_m_blend = evaluate_classifier_metrics(y_churn_val, p_churn_blend, "Ensemble Churn")

    print("\n===================================================================")
    print("=== EXPERIMENT B: CLASSIFICATION & REGRESSION COMPARISON ===")
    print("===================================================================")
    comp_df = pl.DataFrame({
        "Model": [
            "CatBoost Transitions (A3)",
            "PyTorch Multi-Task MLP (Factorized)",
            "PyTorch Multi-Task MLP (Direct)",
            "Ensemble Probabilities (CatBoost + MLP)",
        ],
        "RMSLE": [
            1.72014,
            decomp_mlp_fact["total_rmsle"],
            decomp_mlp_dir["total_rmsle"],
            float(np.sqrt(np.mean(((0.5*z_mlp_fact + 0.5*exp_a_df['z_factorized_a3'].to_numpy()) - np.log1p(y_val))**2))),
        ],
        "Reactivation_AUC": [
            0.7541,
            react_m_mlp["roc_auc"],
            react_m_mlp["roc_auc"],
            react_m_blend["roc_auc"],
        ],
        "Churn_AUC": [
            0.7969,
            churn_m_mlp["roc_auc"],
            churn_m_mlp["roc_auc"],
            churn_m_blend["roc_auc"],
        ],
        "Reactivation_Brier": [
            0.1698,
            react_m_mlp["brier_score"],
            react_m_mlp["brier_score"],
            react_m_blend["brier_score"],
        ],
        "Churn_Brier": [
            0.1534,
            churn_m_mlp["brier_score"],
            churn_m_mlp["brier_score"],
            churn_m_blend["brier_score"],
        ],
    })
    print(comp_df)

    # Save Experiment B predictions
    pred_mlp_df = pl.DataFrame({
        "user_id": user_ids,
        "past_buyer_30d": past_buyer_val,
        "target": y_val,
        "p_buy_mlp_factorized": p_buy_factorized_mlp,
        "p_buy_mlp_direct": p_buy_direct_mlp,
        "z_mlp_direct": z_mlp_direct,
        "z_mlp_factorized": z_mlp_fact,
        "y_pred_mlp_direct": y_pred_mlp_direct,
        "y_pred_mlp_factorized": y_pred_mlp_fact,
    })
    pred_mlp_df.write_parquet(TRANSITIONS_ARTIFACTS / "experiment_B_predictions.parquet")
    print(f"\n[+] Saved Experiment B predictions to {TRANSITIONS_ARTIFACTS / 'experiment_B_predictions.parquet'}")


if __name__ == "__main__":
    main()
