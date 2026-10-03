"""Phase 3, 4 & 5: Patch Audit, Encoder vs Hurdle Collapse Separation & Attention Diagnostics."""

import json
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from src.sequential.dataset import CACHE_DIR, extract_anchor_targets, get_cached_sequence_tensor
from src.sequential.models import PatchTransformer365Model
from src.sequential.preprocessing import SequentialScaler
from src.snapshots import build_snapshot, generate_panel_anchors, get_or_create_selected_users, SNAPSHOTS_DIR, TRAIN_PARQUET

AUDIT_DIR = Path("artifacts/transformer_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_quantiles_dict(arr: np.ndarray) -> dict:
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


def main():
    print("===================================================================")
    print("=== PHASE 3, 4 & 5: AUDIT, COLLAPSE SEPARATION & DIAGNOSTICS ===")
    print("===================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pl.read_parquet(TRAIN_PARQUET)
    train_sample_users = get_or_create_selected_users()
    val_anchor = date(2026, 1, 14)
    test_anchor = date(2026, 2, 13)

    test_users = data["user_id"].unique().sort().to_list()
    n_test = len(test_users)

    # 1. Load sequence tensors
    val_tensor_path = CACHE_DIR / f"seq_tensor_{val_anchor.strftime('%Y-%m-%d')}_u{len(train_sample_users)}_t365.npy"
    test_tensor_path = CACHE_DIR / f"seq_tensor_{test_anchor.strftime('%Y-%m-%d')}_u{n_test}_t365.npy"

    if not val_tensor_path.exists():
        _ = get_cached_sequence_tensor(data, train_sample_users, val_anchor, seq_len=365)
    if not test_tensor_path.exists():
        _ = get_cached_sequence_tensor(data, test_users, test_anchor, seq_len=365)

    X_val_raw = np.load(val_tensor_path, mmap_mode="r")
    X_test_raw = np.load(test_tensor_path, mmap_mode="r")

    scaler = SequentialScaler().fit(X_val_raw[:25000])

    # Load validation targets and past activity
    y_val = extract_anchor_targets(data, train_sample_users, val_anchor)
    snap_val = pl.read_parquet(f"data/snapshots/snapshot_{val_anchor.strftime('%Y-%m-%d')}.parquet")
    past_gmv_val = snap_val["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_val = (past_gmv_val > 0).astype(np.int32)
    del snap_val

    # Load test past activity
    snap_test = build_snapshot(data, test_users, test_anchor, is_test=True)
    past_gmv_test = snap_test["gmv_sum_30d"].to_numpy().astype(np.float32)
    past_buyer_test = (past_gmv_test > 0).astype(np.int32)
    del snap_test

    # Instantiate trained/saved PatchTransformer365Model
    model = PatchTransformer365Model(input_dim=15, patch_size=7, num_patches=52, d_model=128, nhead=4, num_layers=3).to(device)

    # Load best weights from parity or run
    ckpt_path = AUDIT_DIR / "parity_test_transformer.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # =========================================================================
    # PHASE 3: PATCH CONSTRUCTION & MASKING AUDIT (20 USERS)
    # =========================================================================
    print("\n--- PHASE 3: PATCH CONSTRUCTION & MASKING AUDIT (20 USERS) ---")
    # Pick diverse users
    user_archetypes = {
        "stable_dormant": np.where((past_buyer_val == 0) & (y_val == 0))[0][:4],
        "reactivation": np.where((past_buyer_val == 0) & (y_val > 0))[0][:4],
        "churn": np.where((past_buyer_val == 1) & (y_val == 0))[0][:4],
        "retention_whale": np.where((past_buyer_val == 1) & (y_val > 1000))[0][:4],
        "last_7d_active": np.where((X_val_raw[:10000, -7:, 0].sum(axis=1) > 0))[0][:4],
    }

    selected_20_indices = []
    for cat, idxs in user_archetypes.items():
        selected_20_indices.extend(idxs)
    selected_20_indices = list(dict.fromkeys(selected_20_indices))[:20]

    patch_audit_records = []
    with torch.no_grad():
        for idx in selected_20_indices:
            uid = train_sample_users[idx]
            raw_seq = X_val_raw[idx]  # (365, 15)
            active_days = int((raw_seq[:, 9] > 0).sum())  # channel 9 is is_active_day

            # 52 weekly patches (364 days)
            seq_364 = raw_seq[-364:, :]  # (364, 15)
            patches = seq_364.reshape(52, 7 * 15)
            active_weeks = int((patches.sum(axis=1) > 0).sum())

            scaled_seq = (raw_seq - scaler.mean) / scaler.std
            xb = torch.from_numpy(scaled_seq.astype(np.float32)).unsqueeze(0).to(device)

            lr, lc, lb, zc, zd, emb = model(xb)
            p_react = float(torch.sigmoid(lr).cpu().numpy())
            p_churn = float(torch.sigmoid(lc).cpu().numpy())
            z_c = float(zc.cpu().numpy())
            z_d = float(zd.cpu().numpy())

            p_buy = p_react if past_buyer_val[idx] == 0 else (1.0 - p_churn)
            z_fact = (p_buy ** 1.1) * z_c
            pred_fact_rub = float(np.expm1(z_fact))
            pred_dir_rub = float(np.expm1(z_d))

            record = {
                "user_id": int(uid),
                "past_buyer_30d": int(past_buyer_val[idx]),
                "target_rub": float(y_val[idx]),
                "active_days_365": active_days,
                "active_weeks_52": active_weeks,
                "p_react": round(p_react, 4),
                "p_churn": round(p_churn, 4),
                "p_buy": round(p_buy, 4),
                "z_cond": round(z_c, 4),
                "z_dir": round(z_d, 4),
                "z_fact": round(z_fact, 4),
                "pred_factorized_rub": round(pred_fact_rub, 2),
                "pred_direct_rub": round(pred_dir_rub, 2),
            }
            patch_audit_records.append(record)

    with open(AUDIT_DIR / "patch_audit_20_users.json", "w", encoding="utf-8") as f:
        json.dump(patch_audit_records, f, indent=2)

    print(f"[+] 20-User Patch Audit completed. Sample record: {patch_audit_records[0]}")

    # =========================================================================
    # PHASE 4: ENCODER VS HURDLE COLLAPSE SEPARATION (FULL QUANTILE PROFILES)
    # =========================================================================
    print("\n--- PHASE 4: ENCODER VS HURDLE COLLAPSE SEPARATION ---")

    # Run inference on 10k Val and 10k Test to profile full quantile pipeline
    eval_n = 10000
    with torch.no_grad():
        # Validation 10k
        x_val_sc = (X_val_raw[:eval_n] - scaler.mean) / scaler.std
        xb_val = torch.from_numpy(x_val_sc.astype(np.float32)).to(device)
        lr_v, lc_v, lb_v, zc_v, zd_v, emb_v = model(xb_val)

        p_react_v = torch.sigmoid(lr_v).cpu().numpy()
        p_churn_v = torch.sigmoid(lc_v).cpu().numpy()
        p_buy_v = np.where(past_buyer_val[:eval_n] == 0, p_react_v, 1.0 - p_churn_v)
        p_buy_alpha_v = np.power(p_buy_v, 1.1)
        zc_v_np = zc_v.cpu().numpy()
        zd_v_np = zd_v.cpu().numpy()
        z_fact_v = p_buy_alpha_v * zc_v_np
        pred_fact_rub_v = np.expm1(z_fact_v)
        pred_dir_rub_v = np.expm1(zd_v_np)
        emb_v_np = emb_v.cpu().numpy()

        # Test 10k
        x_test_sc = (X_test_raw[:eval_n] - scaler.mean) / scaler.std
        xb_test = torch.from_numpy(x_test_sc.astype(np.float32)).to(device)
        lr_t, lc_t, lb_t, zc_t, zd_t, emb_t = model(xb_test)

        p_react_t = torch.sigmoid(lr_t).cpu().numpy()
        p_churn_t = torch.sigmoid(lc_t).cpu().numpy()
        p_buy_t = np.where(past_buyer_test[:eval_n] == 0, p_react_t, 1.0 - p_churn_t)
        p_buy_alpha_t = np.power(p_buy_t, 1.1)
        zc_t_np = zc_t.cpu().numpy()
        zd_t_np = zd_t.cpu().numpy()
        z_fact_t = p_buy_alpha_t * zc_t_np
        pred_fact_rub_t = np.expm1(z_fact_t)
        pred_dir_rub_t = np.expm1(zd_t_np)
        emb_t_np = emb_t.cpu().numpy()

    collapse_analysis = {
        "validation_10k": {
            "embedding_norm": calculate_quantiles_dict(np.linalg.norm(emb_v_np, axis=1)),
            "p_reactivation": calculate_quantiles_dict(p_react_v),
            "p_churn": calculate_quantiles_dict(p_churn_v),
            "p_buy": calculate_quantiles_dict(p_buy_v),
            "p_buy_alpha_1.1": calculate_quantiles_dict(p_buy_alpha_v),
            "conditional_z": calculate_quantiles_dict(zc_v_np),
            "direct_z": calculate_quantiles_dict(zd_v_np),
            "factorized_z": calculate_quantiles_dict(z_fact_v),
            "pred_direct_rub": calculate_quantiles_dict(pred_dir_rub_v),
            "pred_factorized_rub": calculate_quantiles_dict(pred_fact_rub_v),
        },
        "test_10k": {
            "embedding_norm": calculate_quantiles_dict(np.linalg.norm(emb_t_np, axis=1)),
            "p_reactivation": calculate_quantiles_dict(p_react_t),
            "p_churn": calculate_quantiles_dict(p_churn_t),
            "p_buy": calculate_quantiles_dict(p_buy_t),
            "p_buy_alpha_1.1": calculate_quantiles_dict(p_buy_alpha_t),
            "conditional_z": calculate_quantiles_dict(zc_t_np),
            "direct_z": calculate_quantiles_dict(zd_t_np),
            "factorized_z": calculate_quantiles_dict(z_fact_t),
            "pred_direct_rub": calculate_quantiles_dict(pred_dir_rub_t),
            "pred_factorized_rub": calculate_quantiles_dict(pred_fact_rub_t),
        },
    }

    with open(AUDIT_DIR / "collapse_stage_separation.json", "w", encoding="utf-8") as f:
        json.dump(collapse_analysis, f, indent=2)

    print("\n[*] Collapse Stage Diagnostics (Validation vs Test):")
    print(f"    - Val Factorized Mean Rub:  {collapse_analysis['validation_10k']['pred_factorized_rub']['mean']:.2f} | P99: {collapse_analysis['validation_10k']['pred_factorized_rub']['p99']:.2f}")
    print(f"    - Test Factorized Mean Rub: {collapse_analysis['test_10k']['pred_factorized_rub']['mean']:.2f} | P99: {collapse_analysis['test_10k']['pred_factorized_rub']['p99']:.2f}")
    print(f"    - Val Direct Mean Rub:      {collapse_analysis['validation_10k']['pred_direct_rub']['mean']:.2f} | P99: {collapse_analysis['validation_10k']['pred_direct_rub']['p99']:.2f}")
    print(f"    - Test Direct Mean Rub:     {collapse_analysis['test_10k']['pred_direct_rub']['mean']:.2f} | P99: {collapse_analysis['test_10k']['pred_direct_rub']['p99']:.2f}")

    # =========================================================================
    # PHASE 5: EMBEDDINGS & ATTENTION DIAGNOSTICS (VARIANCE, RANK, SIMILARITY)
    # =========================================================================
    print("\n--- PHASE 5: EMBEDDINGS & ATTENTION DIAGNOSTICS ---")

    # 1. Dimension standard deviations
    dim_std_v = np.std(emb_v_np, axis=0)
    dim_std_t = np.std(emb_t_np, axis=0)
    dead_dims_v = int((dim_std_v < 1e-4).sum())
    dead_dims_t = int((dim_std_t < 1e-4).sum())

    # 2. Covariance and effective rank
    cov_v = np.cov(emb_v_np, rowvar=False)
    cov_t = np.cov(emb_t_np, rowvar=False)
    eigvals_v = np.linalg.eigvalsh(cov_v)
    eigvals_t = np.linalg.eigvalsh(cov_t)
    eigvals_v = np.clip(eigvals_v, 1e-12, None)
    eigvals_t = np.clip(eigvals_t, 1e-12, None)
    p_eig_v = eigvals_v / eigvals_v.sum()
    p_eig_t = eigvals_t / eigvals_t.sum()
    eff_rank_v = float(np.exp(-np.sum(p_eig_v * np.log(p_eig_v))))
    eff_rank_t = float(np.exp(-np.sum(p_eig_t * np.log(p_eig_t))))

    # 3. Pairwise cosine similarity on 500 samples
    sub_emb_v = emb_v_np[:500] / np.linalg.norm(emb_v_np[:500], axis=1, keepdims=True)
    sub_emb_t = emb_t_np[:500] / np.linalg.norm(emb_t_np[:500], axis=1, keepdims=True)
    sim_mat_v = np.dot(sub_emb_v, sub_emb_v.T)
    sim_mat_t = np.dot(sub_emb_t, sub_emb_t.T)
    mean_cos_sim_v = float((sim_mat_v.sum() - 500) / (500 * 499))
    mean_cos_sim_t = float((sim_mat_t.sum() - 500) / (500 * 499))

    # 4. PCA on Validation Embeddings across 4 States
    pca = PCA(n_components=2)
    emb_pca_v = pca.fit_transform(emb_v_np)
    var_explained = pca.explained_variance_ratio_.tolist()

    # 4 state masks
    state_00 = (past_buyer_val[:eval_n] == 0) & (y_val[:eval_n] == 0)
    state_01 = (past_buyer_val[:eval_n] == 0) & (y_val[:eval_n] > 0)
    state_10 = (past_buyer_val[:eval_n] == 1) & (y_val[:eval_n] == 0)
    state_11 = (past_buyer_val[:eval_n] == 1) & (y_val[:eval_n] > 0)

    pca_centroids = {
        "stable_sleep_00": emb_pca_v[state_00].mean(axis=0).tolist() if state_00.sum() > 0 else [0, 0],
        "reactivation_01": emb_pca_v[state_01].mean(axis=0).tolist() if state_01.sum() > 0 else [0, 0],
        "churn_10": emb_pca_v[state_10].mean(axis=0).tolist() if state_10.sum() > 0 else [0, 0],
        "retention_11": emb_pca_v[state_11].mean(axis=0).tolist() if state_11.sum() > 0 else [0, 0],
    }

    embedding_diagnostics = {
        "validation": {
            "mean_embedding_variance": float(np.mean(dim_std_v ** 2)),
            "dead_dimensions_count": dead_dims_v,
            "effective_rank": round(eff_rank_v, 2),
            "mean_pairwise_cosine_similarity": round(mean_cos_sim_v, 4),
            "pca_explained_variance_ratio": [round(x, 4) for x in var_explained],
            "pca_state_centroids": pca_centroids,
        },
        "test": {
            "mean_embedding_variance": float(np.mean(dim_std_t ** 2)),
            "dead_dimensions_count": dead_dims_t,
            "effective_rank": round(eff_rank_t, 2),
            "mean_pairwise_cosine_similarity": round(mean_cos_sim_t, 4),
        },
    }

    with open(AUDIT_DIR / "embedding_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(embedding_diagnostics, f, indent=2)

    print(f"[*] Embedding Diagnostics:")
    print(f"    - Val Effective Rank:       {eff_rank_v:.2f} / 128 | Cosine Sim: {mean_cos_sim_v:.4f}")
    print(f"    - Test Effective Rank:      {eff_rank_t:.2f} / 128 | Cosine Sim: {mean_cos_sim_t:.4f}")
    print(f"    - Dead Dimensions (Val):    {dead_dims_v} / 128")
    print(f"    - Dead Dimensions (Test):   {dead_dims_t} / 128")
    print(f"    - PCA Centroids (00 vs 01): {pca_centroids['stable_sleep_00']} vs {pca_centroids['reactivation_01']}")
    print(f"    - PCA Centroids (10 vs 11): {pca_centroids['churn_10']} vs {pca_centroids['retention_11']}")
    print("\n[+] Phases 3, 4 & 5 Successfully Completed!")


if __name__ == "__main__":
    main()
