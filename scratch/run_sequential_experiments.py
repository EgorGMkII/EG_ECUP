"""Comprehensive Sequential Deep Learning Experiments Suite (Direct GRU, Multi-Task GRU, Embeddings, Blending)."""

import gc
import json
import time
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from catboost import CatBoostRegressor

from src.sequential.dataset import extract_anchor_targets, get_cached_sequence_tensor, OzonSequenceDataset
from src.sequential.embeddings import extract_embeddings_dataframe
from src.sequential.models import DirectGRUModel, MultiTaskGRUModel
from src.sequential.preprocessing import SequentialScaler
from src.sequential.trainer import train_direct_gru, train_multitask_gru
from src.sequential.validation import evaluate_state_transitions, run_purged_sequential_backtest, BACKTEST_ANCHORS
from src.snapshots import generate_panel_anchors, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.validation import get_snapshot_path
from src.hurdle import get_feature_columns

ARTIFACTS_DIR = Path("artifacts/sequential_oof")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = Path("artifacts/sequential_embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("===================================================================")
    print("=== SEQUENTIAL DEEP LEARNING EXPERIMENTS SUITE (PYTORCH + GPU) ===")
    print("===================================================================")

    # 1. Load users subset
    users_100k_df = pl.read_parquet("artifacts/selected_users_100k.parquet")
    users_100k = users_100k_df["user_id"].to_list()
    users_50k = users_100k[:50_000]

    print(f"[*] Loaded {len(users_100k):,} users for comprehensive evaluation")
    data = pl.read_parquet(TRAIN_PARQUET)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: DIRECT GRU BASELINE (50k pilot + 4 Full Regimes)
    # -------------------------------------------------------------------------
    print("\n===================================================================")
    print("=== 1. EXPERIMENT 1: DIRECT GRU BASELINE ===")
    print("===================================================================")

    # Pilot on 50k users (2026-01-14)
    print("\n[*] 1.1. Pilot Run: Direct GRU on 50k users (2026-01-14)...")
    res_pilot = run_purged_sequential_backtest(
        val_anchor=date(2026, 1, 14),
        user_ids=users_50k,
        data=data,
        model_type="direct",
        hidden_dim=128,
        epochs=12,
        batch_size=512,
    )
    print(f"[+] Pilot 50k Direct GRU RMSLE = {res_pilot['rmsle']:.5f}")

    # Full 4-Regime Backtest for Direct GRU
    print("\n[*] 1.2. Full 100k Users Direct GRU across 4 Purged Backtests...")
    direct_results = {}
    for anchor in BACKTEST_ANCHORS:
        res = run_purged_sequential_backtest(
            val_anchor=anchor,
            user_ids=users_100k,
            data=data,
            model_type="direct",
            hidden_dim=128,
            epochs=12,
            batch_size=512,
        )
        direct_results[str(anchor)] = res

    direct_mean_rmsle = float(np.mean([r["rmsle"] for r in direct_results.values()]))
    print(f"\n[+] DIRECT GRU 4-REGIME MEAN RMSLE = {direct_mean_rmsle:.5f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: MULTI-TASK HURDLE GRU (Classification + Conditional Reg)
    # -------------------------------------------------------------------------
    print("\n===================================================================")
    print("=== 2. EXPERIMENT 2: MULTI-TASK HURDLE GRU ===")
    print("===================================================================")
    multitask_results = {}
    for anchor in BACKTEST_ANCHORS:
        res = run_purged_sequential_backtest(
            val_anchor=anchor,
            user_ids=users_100k,
            data=data,
            model_type="multitask",
            hidden_dim=128,
            epochs=12,
            batch_size=512,
        )
        multitask_results[str(anchor)] = res

    multitask_mean_rmsle = float(np.mean([r["rmsle"] for r in multitask_results.values()]))
    print(f"\n[+] MULTI-TASK HURDLE GRU 4-REGIME MEAN RMSLE = {multitask_mean_rmsle:.5f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: ERROR CORRELATION & BLENDING (CatBoost vs GRU)
    # -------------------------------------------------------------------------
    print("\n===================================================================")
    print("=== 3. EXPERIMENT 3: CORRELATION & OOF BLENDING (CatBoost + GRU) ===")
    print("===================================================================")

    # Load CatBoost predictions on post-NY regime (2026-01-14)
    catboost_val_file = Path("artifacts/val_predictions_cv3.parquet")
    cb_df = pl.read_parquet(catboost_val_file)

    cb_pred_log = np.log1p(cb_df["pred_ensemble"].to_numpy())
    gru_direct_pred_log = direct_results["2026-01-14"]["z_pred"]
    gru_mt_pred_log = multitask_results["2026-01-14"]["z_pred"]
    y_true_log = np.log1p(cb_df["target"].to_numpy())

    # Error vectors:
    err_catboost = cb_pred_log - y_true_log
    err_gru_direct = gru_direct_pred_log - y_true_log
    err_gru_mt = gru_mt_pred_log - y_true_log

    # Pearson & Spearman correlations of predictions and errors
    r_pred_cb_gru, _ = pearsonr(cb_pred_log, gru_mt_pred_log)
    r_err_cb_gru, _ = pearsonr(err_catboost, err_gru_mt)
    spearman_err, _ = spearmanr(err_catboost, err_gru_mt)

    print(f"[*] Prediction Pearson Correlation (CatBoost vs MultiTask GRU): {r_pred_cb_gru:.4f}")
    print(f"[*] Error Pearson Correlation (CatBoost vs MultiTask GRU):      {r_err_cb_gru:.4f}")
    print(f"[*] Error Spearman Correlation (CatBoost vs MultiTask GRU):     {spearman_err:.4f}")

    # Grid search for blending weights: z_blend = (1-w)*z_CatBoost + w*z_GRU
    blend_records = []
    for w in np.linspace(0.0, 1.0, 21):
        z_blend = (1.0 - w) * cb_pred_log + w * gru_mt_pred_log
        rmsle = float(np.sqrt(np.mean((z_blend - y_true_log) ** 2)))
        blend_records.append({"w_gru": float(w), "w_catboost": float(1.0 - w), "rmsle": rmsle})

    df_blend = pl.DataFrame(blend_records)
    best_blend = df_blend.sort("rmsle").head(1)
    print("\n[+] Blending Grid Search Results (CatBoost + MultiTask GRU):")
    print(df_blend)
    print(f"\n[+] BEST BLEND: w_CatBoost = {best_blend['w_catboost'][0]:.2f}, w_GRU = {best_blend['w_gru'][0]:.2f} -> RMSLE = {best_blend['rmsle'][0]:.5f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: EMBEDDINGS EXTRACTION & CATBOOST STACKING
    # -------------------------------------------------------------------------
    print("\n===================================================================")
    print("=== 4. EXPERIMENT 4: GRU EMBEDDINGS EXTRACTION & CATBOOST STACKING ===")
    print("===================================================================")

    val_anchor = date(2026, 1, 14)
    emb_df = pl.DataFrame({"user_id": users_100k})
    embs_mt = multitask_results["2026-01-14"]["embeddings"]
    for c in range(embs_mt.shape[1]):
        emb_df = emb_df.with_columns(pl.Series(f"seq_emb_{c:03d}", embs_mt[:, c]))

    emb_path = EMB_DIR / "gru_embeddings_2026-01-14.parquet"
    emb_df.write_parquet(emb_path)
    print(f"[+] Saved 128d GRU embeddings to {emb_path}")

    # Save summary report to JSON
    summary_report = {
        "direct_gru_4regimes": {k: v["rmsle"] for k, v in direct_results.items()},
        "direct_gru_mean_rmsle": direct_mean_rmsle,
        "multitask_gru_4regimes": {k: v["rmsle"] for k, v in multitask_results.items()},
        "multitask_gru_mean_rmsle": multitask_mean_rmsle,
        "correlation": {
            "prediction_pearson": float(r_pred_cb_gru),
            "error_pearson": float(r_err_cb_gru),
            "error_spearman": float(spearman_err),
        },
        "best_blend": {
            "w_catboost": float(best_blend["w_catboost"][0]),
            "w_gru": float(best_blend["w_gru"][0]),
            "rmsle": float(best_blend["rmsle"][0]),
        },
        "state_transitions": multitask_results["2026-01-14"]["state_eval"],
    }

    with open(ARTIFACTS_DIR / "sequential_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n[+] ALL SEQUENTIAL EXPERIMENTS SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    main()
