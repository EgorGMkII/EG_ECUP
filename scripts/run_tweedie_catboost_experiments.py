"""Master script for strictly purged CatBoost Tweedie vs Hurdle experiments (C0..C3)."""

from datetime import date, timedelta
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.btyd_research_pipeline import (
    extract_full_history_rfm_for_anchor,
    compute_exact_btyd_predictions,
)
from src.validation import get_snapshot_path
from lifetimes import BetaGeoFitter, GammaGammaFitter

DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots" if (DATA_DIR / "snapshots").exists() else Path("snapshots")
TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")
USERS_PARQUET = (
    Path("artifacts/selected_users_100k.parquet")
    if Path("artifacts/selected_users_100k.parquet").exists()
    else (Path("selected_users_100k.parquet") if Path("selected_users_100k.parquet").exists() else Path("artifacts/selected_users_100k.parquet"))
)
OUTPUT_ROOT = Path("artifacts/tweedie_catboost")
PLOTS_DIR = OUTPUT_ROOT / "plots"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_ANCHOR = date(2026, 1, 14)
VAL_TARGET_START = VAL_ANCHOR + timedelta(days=1)
VAL_TARGET_END = VAL_ANCHOR + timedelta(days=30)

# 11 strictly purged training anchors (max train target end <= 2026-01-14)
PURGED_TRAIN_ANCHORS = [
    date(2025, 7, 21),
    date(2025, 8, 4),
    date(2025, 8, 18),
    date(2025, 9, 1),
    date(2025, 9, 15),
    date(2025, 9, 29),
    date(2025, 10, 13),
    date(2025, 10, 27),
    date(2025, 11, 10),
    date(2025, 11, 24),
    date(2025, 12, 8),
]

def predict_tweedie_mean(catboost_model: CatBoostRegressor, eval_pool: Pool) -> np.ndarray:
    """Returns predicted mean from CatBoost Tweedie model without double-exponentiation."""
    preds = catboost_model.predict(eval_pool)
    return np.maximum(preds, 0.0)

def main():
    print("=" * 80)
    print("=== STARTING PURGED CATBOOST TWEEDIE VS HURDLE EXPERIMENT SUITE ===")
    print("=" * 80)
    t0_global = time.time()

    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()

    # -------------------------------------------------------------------------
    # 1. ANCHOR MANIFEST & STRICT OVERLAP CHECK
    # -------------------------------------------------------------------------
    manifest_rows = []
    print(f"\n[*] Validation Anchor: {VAL_ANCHOR} (Target: {VAL_TARGET_START} .. {VAL_TARGET_END})")
    print(f"{'Anchor Date':<12} | {'History Window':<23} | {'Target Window':<23} | {'Overlap Days':<15}")
    print("-" * 80)

    for a in PURGED_TRAIN_ANCHORS:
        h_start = a - timedelta(days=179)
        h_end = a
        t_start = a + timedelta(days=1)
        t_end = a + timedelta(days=30)
        overlap_start = max(t_start, VAL_TARGET_START)
        overlap_end = min(t_end, VAL_TARGET_END)
        overlap_days = max(0, (overlap_end - overlap_start).days + 1) if overlap_start <= overlap_end else 0
        assert overlap_days == 0, f"Target leakage on anchor {a}!"
        print(f"{str(a):<12} | {str(h_start)}..{str(h_end):<10} | {str(t_start)}..{str(t_end):<10} | {overlap_days:<15}")
        manifest_rows.append({
            "anchor_date": str(a),
            "history_start": str(h_start),
            "history_end": str(h_end),
            "target_start": str(t_start),
            "target_end": str(t_end),
            "overlap_days_with_validation_target": overlap_days,
            "is_validation": False,
        })

    manifest_rows.append({
        "anchor_date": str(VAL_ANCHOR),
        "history_start": str(VAL_ANCHOR - timedelta(days=179)),
        "history_end": str(VAL_ANCHOR),
        "target_start": str(VAL_TARGET_START),
        "target_end": str(VAL_TARGET_END),
        "overlap_days_with_validation_target": 0,
        "is_validation": True,
    })
    pl.DataFrame(manifest_rows).write_csv(OUTPUT_ROOT / "anchor_manifest.csv")
    print(f"[+] Saved {OUTPUT_ROOT / 'anchor_manifest.csv'}")

    # -------------------------------------------------------------------------
    # 2. FIT BTYD MODELS & EXTRACT FEATURES ON PURGED ANCHORS
    # -------------------------------------------------------------------------
    print("\n[*] Extracting RFM tables and fitting BTYD models on purged train anchors...")
    rfm_tables = {}
    for a in PURGED_TRAIN_ANCHORS + [VAL_ANCHOR]:
        rfm_tables[a] = extract_full_history_rfm_for_anchor(data, user_ids, a)

    tr_rfm_all = pl.concat([rfm_tables[a] for a in PURGED_TRAIN_ANCHORS])
    tr_avail = tr_rfm_all["btyd_available"].to_numpy() > 0
    tr_freq = tr_rfm_all["btyd_frequency"].to_numpy().astype(np.float64)
    tr_rec = tr_rfm_all["btyd_recency"].fill_null(0.0).to_numpy().astype(np.float64)
    tr_T = tr_rfm_all["btyd_T"].to_numpy().astype(np.float64)
    tr_mon = tr_rfm_all["btyd_monetary_value"].fill_null(0.0).to_numpy().astype(np.float64)

    fit_idx = np.random.choice(np.where(tr_avail)[0], size=min(50000, tr_avail.sum()), replace=False)
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(tr_freq[fit_idx], tr_rec[fit_idx], tr_T[fit_idx])

    repeat_mask = (tr_freq > 0) & (tr_mon > 0)
    gg_fit_idx = np.random.choice(np.where(repeat_mask)[0], size=min(50000, repeat_mask.sum()), replace=False)
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(tr_freq[gg_fit_idx], tr_mon[gg_fit_idx])

    btyd_tables = {}
    for a in PURGED_TRAIN_ANCHORS + [VAL_ANCHOR]:
        btyd_tables[a] = compute_exact_btyd_predictions(bgf, ggf, rfm_tables[a], t_horizons=[30])

    # Join with Snapshots
    print("\n[*] Joining snapshots with BTYD features...")
    train_dfs = []
    for a in PURGED_TRAIN_ANCHORS:
        snap_p = get_snapshot_path(a, SNAPSHOTS_DIR)
        snap_df = pl.read_parquet(snap_p)
        b_df = btyd_tables[a].select(["user_id", "btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive"])
        snap_df = snap_df.join(b_df, on="user_id", how="left")
        train_dfs.append(snap_df)
        print(f"  - Loaded purged anchor {a} (Rows: {len(snap_df)})")

    train_data = pl.concat(train_dfs)
    del train_dfs

    val_snap_p = get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR)
    val_data = pl.read_parquet(val_snap_p)
    val_b_df = btyd_tables[VAL_ANCHOR].select(["user_id", "btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive"])
    val_data = val_data.join(val_b_df, on="user_id", how="left")

    # Feature Manifest and Integrity
    btyd_cols = ["btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive"]
    non_feature_cols = {
        "user_id", "anchor_date", "target", "will_buy_30d", "target_gmv",
        "future_gmv", "z_true", "future_buy", "react_target", "churn_target",
        "event_date", "cohort", "btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive"
    }
    c4_feature_cols = [c for c in train_data.columns if c not in non_feature_cols]
    direct_feature_cols = c4_feature_cols + btyd_cols

    print(f"[*] C4 Features: {len(c4_feature_cols)} cols")
    print(f"[*] BTYD Features: {len(btyd_cols)} cols ({btyd_cols})")
    print(f"[*] Direct Features: {len(direct_feature_cols)} cols")

    # Forbidden leak check
    forbidden = ["will_buy_30d", "target", "target_gmv", "future_gmv", "z_true", "future_buy", "react_target", "churn_target"]
    for col in direct_feature_cols:
        assert col not in forbidden, f"Forbidden target column {col} in direct_feature_cols!"

    with open(OUTPUT_ROOT / "feature_manifest.json", "w") as f:
        json.dump({
            "c4_features_count": len(c4_feature_cols),
            "c4_features": c4_feature_cols,
            "btyd_features": btyd_cols,
            "direct_features_count": len(direct_feature_cols),
            "direct_features": direct_feature_cols,
        }, f, indent=2)
    print(f"[+] Saved {OUTPUT_ROOT / 'feature_manifest.json'}")

    # Build and Save Validation Transition Reference
    val_y_rub = val_data["target"].to_numpy().astype(np.float64)
    val_z_true = np.log1p(val_y_rub)
    val_past_buyer = (val_data["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)
    val_fut_buyer = (val_y_rub > 0).astype(np.int32)

    val_trans_group = np.empty(len(val_data), dtype="<U10")
    val_trans_group[(val_past_buyer == 0) & (val_fut_buyer == 0)] = "0->0"
    val_trans_group[(val_past_buyer == 0) & (val_fut_buyer == 1)] = "0->>0"
    val_trans_group[(val_past_buyer == 1) & (val_fut_buyer == 0)] = ">0->0"
    val_trans_group[(val_past_buyer == 1) & (val_fut_buyer == 1)] = ">0->>0"

    val_ref_df = pl.DataFrame({
        "user_id": val_data["user_id"].to_list(),
        "past_state": val_past_buyer,
        "future_state": val_fut_buyer,
        "transition_group": val_trans_group,
        "gmv_true": val_y_rub,
        "z_true": val_z_true,
    })
    val_ref_df.write_parquet(OUTPUT_ROOT / "validation_transition_reference.parquet")
    print(f"[+] Saved validation_transition_reference.parquet (Canonical counts: {dict(val_ref_df['transition_group'].value_counts().iter_rows())})")

    with open(OUTPUT_ROOT / "data_integrity.json", "w") as f:
        json.dump({
            "validation_N": len(val_data),
            "validation_user_id_unique": val_data["user_id"].n_unique(),
            "train_rows_total": len(train_data),
            "canonical_cohort_counts": dict(val_ref_df['transition_group'].value_counts().iter_rows()),
            "zero_leakage_guaranteed": True,
        }, f, indent=2)

    # CatBoost Base Configuration
    cb_params = {
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "task_type": "GPU",
        "verbose": 200,
        "early_stopping_rounds": 100,
    }
    with open(OUTPUT_ROOT / "canonical_config.json", "w") as f:
        json.dump(cb_params, f, indent=2)

    # -------------------------------------------------------------------------
    # PREPARE ARRAYS & POOLS
    # -------------------------------------------------------------------------
    print("\n[*] Preparing numpy matrices for training and evaluation...")
    X_train_c4 = train_data.select(c4_feature_cols).to_numpy().astype(np.float32)
    X_val_c4 = val_data.select(c4_feature_cols).to_numpy().astype(np.float32)

    X_train_direct = train_data.select(direct_feature_cols).to_numpy().astype(np.float32)
    X_val_direct = val_data.select(direct_feature_cols).to_numpy().astype(np.float32)

    train_y_rub = train_data["target"].to_numpy().astype(np.float64)
    train_z = np.log1p(train_y_rub)
    train_past_buyer = (train_data["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)
    train_fut_buyer = (train_y_rub > 0).astype(np.int32)

    val_pool_direct = Pool(X_val_direct)
    val_pool_c4 = Pool(X_val_c4)

    # Scaling Factors for Tweedie
    raw_scale = float(train_y_rub.max())
    y_train_scaled = (train_y_rub / raw_scale).astype(np.float32)
    y_val_scaled = (val_y_rub / raw_scale).astype(np.float32)

    log_scale = float(train_z.max())
    z_train_scaled = (train_z / log_scale).astype(np.float32)
    z_val_scaled = (val_z_true / log_scale).astype(np.float32)

    print(f"[*] Scaling factors: Raw Scale = {raw_scale:.2f}, Log Scale = {log_scale:.5f}")

    registry_rows = []
    predictions_dict = {}

    def evaluate_and_record(
        exp_id: str,
        target_space: str,
        loss_name: str,
        var_power: Any,
        z_pred: np.ndarray,
        gmv_pred: np.ndarray,
        pred_scaled: np.ndarray,
        duration: float,
        extra_preds: Dict[str, np.ndarray] = None,
    ):
        exp_dir = OUTPUT_ROOT / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        diff_sq = (z_pred - val_z_true) ** 2
        total_mse = float(np.mean(diff_sq))
        solo_rmsle = float(np.sqrt(total_mse))

        # Transition metrics
        m00 = (val_past_buyer == 0) & (val_fut_buyer == 0)
        m01 = (val_past_buyer == 0) & (val_fut_buyer == 1)
        m10 = (val_past_buyer == 1) & (val_fut_buyer == 0)
        m11 = (val_past_buyer == 1) & (val_fut_buyer == 1)

        mse00 = float(np.mean(diff_sq[m00]))
        mse01 = float(np.mean(diff_sq[m01]))
        mse10 = float(np.mean(diff_sq[m10]))
        mse11 = float(np.mean(diff_sq[m11]))

        sse00 = float(np.sum(diff_sq[m00]))
        sse01 = float(np.sum(diff_sq[m01]))
        sse10 = float(np.sum(diff_sq[m10]))
        sse11 = float(np.sum(diff_sq[m11]))
        total_sse = float(np.sum(diff_sq))

        # Verification of arithmetic invariant
        weighted_mse = (len(val_data.filter(m00))*mse00 + len(val_data.filter(m01))*mse01 + len(val_data.filter(m10))*mse10 + len(val_data.filter(m11))*mse11) / len(val_data)
        assert abs(total_mse - weighted_mse) < 1e-10, "Transition MSE arithmetic mismatch!"
        assert abs(total_sse - (sse00 + sse01 + sse10 + sse11)) < 1e-6, "Transition SSE sum mismatch!"

        # Sub-metrics
        pos_mask = val_fut_buyer == 1
        zero_mask = val_fut_buyer == 0
        pos_rmsle = float(np.sqrt(np.mean(diff_sq[pos_mask])))
        mean_pred_zero = float(np.mean(gmv_pred[zero_mask]))
        mean_pred_pos = float(np.mean(gmv_pred[pos_mask]))
        bias_log = float(np.mean(z_pred - val_z_true))

        # Distribution percentiles
        p01, p10, p50, p90, p99, pmax = np.percentile(gmv_pred, [1, 10, 50, 90, 99, 100])
        share_lt_1 = float(np.mean(gmv_pred < 1.0))
        share_gt_100 = float(np.mean(gmv_pred > 100.0))
        share_gt_1000 = float(np.mean(gmv_pred > 1000.0))

        # Ranking metrics
        buy_auc = float(roc_auc_score(val_fut_buyer, gmv_pred))
        buy_pr_auc = float(auc(*precision_recall_curve(val_fut_buyer, gmv_pred)[1::-1]))

        react_auc = float(roc_auc_score(val_fut_buyer[val_past_buyer == 0], gmv_pred[val_past_buyer == 0]))
        react_pr_auc = float(auc(*precision_recall_curve(val_fut_buyer[val_past_buyer == 0], gmv_pred[val_past_buyer == 0])[1::-1]))

        churn_auc = float(roc_auc_score((1 - val_fut_buyer)[val_past_buyer == 1], -gmv_pred[val_past_buyer == 1]))
        churn_pr_auc = float(auc(*precision_recall_curve((1 - val_fut_buyer)[val_past_buyer == 1], -gmv_pred[val_past_buyer == 1])[1::-1]))

        # Save Parquet predictions
        pred_dict = {
            "user_id": val_data["user_id"].to_list(),
            "anchor_date": ["2026-01-14"] * len(val_data),
            "gmv_true": val_y_rub,
            "z_true": val_z_true,
            "past_state": val_past_buyer,
            "future_state": val_fut_buyer,
            "transition_group": val_trans_group,
            "prediction_scaled": pred_scaled,
            "z_pred": z_pred,
            "prediction_rub": gmv_pred,
        }
        if extra_preds:
            pred_dict.update(extra_preds)

        pred_df = pl.DataFrame(pred_dict)
        pred_df.write_parquet(exp_dir / "validation_predictions.parquet")
        predictions_dict[exp_id] = pred_df

        # Save metrics json
        metrics = {
            "RMSLE": solo_rmsle,
            "MSE_log": total_mse,
            "Total_SSE": total_sse,
            "Positive_RMSLE": pos_rmsle,
            "Bias_log": bias_log,
            "Mean_prediction_rub": float(np.mean(gmv_pred)),
            "Mean_prediction_actual_zero": mean_pred_zero,
            "Mean_prediction_actual_pos": mean_pred_pos,
            "Percentiles_rub": {"p01": p01, "p10": p10, "p50": p50, "p90": p90, "p99": p99, "max": pmax},
            "Shares": {"lt_1rub": share_lt_1, "gt_100rub": share_gt_100, "gt_1000rub": share_gt_1000},
            "Ranking_AUC": {"Buy_ROC_AUC": buy_auc, "Buy_PR_AUC": buy_pr_auc, "React_ROC_AUC": react_auc, "React_PR_AUC": react_pr_auc, "Churn_ROC_AUC": churn_auc, "Churn_PR_AUC": churn_pr_auc},
            "Transition_MSE": {"0->0": mse00, "0->>0": mse01, ">0->0": mse10, ">0->>0": mse11},
            "Transition_SSE": {"0->0": sse00, "0->>0": sse01, ">0->0": sse10, ">0->>0": sse11},
            "Duration_s": duration,
        }
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        reg_row = {
            "experiment_id": exp_id,
            "target_space": target_space,
            "loss_function": loss_name,
            "variance_power": var_power if var_power is not None else "",
            "seed": 42,
            "RMSLE": solo_rmsle,
            "MSE_log": total_mse,
            "Positive_RMSLE": pos_rmsle,
            "MSE_0_to_0": mse00,
            "MSE_0_to_pos": mse01,
            "MSE_pos_to_0": mse10,
            "MSE_pos_to_pos": mse11,
            "Mean_pred_zero": mean_pred_zero,
            "Mean_pred_pos": mean_pred_pos,
            "Buy_ROC_AUC": buy_auc,
            "Duration_s": duration,
        }
        registry_rows.append(reg_row)
        print(f"\n[+] {exp_id} Complete | RMSLE: {solo_rmsle:.5f} | Pos RMSLE: {pos_rmsle:.5f} | 0->0 MSE: {mse00:.4f} | 0->>0 MSE: {mse01:.4f} | >0->0 MSE: {mse10:.4f} | >0->>0 MSE: {mse11:.4f}")

    # =========================================================================
    # EXPERIMENT C0: HONEST PURGED CATBOOST HURDLE B1
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== EXPERIMENT C0: HONEST PURGED CATBOOST HURDLE B1 (BASELINE) ===")
    print("=" * 80)
    t0 = time.time()

    # 1. Reactivation Classifier (dormant past_buyer == 0)
    mask_dormant_tr = (train_past_buyer == 0)
    pool_react_tr = Pool(X_train_direct[mask_dormant_tr], train_fut_buyer[mask_dormant_tr])
    clf_react = CatBoostClassifier(**cb_params, loss_function="Logloss", eval_metric="Logloss")
    clf_react.fit(pool_react_tr)

    # 2. Churn Classifier (active past_buyer == 1)
    mask_active_tr = (train_past_buyer == 1)
    pool_churn_tr = Pool(X_train_direct[mask_active_tr], (1 - train_fut_buyer[mask_active_tr]))
    clf_churn = CatBoostClassifier(**cb_params, loss_function="Logloss", eval_metric="Logloss")
    clf_churn.fit(pool_churn_tr)

    # 3. Conditional Positive GMV Regressor (only C4 features, buyers only)
    mask_buyers_tr = (train_fut_buyer == 1)
    pool_reg_tr = Pool(X_train_c4[mask_buyers_tr], train_z[mask_buyers_tr])
    reg_cond = CatBoostRegressor(**cb_params, loss_function="RMSE", eval_metric="RMSE")
    reg_cond.fit(pool_reg_tr)

    # Validation Predictions
    p_react = clf_react.predict_proba(val_pool_direct)[:, 1]
    p_churn = clf_churn.predict_proba(val_pool_direct)[:, 1]
    z_cond = reg_cond.predict(val_pool_c4)
    z_cond = np.maximum(z_cond, 0.0)

    p_buy = np.where(val_past_buyer == 0, p_react, 1.0 - p_churn)
    p_buy = np.clip(p_buy, 1e-7, 1.0 - 1e-7)

    alpha = 1.10
    z_fact_c0 = np.power(p_buy, alpha) * z_cond
    gmv_pred_c0 = np.clip(np.expm1(z_fact_c0), 0.0, None)

    dur_c0 = time.time() - t0
    evaluate_and_record(
        exp_id="C0_Purged_Hurdle_B1",
        target_space="Factorized (Hurdle)",
        loss_name="Hurdle (Logloss + RMSE)",
        var_power=None,
        z_pred=z_fact_c0,
        gmv_pred=gmv_pred_c0,
        pred_scaled=z_fact_c0,
        duration=dur_c0,
        extra_preds={"p_reactivation": p_react, "p_churn": p_churn, "p_buy": p_buy, "conditional_z": z_cond, "factorized_z": z_fact_c0},
    )

    # =========================================================================
    # EXPERIMENT C1: DIRECT CATBOOST LOG-RMSE
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== EXPERIMENT C1: DIRECT CATBOOST LOG-RMSE (424 FEATURES) ===")
    print("=" * 80)
    t0 = time.time()

    pool_c1_tr = Pool(X_train_direct, train_z)
    model_c1 = CatBoostRegressor(**cb_params, loss_function="RMSE", eval_metric="RMSE")
    model_c1.fit(pool_c1_tr)

    z_pred_c1 = model_c1.predict(val_pool_direct)
    z_pred_c1 = np.maximum(z_pred_c1, 0.0)
    gmv_pred_c1 = np.expm1(z_pred_c1)

    dur_c1 = time.time() - t0
    evaluate_and_record(
        exp_id="C1_Direct_LogRMSE",
        target_space="log1p(GMV)",
        loss_name="RMSE",
        var_power=None,
        z_pred=z_pred_c1,
        gmv_pred=gmv_pred_c1,
        pred_scaled=z_pred_c1,
        duration=dur_c1,
    )

    # =========================================================================
    # EXPERIMENT C2: TWEEDIE ON RAW GMV (p=1.5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== EXPERIMENT C2: TWEEDIE ON RAW GMV (p=1.5, Scaled to [0,1]) ===")
    print("=" * 80)
    t0 = time.time()

    pool_c2_tr = Pool(X_train_direct, y_train_scaled)
    model_c2 = CatBoostRegressor(**cb_params, loss_function="Tweedie:variance_power=1.5", eval_metric="Tweedie:variance_power=1.5")
    model_c2.fit(pool_c2_tr)

    pred_scaled_c2 = predict_tweedie_mean(model_c2, val_pool_direct)
    gmv_pred_c2 = pred_scaled_c2 * raw_scale
    gmv_pred_c2 = np.maximum(gmv_pred_c2, 0.0)
    z_pred_c2 = np.log1p(gmv_pred_c2)

    dur_c2 = time.time() - t0
    evaluate_and_record(
        exp_id="C2_Raw_Tweedie_p1.5",
        target_space="Raw GMV",
        loss_name="Tweedie",
        var_power=1.5,
        z_pred=z_pred_c2,
        gmv_pred=gmv_pred_c2,
        pred_scaled=pred_scaled_c2,
        duration=dur_c2,
    )

    # =========================================================================
    # EXPERIMENT C3: TWEEDIE ON LOG1P(GMV) (p=1.5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("=== EXPERIMENT C3: TWEEDIE ON LOG1P(GMV) (p=1.5, Scaled to [0,1]) ===")
    print("=" * 80)
    t0 = time.time()

    pool_c3_tr = Pool(X_train_direct, z_train_scaled)
    model_c3 = CatBoostRegressor(**cb_params, loss_function="Tweedie:variance_power=1.5", eval_metric="Tweedie:variance_power=1.5")
    model_c3.fit(pool_c3_tr)

    pred_scaled_c3 = predict_tweedie_mean(model_c3, val_pool_direct)
    z_pred_c3 = pred_scaled_c3 * log_scale
    z_pred_c3 = np.maximum(z_pred_c3, 0.0)
    gmv_pred_c3 = np.expm1(z_pred_c3)

    dur_c3 = time.time() - t0
    evaluate_and_record(
        exp_id="C3_Log_Tweedie_p1.5",
        target_space="log1p(GMV)",
        loss_name="Tweedie",
        var_power=1.5,
        z_pred=z_pred_c3,
        gmv_pred=gmv_pred_c3,
        pred_scaled=pred_scaled_c3,
        duration=dur_c3,
    )

    # -------------------------------------------------------------------------
    # ERROR CORRELATION AND LIMITED BLENDS
    # -------------------------------------------------------------------------
    c0_z = predictions_dict["C0_Purged_Hurdle_B1"]["z_pred"].to_numpy()
    c0_err = c0_z - val_z_true
    c0_rmsle = registry_rows[0]["RMSLE"]

    for row in registry_rows:
        row["Delta_vs_C0"] = row["RMSLE"] - c0_rmsle
        exp_z = predictions_dict[row["experiment_id"]]["z_pred"].to_numpy()
        exp_err = exp_z - val_z_true
        corr = float(np.corrcoef(c0_err, exp_err)[0, 1])
        row["Error_corr_with_C0"] = corr

    pl.DataFrame(registry_rows).write_csv(OUTPUT_ROOT / "experiment_registry.csv")
    print(f"\n[+] Saved {OUTPUT_ROOT / 'experiment_registry.csv'}")

    # Blends with C0
    blend_rows = []
    for exp_id in ["C1_Direct_LogRMSE", "C2_Raw_Tweedie_p1.5", "C3_Log_Tweedie_p1.5"]:
        exp_z = predictions_dict[exp_id]["z_pred"].to_numpy()
        for w in [0.25, 0.50, 0.75]:
            z_blend = w * c0_z + (1 - w) * exp_z
            b_rmsle = float(np.sqrt(np.mean((z_blend - val_z_true) ** 2)))
            blend_rows.append({
                "Blend_Name": f"{w:.2f}*C0 + {1-w:.2f}*{exp_id}",
                "Weight_C0": w,
                "Model_2": exp_id,
                "RMSLE": b_rmsle,
                "Delta_vs_C0": b_rmsle - c0_rmsle,
            })
    pl.DataFrame(blend_rows).write_csv(OUTPUT_ROOT / "blend_summary.csv")
    print(f"[+] Saved {OUTPUT_ROOT / 'blend_summary.csv'}")

    # -------------------------------------------------------------------------
    # GENERATE COMPLETE DIAGNOSTIC PLOTS (SECTION 15)
    # -------------------------------------------------------------------------
    print("\n[*] Generating diagnostic visual plots...")

    # Plot 1: Prediction vs Target in Log-space
    plt.figure(figsize=(12, 10), dpi=150)
    for i, (eid, col) in enumerate([("C0_Purged_Hurdle_B1", "#1E88E5"), ("C1_Direct_LogRMSE", "#43A047"), ("C2_Raw_Tweedie_p1.5", "#FB8C00"), ("C3_Log_Tweedie_p1.5", "#8E24AA")], 1):
        plt.subplot(2, 2, i)
        pz = predictions_dict[eid]["z_pred"].to_numpy()
        idx_sample = np.random.choice(len(val_z_true), 10000, replace=False)
        plt.scatter(val_z_true[idx_sample], pz[idx_sample], alpha=0.12, s=8, color=col)
        plt.plot([0, 12], [0, 12], "r--", linewidth=1.5)
        plt.title(f"{eid} (RMSLE: {registry_rows[i-1]['RMSLE']:.4f})", fontweight="bold")
        plt.xlabel("Ground Truth ln(1 + GMV)")
        plt.ylabel("Predicted ln(1 + GMV)")
        plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_vs_target_log.png")
    plt.close()

    # Plot 2: Transition MSE Decomposition
    plt.figure(figsize=(10, 6), dpi=150)
    t_labels = ["0 -> 0 (Dormant)", "0 -> >0 (Reactivation)", ">0 -> 0 (Churn)", ">0 -> >0 (Retention)"]
    x = np.arange(len(t_labels))
    w = 0.2
    for i, (eid, col) in enumerate([("C0_Purged_Hurdle_B1", "#1E88E5"), ("C1_Direct_LogRMSE", "#43A047"), ("C2_Raw_Tweedie_p1.5", "#FB8C00"), ("C3_Log_Tweedie_p1.5", "#8E24AA")]):
        row = registry_rows[i]
        mses = [row["MSE_0_to_0"], row["MSE_0_to_pos"], row["MSE_pos_to_0"], row["MSE_pos_to_pos"]]
        plt.bar(x + (i - 1.5) * w, mses, w, label=f"{eid}", color=col, alpha=0.85)
    plt.ylabel("MSE on ln(1 + GMV)")
    plt.title("Transition Error Decomposition: C0 vs C1 vs C2 vs C3 (Purged 11 Anchors)", fontweight="bold")
    plt.xticks(x, t_labels)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "transition_mse_decomposition.png")
    plt.close()

    # Plot 3: Actual Zero vs Actual Positive Prediction Distributions
    plt.figure(figsize=(12, 5), dpi=150)
    plt.subplot(1, 2, 1)
    for eid, col in [("C0_Purged_Hurdle_B1", "#1E88E5"), ("C1_Direct_LogRMSE", "#43A047"), ("C2_Raw_Tweedie_p1.5", "#FB8C00"), ("C3_Log_Tweedie_p1.5", "#8E24AA")]:
        pz = predictions_dict[eid]["z_pred"].to_numpy()
        plt.hist(pz[val_fut_buyer == 0], bins=50, density=True, histtype="step", linewidth=2, label=eid, color=col)
    plt.title("Prediction Distribution on ACTUAL ZEROS (y=0)", fontweight="bold")
    plt.xlabel("Predicted ln(1 + GMV)")
    plt.ylabel("Density")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    for eid, col in [("C0_Purged_Hurdle_B1", "#1E88E5"), ("C1_Direct_LogRMSE", "#43A047"), ("C2_Raw_Tweedie_p1.5", "#FB8C00"), ("C3_Log_Tweedie_p1.5", "#8E24AA")]:
        pz = predictions_dict[eid]["z_pred"].to_numpy()
        plt.hist(pz[val_fut_buyer == 1], bins=50, density=True, histtype="step", linewidth=2, label=eid, color=col)
    plt.title("Prediction Distribution on ACTUAL POSITIVES (y>0)", fontweight="bold")
    plt.xlabel("Predicted ln(1 + GMV)")
    plt.ylabel("Density")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_zero_vs_pos_distributions.png")
    plt.close()

    print("[+] All diagnostic plots successfully generated in artifacts/tweedie_catboost/plots/")
    print(f"\n[+] Total pipeline finished in {time.time() - t0_global:.1f}s")


if __name__ == "__main__":
    main()
