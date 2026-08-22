"""Master Execution Script for BTYD Research and CatBoost Integration.

Performs:
1. Full-history RFM feature extraction (2025-01-01 to anchor_date).
2. Reference lifetimes model fitting (BetaGeoFitter, GammaGammaFitter).
3. Detailed feature distribution analysis (Overall & by frequency groups).
4. Standalone Diagnostics (T0: Probability, T1: Count, T2: GMV).
5. CatBoost Incremental Experiments (T3: Count/Prob, T4: Monetary, T5: Full BTYD).
6. Strict invariant checks and export of artifacts to artifacts/btyd_research/.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter
from scipy.stats import spearmanr
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    roc_auc_score,
    root_mean_squared_error,
)
from catboost import CatBoostClassifier, CatBoostRegressor

from src.btyd_research_pipeline import (
    extract_full_history_rfm_for_anchor,
    compute_exact_btyd_predictions,
)
from src.cadence_features import extract_cadence_features_for_anchor
from src.personal_propensity_features import compute_personal_propensity_features
from src.snapshots import generate_panel_anchors
from src.validation import get_snapshot_path
from scripts.validate_experiment_report import validate_report_invariants

DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots" if (DATA_DIR / "snapshots").exists() else Path("snapshots")
TRAIN_PARQUET = DATA_DIR / "train.parquet" if (DATA_DIR / "train.parquet").exists() else Path("train.parquet")
USERS_PARQUET = (
    Path("artifacts/selected_users_100k.parquet")
    if Path("artifacts/selected_users_100k.parquet").exists()
    else (Path("selected_users_100k.parquet") if Path("selected_users_100k.parquet").exists() else Path("artifacts/selected_users_100k.parquet"))
)
OUTPUT_DIR = Path("artifacts/btyd_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_ANCHOR = date(2026, 1, 14)


def main():
    print("=" * 80)
    print("=== STARTING COMPREHENSIVE BTYD RESEARCH PIPELINE (FULL HISTORY) ===")
    print("=" * 80)
    t0_start = time.time()

    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()
    anchors = generate_panel_anchors()
    purge_cutoff = VAL_ANCHOR - timedelta(days=30)
    train_anchors = [a for a in anchors if a <= purge_cutoff][-8:]

    print(f"[*] Total validation users: {len(user_ids)}")
    print(f"[*] Training on 8 panel anchors: {[str(a) for a in train_anchors]}")
    print(f"[*] Validation anchor: {VAL_ANCHOR}")

    # 1. Full-History RFM Extraction across anchors
    print("\n[*] Extracting Full-History RFM tables across anchors (2025-01-01 to anchor)...")
    rfm_tables = {}
    for a in train_anchors + [VAL_ANCHOR]:
        t_a = time.time()
        rfm_df = extract_full_history_rfm_for_anchor(data, user_ids, a)
        rfm_tables[a] = rfm_df
        print(f"  - Extracted RFM for {a} in {time.time() - t_a:.1f}s (Buyers: {(rfm_df['btyd_available'] > 0).sum()}/{len(user_ids)})")

    # 2. Fit Reference lifetimes models on training anchors
    print("\n[*] Fitting Reference BetaGeoFitter and GammaGammaFitter on Training Data...")
    train_rfm_list = [rfm_tables[a] for a in train_anchors]
    tr_rfm_all = pl.concat(train_rfm_list)

    tr_avail = tr_rfm_all["btyd_available"].to_numpy() > 0
    tr_freq = tr_rfm_all["btyd_frequency"].to_numpy().astype(np.float64)
    tr_rec = tr_rfm_all["btyd_recency"].fill_null(0.0).to_numpy().astype(np.float64)
    tr_T = tr_rfm_all["btyd_T"].to_numpy().astype(np.float64)
    tr_mon = tr_rfm_all["btyd_monetary_value"].fill_null(0.0).to_numpy().astype(np.float64)

    # Subsample for robust MLE fitting
    fit_mask = tr_avail
    if fit_mask.sum() > 50000:
        fit_idx = np.random.choice(np.where(fit_mask)[0], size=50000, replace=False)
    else:
        fit_idx = np.where(fit_mask)[0]

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(tr_freq[fit_idx], tr_rec[fit_idx], tr_T[fit_idx])
    print("[+] BetaGeoFitter Parameters:")
    print(f"  r = {bgf.params_['r']:.4f}, alpha = {bgf.params_['alpha']:.4f}, a = {bgf.params_['a']:.4f}, b = {bgf.params_['b']:.4f}")

    # Gamma-Gamma on repeat buyers strictly
    repeat_mask = (tr_freq > 0) & (tr_mon > 0)
    if repeat_mask.sum() > 50000:
        gg_fit_idx = np.random.choice(np.where(repeat_mask)[0], size=50000, replace=False)
    else:
        gg_fit_idx = np.where(repeat_mask)[0]

    corr_freq_mon = float(np.corrcoef(tr_freq[gg_fit_idx], tr_mon[gg_fit_idx])[0, 1])
    print(f"  Diagnostic correlation (frequency vs monetary): {corr_freq_mon:.4f}")

    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(tr_freq[gg_fit_idx], tr_mon[gg_fit_idx])
    print("[+] GammaGammaFitter Parameters:")
    print(f"  p = {ggf.params_['p']:.4f}, q = {ggf.params_['q']:.4f}, v = {ggf.params_['v']:.4f}")

    # 3. Compute BTYD predictions across all anchors
    print("\n[*] Generating complete BTYD feature sets for all anchors...")
    btyd_tables = {}
    for a in train_anchors + [VAL_ANCHOR]:
        btyd_df = compute_exact_btyd_predictions(bgf, ggf, rfm_tables[a], t_horizons=[7, 14, 30])
        btyd_tables[a] = btyd_df

    val_btyd_df = btyd_tables[VAL_ANCHOR]
    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    y_val = val_snap["target"].to_numpy().astype(np.float64)
    y_val_log = np.log1p(y_val)
    fut_buyer_val = (y_val > 0).astype(np.int32)
    past_buyer_val = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)

    # -------------------------------------------------------------------------
    # SECTION 12: FEATURE DISTRIBUTIONS (OVERALL & FREQUENCY GROUPS)
    # -------------------------------------------------------------------------
    print("\n[*] Calculating Feature Distributions (Section 12)...")
    btyd_feature_cols = [c for c in val_btyd_df.columns if c.startswith("btyd_") or c.startswith("gamma_")]
    feat_dists = {}
    for col in btyd_feature_cols:
        vals = val_btyd_df[col].to_numpy().astype(np.float64)
        fin_vals = vals[np.isfinite(vals)]
        feat_dists[col] = {
            "min": float(np.min(fin_vals)) if len(fin_vals) > 0 else 0.0,
            "p01": float(np.percentile(fin_vals, 1)) if len(fin_vals) > 0 else 0.0,
            "p10": float(np.percentile(fin_vals, 10)) if len(fin_vals) > 0 else 0.0,
            "p50": float(np.percentile(fin_vals, 50)) if len(fin_vals) > 0 else 0.0,
            "p90": float(np.percentile(fin_vals, 90)) if len(fin_vals) > 0 else 0.0,
            "p99": float(np.percentile(fin_vals, 99)) if len(fin_vals) > 0 else 0.0,
            "max": float(np.max(fin_vals)) if len(fin_vals) > 0 else 0.0,
            "mean": float(np.mean(fin_vals)) if len(fin_vals) > 0 else 0.0,
            "std": float(np.std(fin_vals)) if len(fin_vals) > 0 else 0.0,
            "missing_rate": float(val_btyd_df[col].is_null().mean()),
            "finite_rate": float(np.isfinite(vals).mean()),
        }
    with open(OUTPUT_DIR / "btyd_feature_distributions.json", "w") as f:
        json.dump(feat_dists, f, indent=2)

    # Breakdown by frequency groups
    freq_arr = val_btyd_df["btyd_frequency"].fill_null(-1).to_numpy()
    groups_def = {
        "0_orders (Non-buyers)": freq_arr == -1,
        "1_order (freq=0)": freq_arr == 0,
        "2_orders (freq=1)": freq_arr == 1,
        "3-5_orders (freq 2-4)": (freq_arr >= 2) & (freq_arr <= 4),
        "6+_orders (freq >= 5)": freq_arr >= 5,
    }

    freq_group_stats = {}
    for g_name, g_mask in groups_def.items():
        sub_p_alive = val_btyd_df["btyd_p_alive"].to_numpy()[g_mask]
        sub_p_buy = val_btyd_df["btyd_p_buy_30d"].to_numpy()[g_mask]
        sub_exp_cnt = val_btyd_df["btyd_expected_purchases_30d"].to_numpy()[g_mask]
        sub_actual_buy = fut_buyer_val[g_mask]
        freq_group_stats[g_name] = {
            "N": int(g_mask.sum()),
            "share": float(g_mask.mean()),
            "mean_p_alive": float(np.mean(sub_p_alive)),
            "mean_p_buy_30d": float(np.mean(sub_p_buy)),
            "mean_exp_purchases_30d": float(np.mean(sub_exp_cnt)),
            "actual_buy_rate": float(np.mean(sub_actual_buy)),
        }
    with open(OUTPUT_DIR / "btyd_frequency_group_distributions.json", "w") as f:
        json.dump(freq_group_stats, f, indent=2)

    # -------------------------------------------------------------------------
    # SECTION 14: STANDALONE EVALUATIONS (T0, T1, T2)
    # -------------------------------------------------------------------------
    print("\n[*] Evaluating Standalone BTYD Models (T0, T1, T2)...")
    
    # T0: Purchase Probability
    p_buy_val = val_btyd_df["btyd_p_buy_30d"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64)
    t0_roc_auc = float(roc_auc_score(fut_buyer_val, p_buy_val))
    prec, rec, _ = precision_recall_curve(fut_buyer_val, p_buy_val)
    t0_pr_auc = float(auc(rec, prec))
    t0_brier = float(brier_score_loss(fut_buyer_val, p_buy_val))
    t0_logloss = float(log_loss(fut_buyer_val, np.clip(p_buy_val, 1e-7, 1.0 - 1e-7)))

    # For available users only
    avail_mask = val_btyd_df["btyd_available"].to_numpy() > 0
    t0_avail_auc = float(roc_auc_score(fut_buyer_val[avail_mask], p_buy_val[avail_mask]))
    t0_avail_brier = float(brier_score_loss(fut_buyer_val[avail_mask], p_buy_val[avail_mask]))

    t0_metrics = {
        "overall_roc_auc": t0_roc_auc,
        "overall_pr_auc": t0_pr_auc,
        "overall_brier": t0_brier,
        "overall_logloss": t0_logloss,
        "available_users_roc_auc": t0_avail_auc,
        "available_users_brier": t0_avail_brier,
        "mean_predicted_p_buy": float(np.mean(p_buy_val)),
        "actual_buy_rate": float(np.mean(fut_buyer_val)),
    }
    with open(OUTPUT_DIR / "T0_standalone_probability_metrics.json", "w") as f:
        json.dump(t0_metrics, f, indent=2)
    print(f"[+] T0 Probability Standalone: ROC-AUC={t0_roc_auc:.4f}, PR-AUC={t0_pr_auc:.4f}, Brier={t0_brier:.4f}")

    # T1: Expected Transactions (count of actual future order days in target window)
    # We estimate actual purchase days in next 30d from future ground truth
    exp_cnt_val = val_btyd_df["btyd_expected_purchases_30d"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64)
    # Future actual buy indicator / rough transaction proxy
    t1_mae = float(mean_absolute_error(fut_buyer_val, exp_cnt_val))
    t1_rmse = float(root_mean_squared_error(fut_buyer_val, exp_cnt_val))
    t1_spearman, _ = spearmanr(exp_cnt_val, fut_buyer_val)

    t1_metrics = {
        "mae": t1_mae,
        "rmse": t1_rmse,
        "spearman_correlation": float(t1_spearman),
    }
    with open(OUTPUT_DIR / "T1_standalone_count_metrics.json", "w") as f:
        json.dump(t1_metrics, f, indent=2)
    print(f"[+] T1 Expected Purchases Standalone: MAE={t1_mae:.4f}, RMSE={t1_rmse:.4f}, Spearman={t1_spearman:.4f}")

    # T2: Expected GMV
    exp_gmv_val = val_btyd_df["btyd_expected_gmv_30d"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64)
    t2_raw_rmsle = float(root_mean_squared_error(y_val_log, np.log1p(np.maximum(0.0, exp_gmv_val))))

    # Fit calibration on train anchors
    tr_btyd_all = pl.concat([btyd_tables[a] for a in train_anchors])
    tr_y_list = [pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))["target"].to_numpy() for a in train_anchors]
    tr_y_all = np.concatenate(tr_y_list)
    tr_exp_gmv_log = np.log1p(tr_btyd_all["btyd_expected_gmv_30d"].to_numpy().astype(np.float64))

    A_tr = np.column_stack([tr_exp_gmv_log, np.ones_like(tr_exp_gmv_log)])
    cal_coeffs, _, _, _ = np.linalg.lstsq(A_tr, np.log1p(tr_y_all), rcond=None)
    t2_cal_z = np.clip(cal_coeffs[0] * np.log1p(exp_gmv_val) + cal_coeffs[1], 0.0, None)
    t2_cal_rmsle = float(root_mean_squared_error(y_val_log, t2_cal_z))

    t2_metrics = {
        "raw_btyd_rmsle": t2_raw_rmsle,
        "calibrated_btyd_rmsle": t2_cal_rmsle,
        "calibration_scale": float(cal_coeffs[0]),
        "calibration_intercept": float(cal_coeffs[1]),
    }
    with open(OUTPUT_DIR / "T2_standalone_gmv_metrics.json", "w") as f:
        json.dump(t2_metrics, f, indent=2)
    print(f"[+] T2 Expected GMV Standalone: Raw RMSLE={t2_raw_rmsle:.4f}, Calibrated RMSLE={t2_cal_rmsle:.4f}")

    # -------------------------------------------------------------------------
    # SECTION 15: INCREMENTAL CATBOOST EXPERIMENTS (T3, T4, T5)
    # -------------------------------------------------------------------------
    print("\n[*] Running Incremental CatBoost Experiments (T3, T4, T5)...")
    
    # Load base features from C4 / C2
    base_feat_cols = [c for c in val_snap.columns if c not in ["user_id", "anchor_date", "target", "current_state", "global_dau", "vs_global_orders_30d", "vs_global_orders_7d", "vs_global_gmv_30d", "vs_global_gmv_7d"]]
    
    # Precompute Cadence and Propensity for all anchors
    cadence_tables = {}
    propensity_tables = {}
    for a in train_anchors + [VAL_ANCHOR]:
        cadence_tables[a] = extract_cadence_features_for_anchor(data, user_ids, a)
        propensity_tables[a] = compute_personal_propensity_features(data, user_ids, a, anchors)

    cad_cols = [c for c in cadence_tables[VAL_ANCHOR].columns if c != "user_id"]
    prop_cols = [c for c in propensity_tables[VAL_ANCHOR].columns if c != "user_id"]

    # Assemble Baseline matrices (Base + Cadence + Propensity = C4)
    X_tr_list = []
    y_tr_list = []
    for a in train_anchors:
        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        feat_base = snap_a.select(base_feat_cols).to_numpy().astype(np.float32)
        feat_cad = cadence_tables[a].select(cad_cols).to_numpy().astype(np.float32)
        feat_prop = propensity_tables[a].select(prop_cols).to_numpy().astype(np.float32)
        X_tr_list.append(np.hstack([feat_base, feat_cad, feat_prop]))
        y_tr_list.append(snap_a["target"].to_numpy().astype(np.float64))
        del snap_a

    X_tr_c4 = np.vstack(X_tr_list)
    y_tr = np.concatenate(y_tr_list)
    
    X_val_c4 = np.hstack([
        val_snap.select(base_feat_cols).to_numpy().astype(np.float32),
        cadence_tables[VAL_ANCHOR].select(cad_cols).to_numpy().astype(np.float32),
        propensity_tables[VAL_ANCHOR].select(prop_cols).to_numpy().astype(np.float32),
    ])

    # BTYD subsets:
    btyd_prob_count_cols = [
        "btyd_p_alive", "btyd_logit_p_alive", "btyd_p_buy_30d", "btyd_p_zero_30d",
        "btyd_expected_purchases_30d", "btyd_frequency", "btyd_recency", "btyd_T", "btyd_available"
    ]
    btyd_monetary_cols = [
        "btyd_expected_monetary_value", "btyd_expected_gmv_30d", "btyd_log_expected_gmv_30d", "gamma_gamma_available"
    ]
    btyd_all_cols = btyd_prob_count_cols + btyd_monetary_cols

    def get_btyd_matrix(cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tr_m = np.vstack([btyd_tables[a].select(cols).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32) for a in train_anchors])
        val_m = btyd_tables[VAL_ANCHOR].select(cols).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
        return tr_m, val_m

    X_tr_prob, X_val_prob = get_btyd_matrix(btyd_prob_count_cols)
    X_tr_mon, X_val_mon = get_btyd_matrix(btyd_monetary_cols)
    X_tr_all_btyd, X_val_all_btyd = get_btyd_matrix(btyd_all_cols)

    experiments = {
        "Baseline_C4": (X_tr_c4, X_val_c4),
        "T3_CatBoost_BTYD_ProbCount": (np.hstack([X_tr_c4, X_tr_prob]), np.hstack([X_val_c4, X_val_prob])),
        "T4_CatBoost_BTYD_Monetary": (np.hstack([X_tr_c4, X_tr_mon]), np.hstack([X_val_c4, X_val_mon])),
        "T5_CatBoost_BTYD_Full": (np.hstack([X_tr_c4, X_tr_all_btyd]), np.hstack([X_val_c4, X_val_all_btyd])),
    }

    inc_results = []
    base_pred_df = None

    for exp_name, (X_tr_exp, X_val_exp) in experiments.items():
        print(f"\n[*] Training {exp_name} ({X_tr_exp.shape[1]} features)...")
        t_exp = time.time()
        
        # 1. Classifier
        clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.06,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=-1,
            verbose=False,
        )
        clf.fit(X_tr_exp, (y_tr > 0).astype(np.int32))
        p_val_exp = clf.predict_proba(X_val_exp)[:, 1]

        # 2. Regressor
        buyer_mask = y_tr > 0
        reg = CatBoostRegressor(
            iterations=500,
            learning_rate=0.06,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            thread_count=-1,
            verbose=False,
        )
        reg.fit(X_tr_exp[buyer_mask], np.log1p(y_tr[buyer_mask]))
        z_cond_exp = reg.predict(X_val_exp)

        z_fact_exp = (np.power(p_val_exp, 1.10) * z_cond_exp).astype(np.float64)
        pred_rub_exp = np.clip(np.expm1(z_fact_exp), 0.0, None)

        pred_df_exp = pl.DataFrame({
            "user_id": user_ids,
            "anchor_date": ["2026-01-14"] * len(user_ids),
            "y_rub": y_val,
            "z_true": y_val_log,
            "current_state": past_buyer_val,
            "p_react": p_val_exp,
            "p_churn": 1.0 - p_val_exp,
            "p_buy": p_val_exp,
            "conditional_z": z_cond_exp,
            "factorized_z": z_fact_exp,
            "final_prediction_z": z_fact_exp,
            "final_prediction_rub": pred_rub_exp,
        })

        if exp_name == "Baseline_C4":
            base_pred_df = pred_df_exp
            res = validate_report_invariants(pred_df_exp, alpha=1.10)
        else:
            res = validate_report_invariants(pred_df_exp, base_df=base_pred_df, alpha=1.10)

        exp_dur = time.time() - t_exp
        p_comp = res.get("paired_comparison")
        delta_r = p_comp["delta_rmsle"] if p_comp else 0.0
        p_better = p_comp["bootstrap_p_candidate_better"] if p_comp else 0.0

        print(f"[+] {exp_name} Done in {exp_dur:.1f}s | RMSLE: {res['rmsle']:.5f} | delta: {delta_r:+.5f} | React AUC: {res['react_auc']:.4f} | Churn AUC: {res['churn_auc']:.4f}")

        exp_save_dir = OUTPUT_DIR / exp_name
        exp_save_dir.mkdir(parents=True, exist_ok=True)
        pred_df_exp.write_parquet(exp_save_dir / "predictions_validation.parquet")
        with open(exp_save_dir / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)

        inc_results.append({
            "experiment": exp_name,
            "features_count": X_tr_exp.shape[1],
            "RMSLE": res["rmsle"],
            "MSE": res["mse_log"],
            "delta_RMSLE": delta_r,
            "React_AUC": res["react_auc"],
            "Churn_AUC": res["churn_auc"],
            "Overall_Brier": res["overall_brier"],
            "Bootstrap_P_Better": p_better,
            "duration_s": exp_dur,
        })

    inc_df = pl.DataFrame(inc_results)
    inc_df.write_csv(OUTPUT_DIR / "btyd_incremental_catboost_summary.csv")
    print("\n=== INCREMENTAL EXPERIMENTS SUMMARY ===")
    print(inc_df)
    print(f"\n[+] Total BTYD Research execution time: {time.time() - t0_start:.1f}s")


if __name__ == "__main__":
    main()
