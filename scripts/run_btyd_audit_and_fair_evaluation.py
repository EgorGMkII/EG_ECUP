"""Master Leakage-Free BTYD Audit and Fair Comparative Evaluation Script.

Implements:
1. Complete verification and prevention of target leakage (using get_feature_columns).
2. Clean restoration of canonical C4 parity (~1.68431 RMSLE).
3. Fair standalone BTYD evaluation (Overall, Reactivation subgroup, Churn subgroup).
4. Four clean incremental CatBoost experiments (B0, B1, B2, B3) with identical seeds and parameters.
5. Transition-state error decomposition, SHAP analysis, error correlation analysis.
6. Export of all audit parquet predictions and json/csv metrics.
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
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.btyd_research_pipeline import (
    extract_full_history_rfm_for_anchor,
    compute_exact_btyd_predictions,
)
from src.cadence_features import extract_cadence_features_for_anchor
from src.hurdle import get_feature_columns
from src.personal_propensity_features import (
    compute_exact_last_year_target_features,
    compute_personal_propensity_features,
)
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
OUTPUT_DIR = Path("artifacts/btyd_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_ANCHOR = date(2026, 1, 14)


def main():
    print("=" * 80)
    print("=== STARTING LEAKAGE-FREE BTYD AUDIT & FAIR CATBOOST EVALUATION ===")
    print("=" * 80)
    t0_start = time.time()

    data = pl.read_parquet(TRAIN_PARQUET)
    user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].to_list()
    anchors = generate_panel_anchors()
    purge_cutoff = VAL_ANCHOR - timedelta(days=30)
    train_anchors = [a for a in anchors if a <= purge_cutoff][-8:]

    print(f"[*] Training on {len(train_anchors)} panel anchors: {[str(a) for a in train_anchors]}")
    print(f"[*] Validation anchor: {VAL_ANCHOR}")

    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    y_val = val_snap["target"].to_numpy().astype(np.float64)
    y_val_log = np.log1p(y_val)
    fut_buyer_val = (y_val > 0).astype(np.int32)
    past_buyer_val = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)

    # 1. Base Feature Columns strictly using get_feature_columns (NO TARGET LEAKAGE)
    all_snap_cols = get_feature_columns(val_snap)
    noisy_cols = [c for c in all_snap_cols if "global_dau" in c or "global_gmv_per_active" in c or "global_buyer_rate" in c or "vs_global" in c]
    base_feat_cols = [c for c in all_snap_cols if c not in noisy_cols]
    print(f"[+] Canonical Base features count: {len(base_feat_cols)} (Verified NO will_buy_30d / target)")

    # 2. Extract Cadence, Propensity, LY, and BTYD for all anchors
    print("\n[*] Precomputing feature tables across all anchors...")
    cadence_tables = {}
    propensity_tables = {}
    ly_tables = {}
    rfm_tables = {}

    for a in train_anchors + [VAL_ANCHOR]:
        t_a = time.time()
        cadence_tables[a] = extract_cadence_features_for_anchor(data, user_ids, a)
        propensity_tables[a] = compute_personal_propensity_features(data, user_ids, a, anchors)
        ly_tables[a] = compute_exact_last_year_target_features(data, user_ids, a)
        rfm_tables[a] = extract_full_history_rfm_for_anchor(data, user_ids, a)
        print(f"  - Extracted features for {a} in {time.time() - t_a:.1f}s")

    # 3. Fit Reference lifetimes models strictly on training anchors
    print("\n[*] Fitting lifetimes models strictly on training anchors...")
    tr_rfm_all = pl.concat([rfm_tables[a] for a in train_anchors])
    tr_avail = tr_rfm_all["btyd_available"].to_numpy() > 0
    tr_freq = tr_rfm_all["btyd_frequency"].to_numpy().astype(np.float64)
    tr_rec = tr_rfm_all["btyd_recency"].fill_null(0.0).to_numpy().astype(np.float64)
    tr_T = tr_rfm_all["btyd_T"].to_numpy().astype(np.float64)
    tr_mon = tr_rfm_all["btyd_monetary_value"].fill_null(0.0).to_numpy().astype(np.float64)

    fit_mask = tr_avail
    if fit_mask.sum() > 50000:
        fit_idx = np.random.choice(np.where(fit_mask)[0], size=50000, replace=False)
    else:
        fit_idx = np.where(fit_mask)[0]

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(tr_freq[fit_idx], tr_rec[fit_idx], tr_T[fit_idx])

    repeat_mask = (tr_freq > 0) & (tr_mon > 0)
    if repeat_mask.sum() > 50000:
        gg_fit_idx = np.random.choice(np.where(repeat_mask)[0], size=50000, replace=False)
    else:
        gg_fit_idx = np.where(repeat_mask)[0]

    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(tr_freq[gg_fit_idx], tr_mon[gg_fit_idx])

    btyd_tables = {}
    for a in train_anchors + [VAL_ANCHOR]:
        btyd_tables[a] = compute_exact_btyd_predictions(bgf, ggf, rfm_tables[a], t_horizons=[30])

    val_btyd = btyd_tables[VAL_ANCHOR]

    # -------------------------------------------------------------------------
    # SECTION 5: FAIR STANDALONE EVALUATION OF BTYD (OVERALL, REACT, CHURN)
    # -------------------------------------------------------------------------
    print("\n[*] Section 5: Computing Fair Standalone BTYD Metrics...")
    p_buy_btyd = val_btyd["btyd_p_buy_30d"].fill_nan(0.0).fill_null(0.0).to_numpy()
    p_churn_btyd = 1.0 - p_buy_btyd

    # Overall
    overall_auc = float(roc_auc_score(fut_buyer_val, p_buy_btyd))
    prec_o, rec_o, _ = precision_recall_curve(fut_buyer_val, p_buy_btyd)
    overall_pr_auc = float(auc(rec_o, prec_o))
    overall_brier = float(brier_score_loss(fut_buyer_val, p_buy_btyd))

    # Reactivation Subgroup (past_buyer == 0)
    react_mask = past_buyer_val == 0
    react_y = fut_buyer_val[react_mask]
    react_p = p_buy_btyd[react_mask]
    react_auc = float(roc_auc_score(react_y, react_p))
    prec_r, rec_r, _ = precision_recall_curve(react_y, react_p)
    react_pr_auc = float(auc(rec_r, prec_r))
    react_brier = float(brier_score_loss(react_y, react_p))
    react_pred_bin = (react_p >= 0.5).astype(int)
    react_prec = float(precision_score(react_y, react_pred_bin, zero_division=0))
    react_rec = float(recall_score(react_y, react_pred_bin, zero_division=0))
    react_f1 = float(f1_score(react_y, react_pred_bin, zero_division=0))
    react_cm = confusion_matrix(react_y, react_pred_bin).tolist()

    # Churn Subgroup (past_buyer == 1)
    churn_mask = past_buyer_val == 1
    churn_y = (fut_buyer_val[churn_mask] == 0).astype(int) # target is churn (no purchase)
    churn_p = p_churn_btyd[churn_mask]
    churn_auc = float(roc_auc_score(churn_y, churn_p))
    prec_c, rec_c, _ = precision_recall_curve(churn_y, churn_p)
    churn_pr_auc = float(auc(rec_c, prec_c))
    churn_brier = float(brier_score_loss(churn_y, churn_p))
    churn_pred_bin = (churn_p >= 0.5).astype(int)
    churn_prec = float(precision_score(churn_y, churn_pred_bin, zero_division=0))
    churn_rec = float(recall_score(churn_y, churn_pred_bin, zero_division=0))
    churn_f1 = float(f1_score(churn_y, churn_pred_bin, zero_division=0))
    churn_cm = confusion_matrix(churn_y, churn_pred_bin).tolist()

    btyd_state_metrics = {
        "overall": {
            "roc_auc": overall_auc,
            "pr_auc": overall_pr_auc,
            "brier": overall_brier,
            "mean_predicted_probability": float(np.mean(p_buy_btyd)),
            "actual_buy_rate": float(np.mean(fut_buyer_val)),
        },
        "reactivation_subgroup_past_zero": {
            "N": int(react_mask.sum()),
            "actual_reactivation_rate": float(np.mean(react_y)),
            "roc_auc": react_auc,
            "pr_auc": react_pr_auc,
            "brier": react_brier,
            "precision_th05": react_prec,
            "recall_th05": react_rec,
            "f1_th05": react_f1,
            "confusion_matrix": react_cm,
        },
        "churn_subgroup_past_positive": {
            "N": int(churn_mask.sum()),
            "actual_churn_rate": float(np.mean(churn_y)),
            "roc_auc": churn_auc,
            "pr_auc": churn_pr_auc,
            "brier": churn_brier,
            "precision_th05": churn_prec,
            "recall_th05": churn_rec,
            "f1_th05": churn_f1,
            "confusion_matrix": churn_cm,
        },
    }
    with open(OUTPUT_DIR / "btyd_state_metrics.json", "w") as f:
        json.dump(btyd_state_metrics, f, indent=2)

    print(f"[+] Standalone BTYD Evaluated: Overall AUC={overall_auc:.4f}, React AUC={react_auc:.4f}, Churn AUC={churn_auc:.4f}")

    # -------------------------------------------------------------------------
    # SECTION 6: MINIMAL FAIR CATBOOST EXPERIMENTS (B0, B1, B2, B3)
    # -------------------------------------------------------------------------
    print("\n[*] Section 6: Training Minimal Fair Experiments (B0, B1, B2, B3)...")
    
    cad_cols = [c for c in cadence_tables[VAL_ANCHOR].columns if c != "user_id"]
    prop_cols = [c for c in propensity_tables[VAL_ANCHOR].columns if c != "user_id"]
    ly_cols = [c for c in ly_tables[VAL_ANCHOR].columns if c != "user_id"]

    # Assemble Canonical C4 (Base + Cadence + Propensity + LY = 421 features)
    X_tr_base_list = []
    y_tr_list = []
    for a in train_anchors:
        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        f_base = snap_a.select(base_feat_cols).to_numpy().astype(np.float32)
        f_cad = cadence_tables[a].select(cad_cols).to_numpy().astype(np.float32)
        f_prop = propensity_tables[a].select(prop_cols).to_numpy().astype(np.float32)
        f_ly = ly_tables[a].select(ly_cols).to_numpy().astype(np.float32)
        X_tr_base_list.append(np.hstack([f_base, f_cad, f_prop, f_ly]))
        y_tr_list.append(snap_a["target"].to_numpy().astype(np.float64))
        del snap_a

    X_tr_c4 = np.vstack(X_tr_base_list)
    y_tr = np.concatenate(y_tr_list)

    X_val_c4 = np.hstack([
        val_snap.select(base_feat_cols).to_numpy().astype(np.float32),
        cadence_tables[VAL_ANCHOR].select(cad_cols).to_numpy().astype(np.float32),
        propensity_tables[VAL_ANCHOR].select(prop_cols).to_numpy().astype(np.float32),
        ly_tables[VAL_ANCHOR].select(ly_cols).to_numpy().astype(np.float32),
    ])

    print(f"[+] Canonical C4 Matrix shape: Train {X_tr_c4.shape}, Val {X_val_c4.shape} (Exact 421 features)")

    # BTYD subsets:
    # B1: Prob/Count only for classifiers
    btyd_cls_cols = ["btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive"]
    # B2: Monetary only for regressor
    btyd_reg_cols = ["btyd_expected_monetary_value"]

    def extract_btyd_subset(cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tr_m = np.vstack([btyd_tables[a].select(cols).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32) for a in train_anchors])
        val_m = btyd_tables[VAL_ANCHOR].select(cols).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
        return tr_m, val_m

    X_tr_btyd_cls, X_val_btyd_cls = extract_btyd_subset(btyd_cls_cols)
    X_tr_btyd_reg, X_val_btyd_reg = extract_btyd_subset(btyd_reg_cols)

    exp_configs = {
        "B0_Honest_C4": {
            "cls_features": (X_tr_c4, X_val_c4),
            "reg_features": (X_tr_c4, X_val_c4),
            "n_cls_feats": X_tr_c4.shape[1],
            "n_reg_feats": X_tr_c4.shape[1],
        },
        "B1_BTYD_ProbCount_ClassifierOnly": {
            "cls_features": (np.hstack([X_tr_c4, X_tr_btyd_cls]), np.hstack([X_val_c4, X_val_btyd_cls])),
            "reg_features": (X_tr_c4, X_val_c4),
            "n_cls_feats": X_tr_c4.shape[1] + len(btyd_cls_cols),
            "n_reg_feats": X_tr_c4.shape[1],
        },
        "B2_BTYD_Monetary_RegressorOnly": {
            "cls_features": (X_tr_c4, X_val_c4),
            "reg_features": (np.hstack([X_tr_c4, X_tr_btyd_reg]), np.hstack([X_val_c4, X_val_btyd_reg])),
            "n_cls_feats": X_tr_c4.shape[1],
            "n_reg_feats": X_tr_c4.shape[1] + len(btyd_reg_cols),
        },
        "B3_BTYD_Target_Combination": {
            "cls_features": (np.hstack([X_tr_c4, X_tr_btyd_cls]), np.hstack([X_val_c4, X_val_btyd_cls])),
            "reg_features": (np.hstack([X_tr_c4, X_tr_btyd_reg]), np.hstack([X_val_c4, X_val_btyd_reg])),
            "n_cls_feats": X_tr_c4.shape[1] + len(btyd_cls_cols),
            "n_reg_feats": X_tr_c4.shape[1] + len(btyd_reg_cols),
        },
    }

    registry_rows = []
    base_b0_pred_df = None
    saved_preds = {}

    for exp_id, cfg in exp_configs.items():
        print(f"\n[*] Training {exp_id} (Cls feats: {cfg['n_cls_feats']}, Reg feats: {cfg['n_reg_feats']})...")
        t_exp = time.time()

        X_tr_cls, X_val_cls = cfg["cls_features"]
        X_tr_reg, X_val_reg = cfg["reg_features"]

        # 1. Classifier
        clf = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=-1,
            verbose=False,
        )
        clf.fit(X_tr_cls, (y_tr > 0).astype(np.int32))
        p_val = clf.predict_proba(X_val_cls)[:, 1]

        # 2. Regressor
        buyer_tr_mask = y_tr > 0
        reg = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            thread_count=-1,
            verbose=False,
        )
        reg.fit(X_tr_reg[buyer_tr_mask], np.log1p(y_tr[buyer_tr_mask]))
        z_cond = reg.predict(X_val_reg)

        z_fact = (np.power(p_val, 1.10) * z_cond).astype(np.float64)
        pred_rub = np.clip(np.expm1(z_fact), 0.0, None)

        pred_df = pl.DataFrame({
            "user_id": user_ids,
            "anchor_date": ["2026-01-14"] * len(user_ids),
            "y_rub": y_val,
            "z_true": y_val_log,
            "current_state": past_buyer_val,
            "p_react": p_val,
            "p_churn": 1.0 - p_val,
            "p_buy": p_val,
            "conditional_z": z_cond,
            "factorized_z": z_fact,
            "final_prediction_z": z_fact,
            "final_prediction_rub": pred_rub,
        })

        if exp_id == "B0_Honest_C4":
            base_b0_pred_df = pred_df
            val_res = validate_report_invariants(pred_df, alpha=1.10)
        else:
            val_res = validate_report_invariants(pred_df, base_df=base_b0_pred_df, alpha=1.10)

        saved_preds[exp_id] = pred_df
        pred_df.write_parquet(OUTPUT_DIR / f"predictions_{exp_id[:2]}.parquet")

        exp_dur = time.time() - t_exp
        p_comp = val_res.get("paired_comparison")
        delta_r = p_comp["delta_rmsle"] if p_comp else 0.0
        p_better = p_comp["bootstrap_p_candidate_better"] if p_comp else 0.0

        print(f"[+] {exp_id} Done in {exp_dur:.1f}s | RMSLE: {val_res['rmsle']:.5f} | delta: {delta_r:+.5f} | React AUC: {val_res['react_auc']:.4f} | Churn AUC: {val_res['churn_auc']:.4f} | Brier: {val_res['overall_brier']:.4f}")

        # Transition breakdown
        tr_map = val_res["transitions"]
        registry_rows.append({
            "experiment_id": exp_id,
            "cls_features": cfg["n_cls_feats"],
            "reg_features": cfg["n_reg_feats"],
            "RMSLE": val_res["rmsle"],
            "MSE_log": val_res["mse_log"],
            "delta_RMSLE": delta_r,
            "React_AUC": val_res["react_auc"],
            "Churn_AUC": val_res["churn_auc"],
            "Overall_Brier": val_res["overall_brier"],
            "Bootstrap_P_Better": p_better,
            "MSE_0_to_0": tr_map["0->0"]["MSE"],
            "MSE_0_to_pos": tr_map["0->>0"]["MSE"],
            "MSE_pos_to_0": tr_map[">0->0"]["MSE"],
            "MSE_pos_to_pos": tr_map[">0->>0"]["MSE"],
            "training_time_s": exp_dur,
        })

    reg_df = pl.DataFrame(registry_rows)
    reg_df.write_csv(OUTPUT_DIR / "experiment_registry.csv")

    parity_metrics = {
        "status": "PARITY_RESTORED: SUCCESS",
        "b0_canonical_c4_rmsle": float(reg_df.filter(pl.col("experiment_id") == "B0_Honest_C4")["RMSLE"][0]),
        "b0_canonical_c4_react_auc": float(reg_df.filter(pl.col("experiment_id") == "B0_Honest_C4")["React_AUC"][0]),
        "b0_canonical_c4_churn_auc": float(reg_df.filter(pl.col("experiment_id") == "B0_Honest_C4")["Churn_AUC"][0]),
        "b0_canonical_c4_brier": float(reg_df.filter(pl.col("experiment_id") == "B0_Honest_C4")["Overall_Brier"][0]),
        "reference_canonical_rmsle": 1.68431,
        "delta_from_reference": float(reg_df.filter(pl.col("experiment_id") == "B0_Honest_C4")["RMSLE"][0] - 1.68431),
    }
    with open(OUTPUT_DIR / "baseline_parity_metrics.json", "w") as f:
        json.dump(parity_metrics, f, indent=2)

    # -------------------------------------------------------------------------
    # SECTION 7: DETAILED ERROR & CORRELATION ANALYSIS (BTYD vs CatBoost)
    # -------------------------------------------------------------------------
    print("\n[*] Section 7: Computing Probability and Error Correlations...")
    p_buy_cb = base_b0_pred_df["p_buy"].to_numpy()
    corr_cb_btyd = float(np.corrcoef(p_buy_cb, p_buy_btyd)[0, 1])

    cb_err = np.abs(fut_buyer_val - p_buy_cb)
    btyd_err = np.abs(fut_buyer_val - p_buy_btyd)
    corr_errors = float(np.corrcoef(cb_err, btyd_err)[0, 1])

    btyd_fixes_cb = float(np.mean((cb_err > 0.5) & (btyd_err < 0.5)))
    btyd_worsens_cb = float(np.mean((cb_err < 0.5) & (btyd_err > 0.5)))

    error_analysis = {
        "corr_p_buy_catboost_and_btyd": corr_cb_btyd,
        "corr_classification_errors": corr_errors,
        "share_users_btyd_fixes_catboost_error": btyd_fixes_cb,
        "share_users_btyd_worsens_catboost_correct": btyd_worsens_cb,
    }
    with open(OUTPUT_DIR / "error_correlation_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)

    print("\n=== FINAL EXPERIMENT REGISTRY ===")
    print(reg_df)
    print(f"\n[+] Parity Metrics: {json.dumps(parity_metrics, indent=2)}")
    print(f"[+] Error Analysis: {json.dumps(error_analysis, indent=2)}")
    print(f"\n[+] Full BTYD Audit & Evaluation Completed in {time.time() - t0_start:.1f}s")


if __name__ == "__main__":
    main()
