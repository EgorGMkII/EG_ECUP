"""Full Orchestration Runner for CatBoost Cadence, Propensity, and BTYD Experiments (C0-C5, B0-B3)."""

import gc
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import brier_score_loss, roc_auc_score, root_mean_squared_error

sys.path.insert(0, os.getcwd())

from src.btyd_pipeline import generate_btyd_dataset_for_anchor
from src.cadence_features import extract_cadence_features_for_anchor
from src.hurdle import get_feature_columns
from src.personal_propensity_features import (
    compute_exact_last_year_target_features,
    compute_personal_propensity_features,
)
from src.snapshots import SNAPSHOTS_DIR, TRAIN_PARQUET, generate_panel_anchors
from src.validation import get_snapshot_path
from scripts.validate_experiment_report import validate_report_invariants

AUDIT_DIR = Path("artifacts/catboost_cadence_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ANCHOR = date(2026, 1, 14)


def train_and_eval_catboost_hurdle(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    user_ids_val: np.ndarray,
    past_buyer_val: np.ndarray,
    alpha: float = 1.10,
    iterations: int = 600,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
    cat_features: Optional[List[int]] = None,
) -> Tuple[pl.DataFrame, Dict]:
    """Trains Hurdle CatBoost Classifier & Regressor and evaluates on validation set."""
    y_tr_bin = (y_tr > 0).astype(np.int32)
    buyer_mask_tr = y_tr > 0
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    # 1. Classifier
    clf = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        thread_count=-1,
        verbose=False,
        cat_features=cat_features,
    )
    clf.fit(X_tr, y_tr_bin)
    p_val = clf.predict_proba(X_val)[:, 1]

    # 2. Conditional Regressor (trained only on positive buyers)
    reg = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=random_seed,
        thread_count=-1,
        verbose=False,
        cat_features=cat_features,
    )
    if cat_features:
        X_tr_pos = X_tr[buyer_mask_tr]
    else:
        X_tr_pos = X_tr[buyer_mask_tr]
    reg.fit(X_tr_pos, y_tr_log[buyer_mask_tr])
    z_cond = reg.predict(X_val)

    # Hurdle factorization
    z_fact = (np.power(p_val, alpha) * z_cond).astype(np.float64)
    pred_rub = np.clip(np.expm1(z_fact), 0.0, None)

    pred_df = pl.DataFrame({
        "user_id": user_ids_val,
        "anchor_date": ["2026-01-14"] * len(user_ids_val),
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

    return pred_df, {"clf": clf, "reg": reg}


def main():
    print("===================================================================")
    print("=== STARTING CATBOOST CADENCE & BTYD EXPERIMENTAL SUITE ===")
    print("===================================================================")
    t0_all = time.time()

    data = pl.read_parquet(TRAIN_PARQUET)
    anchors = generate_panel_anchors()
    purge_cutoff = VAL_ANCHOR - timedelta(days=30)
    train_anchors = [a for a in anchors if a <= purge_cutoff][-8:]
    all_anchors = anchors

    print(f"[*] Training on {len(train_anchors)} panel anchors: {[str(a) for a in train_anchors]}")
    print(f"[*] Validation anchor: {VAL_ANCHOR}")

    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    user_ids = val_snap["user_id"].to_list()
    user_ids_arr = np.array(user_ids, dtype=np.int64)
    y_val = val_snap["target"].to_numpy().astype(np.float64)
    past_buyer_val = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)

    # Base feature columns
    all_cols = get_feature_columns(val_snap)
    noisy = [c for c in all_cols if "global_dau" in c or "global_gmv_per_active" in c or "global_buyer_rate" in c or "vs_global" in c]
    base_feat_cols = [c for c in all_cols if c not in noisy]

    # 1. Feature Extraction across training & validation
    print("\n[*] Precomputing Cadence, Propensity, and Last-Year feature tables...")
    cadence_tables = {}
    propensity_tables = {}
    ly_tables = {}
    btyd_tables = {}

    for a in train_anchors + [VAL_ANCHOR]:
        t_a = time.time()
        c_df = extract_cadence_features_for_anchor(data, user_ids, a)
        p_df = compute_personal_propensity_features(data, user_ids, a, all_anchors)
        ly_df = compute_exact_last_year_target_features(data, user_ids, a)
        b_df, _, _ = generate_btyd_dataset_for_anchor(data, user_ids, a)

        cadence_tables[a] = c_df
        propensity_tables[a] = p_df
        ly_tables[a] = ly_df
        btyd_tables[a] = b_df
        print(f"  - Extracted features for {a} in {time.time()-t_a:.1f}s")

    cadence_cols = [c for c in cadence_tables[VAL_ANCHOR].columns if c != "user_id"]
    propensity_cols = [c for c in propensity_tables[VAL_ANCHOR].columns if c != "user_id"]
    ly_cols = [c for c in ly_tables[VAL_ANCHOR].columns if c != "user_id"]
    btyd_cols = [c for c in btyd_tables[VAL_ANCHOR].columns if c != "user_id"]

    print(f"\n[+] Feature Dimensions:")
    print(f"  - Base (C0): {len(base_feat_cols)} cols")
    print(f"  - Cadence:   {len(cadence_cols)} cols")
    print(f"  - Propensity:{len(propensity_cols)} cols")
    print(f"  - Exact LY:  {len(ly_cols)} cols")
    print(f"  - BTYD:      {len(btyd_cols)} cols")

    # -------------------------------------------------------------------------
    # SECTION 9.5 & 10: BTYD SANITY CHECKS & B0 STANDALONE EVALUATION
    # -------------------------------------------------------------------------
    print("\n[*] Performing BTYD Sanity Checks and Standalone Diagnostics (B0)...")
    val_btyd_df = btyd_tables[VAL_ANCHOR]
    fut_buyer_val = (y_val > 0).astype(np.int32)
    y_val_log = np.log1p(y_val)

    btyd_sanity = {}
    for col in btyd_cols:
        vals = val_btyd_df[col].to_numpy().astype(np.float64)
        btyd_sanity[col] = {
            "min": float(np.min(vals)),
            "p01": float(np.percentile(vals, 1)),
            "p10": float(np.percentile(vals, 10)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "missing_rate": float(val_btyd_df[col].is_null().mean()),
            "finite_rate": float(np.isfinite(vals).mean()),
        }

    p_alive_val = val_btyd_df["btyd_p_alive"].to_numpy().astype(np.float64)
    p_buy_val = val_btyd_df["btyd_p_at_least_one_purchase_30d"].to_numpy().astype(np.float64)
    exp_gmv_val = val_btyd_df["btyd_expected_gmv_30d"].to_numpy().astype(np.float64)
    exp_cnt_val = val_btyd_df["btyd_expected_purchases_30d"].to_numpy().astype(np.float64)

    btyd_sanity["summary_correlations"] = {
        "share_p_alive_gt_099": float(np.mean(p_alive_val > 0.99)),
        "share_p_buy_gt_099": float(np.mean(p_buy_val > 0.99)),
        "corr_p_alive_and_p_buy": float(np.corrcoef(p_alive_val, p_buy_val)[0, 1]),
        "corr_exp_count_and_actual_buy": float(np.corrcoef(exp_cnt_val, fut_buyer_val)[0, 1]),
        "corr_exp_gmv_and_actual_log_gmv": float(np.corrcoef(np.log1p(exp_gmv_val), y_val_log)[0, 1]),
    }

    with open(AUDIT_DIR / "btyd_sanity_checks.json", "w") as f:
        json.dump(btyd_sanity, f, indent=2)

    # B0 Standalone Evaluation
    b0_auc = float(roc_auc_score(fut_buyer_val, p_buy_val))
    b0_brier = float(brier_score_loss(fut_buyer_val, p_buy_val))
    b0_raw_rmsle = float(root_mean_squared_error(y_val_log, np.log1p(np.clip(exp_gmv_val, 0, None))))
    
    # Assemble Base matrices
    X_tr_base_list, y_tr_list = [], []
    for a in train_anchors:
        snap_a = pl.read_parquet(get_snapshot_path(a, SNAPSHOTS_DIR))
        X_tr_base_list.append(snap_a.select(base_feat_cols).to_numpy().astype(np.float32))
        y_tr_list.append(snap_a["target"].to_numpy().astype(np.float64))
        del snap_a

    X_tr_base = np.vstack(X_tr_base_list)
    y_tr = np.concatenate(y_tr_list)
    X_val_base = val_snap.select(base_feat_cols).to_numpy().astype(np.float32)

    # Simple linear log-calibration on train
    tr_btyd_df = pl.concat([btyd_tables[a] for a in train_anchors])
    tr_y_log = np.log1p(y_tr)
    tr_exp_gmv_log = np.log1p(tr_btyd_df["btyd_expected_gmv_30d"].to_numpy().astype(np.float64))
    
    # Fit scalar scale & bias
    A = np.column_stack([tr_exp_gmv_log, np.ones_like(tr_exp_gmv_log)])
    cal_params, _, _, _ = np.linalg.lstsq(A, tr_y_log, rcond=None)
    b0_cal_z = np.clip(cal_params[0] * np.log1p(exp_gmv_val) + cal_params[1], 0, None)
    b0_cal_rmsle = float(root_mean_squared_error(y_val_log, b0_cal_z))

    b0_metrics = {
        "b0_auc_p_buy_30d": b0_auc,
        "b0_brier_p_buy_30d": b0_brier,
        "b0_raw_expected_gmv_rmsle": b0_raw_rmsle,
        "b0_calibrated_expected_gmv_rmsle": b0_cal_rmsle,
    }
    with open(AUDIT_DIR / "B0_standalone_btyd_metrics.json", "w") as f:
        json.dump(b0_metrics, f, indent=2)
    print(f"[+] BTYD Sanity & B0 Diagnostics Complete: AUC={b0_auc:.4f}, Brier={b0_brier:.4f}, Cal RMSLE={b0_cal_rmsle:.4f}")

    # Helper to stack extra feature blocks
    def get_stacked_block(tables: Dict[date, pl.DataFrame], cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tr_list = [tables[a].select(cols).to_numpy().astype(np.float32) for a in train_anchors]
        return np.vstack(tr_list), tables[VAL_ANCHOR].select(cols).to_numpy().astype(np.float32)

    X_tr_cad, X_val_cad = get_stacked_block(cadence_tables, cadence_cols)
    X_tr_prop, X_val_prop = get_stacked_block(propensity_tables, propensity_cols)
    X_tr_ly, X_val_ly = get_stacked_block(ly_tables, ly_cols)
    X_tr_btyd, X_val_btyd = get_stacked_block(btyd_tables, btyd_cols)

    # -------------------------------------------------------------------------
    # EXPERIMENTS SUITE DEFINITION
    # -------------------------------------------------------------------------
    experiments = {
        "C0_Baseline": (X_tr_base, X_val_base, None),
        "C1_Cadence": (np.hstack([X_tr_base, X_tr_cad]), np.hstack([X_val_base, X_val_cad]), None),
        "C2_Propensity": (np.hstack([X_tr_base, X_tr_prop]), np.hstack([X_val_base, X_val_prop]), None),
        "C3_Exact_LY": (np.hstack([X_tr_base, X_tr_ly]), np.hstack([X_val_base, X_val_ly]), None),
        "C4_Combined_Best": (np.hstack([X_tr_base, X_tr_cad, X_tr_prop, X_tr_ly]), np.hstack([X_val_base, X_val_cad, X_val_prop, X_val_ly]), None),
        "B1_CatBoost_BTYD": (np.hstack([X_tr_base, X_tr_cad, X_tr_prop, X_tr_ly, X_tr_btyd]), np.hstack([X_val_base, X_val_cad, X_val_prop, X_val_ly, X_val_btyd]), None),
    }

    # B2 Component Ablations
    # B2.1: without p_alive
    no_alive_idx = [i for i, c in enumerate(btyd_cols) if "p_alive" not in c]
    X_tr_b2_1 = np.hstack([X_tr_base, X_tr_cad, X_tr_prop, X_tr_ly, X_tr_btyd[:, no_alive_idx]])
    X_val_b2_1 = np.hstack([X_val_base, X_val_cad, X_val_prop, X_val_ly, X_val_btyd[:, no_alive_idx]])
    experiments["B2_No_P_Alive"] = (X_tr_b2_1, X_val_b2_1, None)

    # B2.2: without expected gmv/monetary
    no_mon_idx = [i for i, c in enumerate(btyd_cols) if "gmv" not in c and "monetary" not in c]
    X_tr_b2_2 = np.hstack([X_tr_base, X_tr_cad, X_tr_prop, X_tr_ly, X_tr_btyd[:, no_mon_idx]])
    X_val_b2_2 = np.hstack([X_val_base, X_val_cad, X_val_prop, X_val_ly, X_val_btyd[:, no_mon_idx]])
    experiments["B2_No_Monetary"] = (X_tr_b2_2, X_val_b2_2, None)

    results_registry = []
    baseline_pred_df = None

    print("\n" + "=" * 80)
    print("STARTING MODEL TRAINING ACROSS ALL CONFIGURATIONS")
    print("=" * 80)

    for exp_id, (X_tr_exp, X_val_exp, cat_f) in experiments.items():
        print(f"\n[*] Training {exp_id} ({X_tr_exp.shape[1]} features on {len(X_tr_exp):,} samples)...")
        t_exp = time.time()
        pred_df, models = train_and_eval_catboost_hurdle(
            X_tr=X_tr_exp,
            y_tr=y_tr,
            X_val=X_val_exp,
            y_val=y_val,
            user_ids_val=user_ids_arr,
            past_buyer_val=past_buyer_val,
            alpha=1.10,
            iterations=600,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            cat_features=cat_f,
        )

        exp_dir = AUDIT_DIR / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        pred_df.write_parquet(exp_dir / "predictions_validation.parquet")

        if exp_id == "C0_Baseline":
            baseline_pred_df = pred_df
            res = validate_report_invariants(pred_df, alpha=1.10)
        else:
            res = validate_report_invariants(pred_df, base_df=baseline_pred_df, alpha=1.10)

        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)

        p_comp = res.get("paired_comparison")
        delta_r = p_comp["delta_rmsle"] if p_comp else 0.0
        p_better = p_comp["bootstrap_p_candidate_better"] if p_comp else 0.0

        results_registry.append({
            "experiment_id": exp_id,
            "feature_count": X_tr_exp.shape[1],
            "RMSLE": res["rmsle"],
            "MSE": res["mse_log"],
            "delta_RMSLE": delta_r,
            "React_AUC": res["react_auc"],
            "Churn_AUC": res["churn_auc"],
            "Overall_Brier": res["overall_brier"],
            "Bootstrap_P_Better": p_better,
            "training_time_s": round(time.time() - t_exp, 1),
        })

        print(f"[+] {exp_id} Done in {time.time()-t_exp:.1f}s | RMSLE: {res['rmsle']:.5f} | delta: {delta_r:+.5f} | React AUC: {res['react_auc']:.4f} | Churn AUC: {res['churn_auc']:.4f}")

    summary_df = pl.DataFrame(results_registry)
    summary_path = AUDIT_DIR / "catboost_cadence_btyd_summary.csv"
    summary_df.write_csv(summary_path)
    print("\n" + "=" * 80)
    print("FINAL SUMMARY OF CATBOOST CADENCE & BTYD EXPERIMENTS:")
    print("=" * 80)
    print(summary_df)
    print(f"\n[+] Total Pipeline Completed in {time.time()-t0_all:.1f}s")


if __name__ == "__main__":
    main()
