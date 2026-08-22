"""Feature Importance, SHAP Analysis, and Group Drop-Column Ablation for Baseline CatBoost."""

import os
import sys
sys.path.insert(0, os.getcwd())
import json
import time
from pathlib import Path
from datetime import date
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error

from src.validation import get_snapshot_path
from src.snapshots import SNAPSHOTS_DIR

AUDIT_DIR = Path("artifacts/catboost_cadence_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ANCHOR = date(2026, 1, 14)


def main():
    print("[*] Running Step 6: Feature Importance, SHAP & Group Ablation...")
    val_snap = pl.read_parquet(get_snapshot_path(VAL_ANCHOR, SNAPSHOTS_DIR))
    catalog = pl.read_csv(AUDIT_DIR / "feature_catalog.csv")
    feat_cols = catalog["feature_name"].to_list()
    feat_to_group = dict(zip(catalog["feature_name"], catalog["feature_group"]))

    clf = CatBoostClassifier().load_model("models/clean_classifier_final.cbm")
    reg = CatBoostRegressor().load_model("models/clean_regressor_final.cbm")

    X_val = val_snap.select(feat_cols).to_numpy().astype(np.float32)
    y_val = val_snap["target"].to_numpy().astype(np.float64)
    y_log = np.log1p(y_val)
    fut_buyer = (y_val > 0).astype(np.int32)
    past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)

    # 1. PredictionValuesChange Feature Importance
    print("[*] Computing Feature Importances (PredictionValuesChange)...")
    clf_imp = clf.get_feature_importance(type="PredictionValuesChange")
    reg_imp = reg.get_feature_importance(type="PredictionValuesChange")

    imp_df = pl.DataFrame({
        "feature_name": feat_cols,
        "feature_group": [feat_to_group.get(f, "prochie") for f in feat_cols],
        "classifier_importance": clf_imp,
        "regressor_importance": reg_imp,
        "mean_importance": (clf_imp + reg_imp) / 2.0,
    }).sort("mean_importance", descending=True)

    imp_df.write_csv(AUDIT_DIR / "baseline_feature_importance.csv")
    print(f"[+] Saved baseline feature importance to {AUDIT_DIR / 'baseline_feature_importance.csv'}")

    # Group-level importance
    grp_imp = imp_df.group_by("feature_group").agg([
        pl.col("classifier_importance").sum().alias("sum_clf_importance"),
        pl.col("regressor_importance").sum().alias("sum_reg_importance"),
        pl.col("mean_importance").sum().alias("sum_mean_importance"),
        pl.len().alias("feature_count"),
    ]).sort("sum_mean_importance", descending=True)

    grp_imp.write_csv(AUDIT_DIR / "baseline_group_importance.csv")
    print(f"[+] Saved baseline group importance to {AUDIT_DIR / 'baseline_group_importance.csv'}")

    # 2. Stratified SHAP Analysis (Sample of 10,000 for efficiency)
    print("\n[*] Computing Stratified SHAP Summary (10,000 validation samples)...")
    np.random.seed(42)
    n_sample = 10000
    strat_indices = []
    for c_val, f_val in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (past_buyer == c_val) & (fut_buyer == f_val)
        idx_sub = np.where(mask)[0]
        sampled = np.random.choice(idx_sub, size=min(n_sample // 4, len(idx_sub)), replace=False)
        strat_indices.extend(sampled)

    strat_indices = np.array(strat_indices)
    X_shap = X_val[strat_indices]
    y_shap_bin = fut_buyer[strat_indices]

    pool_shap = Pool(X_shap, y_shap_bin)
    shap_vals_clf = clf.get_feature_importance(pool_shap, type="ShapValues")
    # ShapValues returned as [N, n_features + 1]
    mean_abs_shap_clf = np.mean(np.abs(shap_vals_clf[:, :-1]), axis=0)

    shap_df = pl.DataFrame({
        "feature_name": feat_cols,
        "feature_group": [feat_to_group.get(f, "prochie") for f in feat_cols],
        "mean_abs_shap_classifier": mean_abs_shap_clf,
    }).sort("mean_abs_shap_classifier", descending=True)

    shap_df.write_csv(AUDIT_DIR / "baseline_shap_summary.csv")
    print(f"[+] Saved baseline SHAP summary to {AUDIT_DIR / 'baseline_shap_summary.csv'}")

    # 3. Group Drop-Column Ablation (Zero out feature groups on validation)
    print("\n[*] Running Group Drop-Column Feature Ablation on Validation...")
    base_p = clf.predict_proba(X_val)[:, 1]
    base_z = reg.predict(X_val)
    base_pred_rub = np.clip(np.expm1(np.power(base_p, 1.10) * base_z), 0, None)
    base_rmsle = float(root_mean_squared_error(y_log, np.log1p(base_pred_rub)))
    base_mse = float(np.mean((y_log - np.log1p(base_pred_rub)) ** 2))

    ablation_results = [
        {
            "group_ablated": "NONE (Full Baseline)",
            "features_removed": 0,
            "RMSLE": base_rmsle,
            "MSE": base_mse,
            "delta_RMSLE": 0.0,
            "delta_MSE": 0.0,
            "pct_MSE_increase": 0.0,
        }
    ]

    target_groups = ["recency", "lifetime/user tenure", "frequency", "trend/decay", "calendar", "search/cart/order funnel", "monetary"]
    for grp in target_groups:
        grp_mask = [1.0 if feat_to_group.get(f) != grp else 0.0 for f in feat_cols]
        X_abl = X_val * np.array(grp_mask, dtype=np.float32)

        p_abl = clf.predict_proba(X_abl)[:, 1]
        z_abl = reg.predict(X_abl)
        pred_abl_rub = np.clip(np.expm1(np.power(p_abl, 1.10) * z_abl), 0, None)

        r_abl = float(root_mean_squared_error(y_log, np.log1p(pred_abl_rub)))
        m_abl = float(np.mean((y_log - np.log1p(pred_abl_rub)) ** 2))
        n_rem = int(sum(1 for g in grp_mask if g == 0.0))

        ablation_results.append({
            "group_ablated": f"Without {grp}",
            "features_removed": n_rem,
            "RMSLE": r_abl,
            "MSE": m_abl,
            "delta_RMSLE": r_abl - base_rmsle,
            "delta_MSE": m_abl - base_mse,
            "pct_MSE_increase": ((m_abl - base_mse) / base_mse) * 100.0,
        })

    abl_df = pl.DataFrame(ablation_results).sort("delta_RMSLE", descending=True)
    abl_df.write_csv(AUDIT_DIR / "baseline_group_ablation.csv")
    print("\nGroup Drop-Column Ablation Results:")
    print(abl_df)


if __name__ == "__main__":
    main()
