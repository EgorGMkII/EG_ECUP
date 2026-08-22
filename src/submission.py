"""End-to-End Submission Pipeline: Full 250k test snapshot generation, Hurdle + Direct Log-Space Ensemble, and CSV export."""

import gc
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor

from src.features import compute_global_platform_table
from src.hurdle import get_feature_columns
from src.snapshots import build_snapshot, generate_panel_anchors, SNAPSHOTS_DIR, TRAIN_PARQUET
from src.validation import get_snapshot_path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_full_250k_test_snapshot(
    data_path: Union[str, Path] = TRAIN_PARQUET,
    output_path: Union[str, Path] = SNAPSHOTS_DIR / "snapshot_2026-02-13_test_250k.parquet",
) -> pl.DataFrame:
    """Builds the feature snapshot for all 250,000 users on test anchor 2026-02-13."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[+] Loaded existing 250k test snapshot from {output_path}")
        return pl.read_parquet(output_path)

    print("[*] Generating full 250,000 user test snapshot for 2026-02-13...")
    data = pl.read_parquet(data_path)
    all_users = (
        pl.scan_parquet(data_path)
        .select(pl.col("user_id").unique())
        .collect()["user_id"]
        .sort()
        .to_list()
    )

    global_table = compute_global_platform_table(data)
    test_snap = build_snapshot(
        data=data,
        user_ids=all_users,
        anchor_date=date(2026, 2, 13),
        global_table=global_table,
        is_test=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_snap.write_parquet(output_path)
    print(f"[+] Saved 250k test snapshot to {output_path} ({test_snap.height:,} rows, {test_snap.width} cols)")
    return test_snap


def generate_submission(
    snapshots_dir: Path = SNAPSHOTS_DIR,
    output_csv: Path = Path("data/submission.csv"),
    alpha: float = 1.1,
    ensemble_hurdle_weight: float = 0.5,
    iterations: int = 600,
    learning_rate: float = 0.05,
) -> pl.DataFrame:
    """Trains final Hurdle + Direct Log-Space Ensemble on all 22 panel training snapshots and produces submission.csv."""
    anchors = generate_panel_anchors()
    print(f"[*] Loading all {len(anchors)} panel training snapshots for full fit...")

    # Load 1 snapshot to get feature columns
    sample_df = pl.read_parquet(get_snapshot_path(anchors[0], snapshots_dir))
    all_feature_cols = get_feature_columns(sample_df)

    # Filter out noisy macro-metrics as proven by diagnostic ablation
    noisy_global_cols = [c for c in all_feature_cols if "global_dau" in c or "global_gmv_per_active" in c or "global_buyer_rate" in c or "vs_global" in c]
    feature_cols = [c for c in all_feature_cols if c not in noisy_global_cols]
    print(f"[*] Filtered {len(noisy_global_cols)} noisy macro-features. Optimal feature count: {len(feature_cols)}")

    # Load all training snapshots with memory safety (float32)
    train_X_list, train_y_list = [], []
    cols_to_load = feature_cols + ["target"]

    for a in anchors:
        df_a = pl.read_parquet(get_snapshot_path(a, snapshots_dir), columns=cols_to_load)
        train_X_list.append(df_a.select(feature_cols).to_numpy().astype(np.float32))
        train_y_list.append(df_a["target"].to_numpy().astype(np.float32))
        del df_a

    X_tr = np.vstack(train_X_list)
    y_tr_target = np.concatenate(train_y_list)
    del train_X_list, train_y_list
    gc.collect()

    print(f"[+] Combined full training panel: {len(X_tr):,} rows across {len(anchors)} anchor dates")

    # Load 250k Test snapshot
    test_df = build_full_250k_test_snapshot()
    X_ts = test_df.select(feature_cols).to_numpy().astype(np.float32)
    user_ids = test_df["user_id"].to_list()
    del test_df
    gc.collect()

    y_tr_bin = (y_tr_target > 0).astype(np.int32)
    tr_buyer_mask = y_tr_target > 0

    # 1. Train Stage 1 Classifier
    print(f"\n[1/3] Training Stage 1 Classifier on {len(X_tr):,} rows ({int(np.sum(y_tr_bin)):,} buyers)...")
    clf = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        thread_count=4,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
    )
    clf.fit(X_tr, y_tr_bin)
    clf.save_model(MODELS_DIR / "full_classifier_final.cbm")
    p_test = clf.predict_proba(X_ts)[:, 1]
    p_test_adj = np.power(p_test, alpha)
    del clf, y_tr_bin
    gc.collect()

    # 2. Train Stage 2 Conditional Regressor (Active Buyers Only)
    X_tr_buyers = X_tr[tr_buyer_mask]
    y_tr_buyers_log = np.log1p(y_tr_target[tr_buyer_mask])

    print(f"\n[2/3] Training Stage 2 Conditional Regressor on {len(X_tr_buyers):,} active buyer rows...")
    reg = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        thread_count=4,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )
    reg.fit(X_tr_buyers, y_tr_buyers_log)
    reg.save_model(MODELS_DIR / "full_regressor_final.cbm")
    pred_test_buyers_log = reg.predict(X_ts)
    del reg, X_tr_buyers, y_tr_buyers_log, tr_buyer_mask
    gc.collect()

    # 3. Train Direct Regressor (All Rows)
    print(f"\n[3/3] Training Direct Regressor on {len(X_tr):,} rows...")
    direct_reg = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        thread_count=4,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )
    direct_reg.fit(X_tr, np.log1p(y_tr_target))
    direct_reg.save_model(MODELS_DIR / "full_direct_regressor_final.cbm")
    pred_test_direct_log = direct_reg.predict(X_ts)
    del direct_reg, X_tr, y_tr_target, X_ts
    gc.collect()

    # 4. Final Log-Space Blended Ensemble
    print("\n[*] Blending predictions in log-space...")
    z_hurdle = p_test_adj * pred_test_buyers_log
    z_direct = pred_test_direct_log
    w = ensemble_hurdle_weight
    z_final = (1.0 - w) * z_direct + w * z_hurdle

    final_pred = np.clip(np.expm1(z_final), 0.0, None)

    # 5. Format and Export Submission
    sub = pl.DataFrame({
        "user_id": user_ids,
        "predict": final_pred,
    })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.write_csv(output_csv)

    print(f"\n[+] SUCCESS! Exported final submission to {output_csv}")
    print(f"    Total users: {sub.height:,}")
    print(f"    Mean Predicted GMV:   {float(sub['predict'].mean()):.2f} rub")
    print(f"    Median Predicted GMV: {float(sub['predict'].median()):.2f} rub")
    print(f"    P90 Predicted GMV:    {float(np.percentile(sub['predict'], 90)):.2f} rub")
    print(f"    P99 Predicted GMV:    {float(np.percentile(sub['predict'], 99)):.2f} rub")
    print(f"    Max Predicted GMV:    {float(sub['predict'].max()):.2f} rub")
    print(f"    Null / NaN count:     {sub['predict'].is_null().sum()}")
    return sub


if __name__ == "__main__":
    generate_submission()
