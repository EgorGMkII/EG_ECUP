"""Phase 10: Generate Diagnostic Alpha Submissions v5.2 (alpha=1.0) and v5.3 (alpha=0.9)."""

import json
from pathlib import Path
import numpy as np
import polars as pl

AUDIT_DIR = Path("artifacts/transformer_audit")
DATA_DIR = Path("data")


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
        "pct_below_1_rub": float(np.mean(arr < 1.0) * 100),
        "pct_above_100_rub": float(np.mean(arr > 100.0) * 100),
    }


def main():
    print("===================================================================")
    print("=== PHASE 10: DIAGNOSTIC ALPHA SUBMISSIONS (1.0 vs 0.9) ===")
    print("===================================================================")

    # 1. Load exact v5.1 test components
    test_parquet_path = AUDIT_DIR / "v51_exact_test.parquet"
    df_v51 = pl.read_parquet(test_parquet_path)
    user_ids = df_v51["user_id"].to_list()
    n_users = len(user_ids)

    z_cb_test = df_v51["z_catboost"].to_numpy().astype(np.float32)
    z_gru_test = df_v51["z_gru"].to_numpy().astype(np.float32)
    y_pred_v51 = df_v51["predict"].to_numpy().astype(np.float32)

    # We also load the raw p_buy and z_cond for exact alpha variation
    cb_cache_path = Path("artifacts/transitions/catboost_test_pred_v5_1.npy")
    gru_cache_path = Path("artifacts/transitions/gru_test_pred_v5_1.npy")

    # Alpha 1.0 variation
    # Notice: In v5.1, z_cb = 0.5 * z_direct + 0.5 * (p_buy^1.1 * z_cond)
    # And z_gru = (p_buy^1.1 * z_cond)
    # For alpha variation, we reconstruct with alpha=1.0 and alpha=0.9
    # Ratio scaling: z_alpha = z_v51 * (p_buy^alpha / p_buy^1.1)
    # To be numerically exact, let's load components or apply power transform:

    # Extract effective p_buy from z_gru (since z_gru = p^1.1 * z_cond where z_cond ~ 5.2):
    # p_eff = (z_gru / 5.2) ^ (1/1.1)
    # Scaling factor for alpha: factor = p_eff ^ (alpha - 1.1)

    # -------------------------------------------------------------------------
    # Submission v5.2: alpha = 1.0 (Less zero-shrinkage)
    # -------------------------------------------------------------------------
    # In log space, changing alpha from 1.1 to 1.0 increases log GMV by ~ +0.05 on active users
    # We apply exact alpha=1.0 to both components:
    scale_10 = np.where(z_gru_test > 0.01, np.power(np.clip(z_gru_test / 5.2, 0.05, 1.0), 1.0 - 1.1), 1.0)
    z_gru_10 = (z_gru_test * scale_10).astype(np.float32)
    z_cb_10 = (z_cb_test * scale_10).astype(np.float32)

    z_final_10 = (0.50 * z_cb_10 + 0.50 * z_gru_10).astype(np.float32)
    y_pred_10 = np.clip(np.expm1(z_final_10), 0.0, None)

    # -------------------------------------------------------------------------
    # Submission v5.3: alpha = 0.9 (More aggressive holiday scale)
    # -------------------------------------------------------------------------
    scale_09 = np.where(z_gru_test > 0.01, np.power(np.clip(z_gru_test / 5.2, 0.05, 1.0), 0.9 - 1.1), 1.0)
    z_gru_09 = (z_gru_test * scale_09).astype(np.float32)
    z_cb_09 = (z_cb_test * scale_09).astype(np.float32)

    z_final_09 = (0.50 * z_cb_09 + 0.50 * z_gru_09).astype(np.float32)
    y_pred_09 = np.clip(np.expm1(z_final_09), 0.0, None)

    # Compare distributions
    stats_v51 = calculate_quantiles_dict(y_pred_v51)
    stats_10 = calculate_quantiles_dict(y_pred_10)
    stats_09 = calculate_quantiles_dict(y_pred_09)

    alpha_comparison = {
        "v5.1 (alpha=1.1, LB=1.68029)": stats_v51,
        "v5.2 (alpha=1.0)": stats_10,
        "v5.3 (alpha=0.9)": stats_09,
    }

    with open(AUDIT_DIR / "alpha_diagnostic_comparison.json", "w", encoding="utf-8") as f:
        json.dump(alpha_comparison, f, indent=2)

    # Save CSVs
    sub_v52_path = DATA_DIR / "submission_v5_2_alpha10.csv"
    sub_v53_path = DATA_DIR / "submission_v5_3_alpha09.csv"

    pl.DataFrame({"user_id": user_ids, "predict": y_pred_10}).write_csv(sub_v52_path)
    pl.DataFrame({"user_id": user_ids, "predict": y_pred_09}).write_csv(sub_v53_path)

    # Copy v5.2 as the primary submission.csv
    pl.DataFrame({"user_id": user_ids, "predict": y_pred_10}).write_csv(DATA_DIR / "submission.csv")

    print("\n[*] Diagnostic Alpha Submissions Created:")
    print(f"    1. Submission v5.1 (alpha=1.1, Reference): Mean={stats_v51['mean']:.2f} | P50={stats_v51['p50']:.2f} | P99={stats_v51['p99']:.2f} | % < 1 rub: {stats_v51['pct_below_1_rub']:.1f}%")
    print(f"    2. Submission v5.2 (alpha=1.0, Saved):     Mean={stats_10['mean']:.2f} | P50={stats_10['p50']:.2f} | P99={stats_10['p99']:.2f} | % < 1 rub: {stats_10['pct_below_1_rub']:.1f}%")
    print(f"    3. Submission v5.3 (alpha=0.9, Saved):     Mean={stats_09['mean']:.2f} | P50={stats_09['p50']:.2f} | P99={stats_09['p99']:.2f} | % < 1 rub: {stats_09['pct_below_1_rub']:.1f}%")
    print(f"\n[+] data/submission.csv is now populated with Submission v5.2 (alpha=1.0) ready for upload!")


if __name__ == "__main__":
    main()
