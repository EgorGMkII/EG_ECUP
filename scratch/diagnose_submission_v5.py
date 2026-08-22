"""Forensic diagnosis of Submission v5 failure (1.876 on LB)."""

import numpy as np
import polars as pl
from pathlib import Path

def main():
    print("===================================================================")
    print("=== DIAGNOSIS OF SUBMISSION V5 FAILURE ===")
    print("===================================================================")

    sub_v5 = pl.read_csv("data/submission.csv")
    p_v5 = sub_v5["predict"].to_numpy()

    print(f"Submission v5 Stats (Score: 1.876):")
    print(f"  Count: {len(p_v5):,}")
    print(f"  Min:    {np.min(p_v5):.4f}")
    print(f"  P1:     {np.percentile(p_v5, 1):.4f}")
    print(f"  P10:    {np.percentile(p_v5, 10):.4f}")
    print(f"  P50:    {np.percentile(p_v5, 50):.4f}")
    print(f"  Mean:   {np.mean(p_v5):.4f}")
    print(f"  P90:    {np.percentile(p_v5, 90):.4f}")
    print(f"  P99:    {np.percentile(p_v5, 99):.4f}")
    print(f"  Max:    {np.max(p_v5):.4f}")

    # Check Submission v4 (Score: 1.693) if available
    # Or previous valid predictions
    if Path("artifacts/transitions/baseline_cv3_audit.parquet").exists():
        audit = pl.read_parquet("artifacts/transitions/baseline_cv3_audit.parquet")
        z_cb = audit["z_pred_catboost"].to_numpy()
        z_gru = audit["z_pred_gru"].to_numpy()
        z_ens = audit["z_pred_ensemble_v4"].to_numpy()
        p_ens = np.expm1(z_ens)

        print(f"\nBaseline Validation v4 Stats (Score 1.693):")
        print(f"  P10:    {np.percentile(p_ens, 10):.4f}")
        print(f"  P50:    {np.percentile(p_ens, 50):.4f}")
        print(f"  Mean:   {np.mean(p_ens):.4f}")
        print(f"  P90:    {np.percentile(p_ens, 90):.4f}")
        print(f"  P99:    {np.percentile(p_ens, 99):.4f}")
        print(f"  Max:    {np.max(p_ens):.4f}")

if __name__ == "__main__":
    main()
