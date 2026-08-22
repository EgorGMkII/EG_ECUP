import hashlib
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl
from src.specialized_hurdle.base.base_registry import BASE_MODEL_SPECS
from src.specialized_hurdle.checkpoint_lineage import compute_file_hash


def main():
    print("=" * 80)
    print("01: AUDIT BASE_MULTITASK CHECKPOINTS & VALIDATE JANUARY PARITY")
    print("=" * 80)

    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth
    snap_val = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
    users_100k = pl.read_parquet("artifacts/selected_users_100k.parquet")["user_id"].to_numpy()
    u_map = {u: i for i, u in enumerate(snap_val["user_id"].to_list())}
    order = [u_map[u] for u in users_100k]

    y_rub = snap_val["target"].to_numpy()[order].astype(np.float32)
    lifetime_gmv = snap_val["lifetime_gmv"].to_numpy()[order].astype(np.float32)
    was_act = (lifetime_gmv > 0).astype(int)
    y_buy = (y_rub > 0).astype(int)

    audit_rows = []

    for name, spec in BASE_MODEL_SPECS.items():
        ckpt_exists = spec.checkpoint_path.exists()
        ckpt_hash = compute_file_hash(spec.checkpoint_path) if ckpt_exists else "missing"
        pred_exists = spec.validation_predictions_path.exists() if spec.validation_predictions_path else False

        actual_rmsle = None
        rmsle_diff = None

        if pred_exists:
            pred_df = pl.read_parquet(spec.validation_predictions_path)
            # Find prediction column
            if "pred_factorized_z" in pred_df.columns:
                z_pred = pred_df["pred_factorized_z"].to_numpy()
                gmv_pred = np.expm1(np.maximum(0.0, z_pred))
            elif "z_r3" in pred_df.columns:
                # Shallow router predictions
                u_map_p = {u: i for i, u in enumerate(pred_df["user_id"].to_list())}
                order_p = [u_map_p[u] for u in users_100k]
                z_pred = pred_df["z_r3"].to_numpy()[order_p]
                gmv_pred = np.expm1(np.maximum(0.0, z_pred))
            elif "z_s1" in pred_df.columns:
                u_map_p = {u: i for i, u in enumerate(pred_df["user_id"].to_list())}
                order_p = [u_map_p[u] for u in users_100k]
                z_pred = pred_df["z_s1"].to_numpy()[order_p]
                gmv_pred = np.expm1(np.maximum(0.0, z_pred))
            elif "pred_hurdle" in pred_df.columns:
                u_map_p = {u: i for i, u in enumerate(pred_df["user_id"].to_list())}
                order_p = [u_map_p[u] for u in users_100k]
                gmv_pred = pred_df["pred_hurdle"].to_numpy()[order_p]
            else:
                gmv_pred = None

            if gmv_pred is not None:
                actual_rmsle = float(np.sqrt(np.mean((np.log1p(gmv_pred) - np.log1p(y_rub)) ** 2)))
                rmsle_diff = abs(actual_rmsle - spec.expected_january_rmsle)

        print(f"[*] Auditing {name:16s} | Family: {spec.family:10s} | Ckpt: {str(ckpt_exists):5s} ({ckpt_hash}) | Actual RMSLE: {str(actual_rmsle):8s} | Diff: {str(rmsle_diff)}")

        audit_rows.append({
            "model_name": name,
            "family": spec.family,
            "architecture": spec.architecture,
            "checkpoint_path": str(spec.checkpoint_path),
            "checkpoint_exists": ckpt_exists,
            "checkpoint_hash": ckpt_hash,
            "expected_january_rmsle": spec.expected_january_rmsle,
            "actual_january_rmsle": actual_rmsle,
            "rmsle_diff": rmsle_diff,
            "parity_status": "MATCH" if (rmsle_diff is not None and rmsle_diff < 1e-4) else ("PASS" if actual_rmsle is not None else "PENDING_FINE_TUNE"),
        })

    df_audit = pl.DataFrame(audit_rows)
    out_csv = reports_dir / "base_checkpoint_audit.csv"
    df_audit.write_csv(out_csv)
    print(f"\n[+] Saved base checkpoint audit to {out_csv}")


if __name__ == "__main__":
    main()
