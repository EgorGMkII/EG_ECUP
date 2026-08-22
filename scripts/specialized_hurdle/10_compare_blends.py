"""Script 10: Compare Pre-Defined Blend Combinations on January Holdout."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl


def main():
    print("=" * 80)
    print("10: COMPARE FIXED BLENDS ON JANUARY HOLDOUT")
    print("=" * 80)

    reports_dir = Path("artifacts/specialized_hurdle/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    snap_val = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
    users_100k = pl.read_parquet("artifacts/selected_users_100k.parquet")["user_id"].to_numpy()
    u_map = {u: i for i, u in enumerate(snap_val["user_id"].to_list())}
    order = [u_map[u] for u in users_100k]

    y_rub = snap_val["target"].to_numpy()[order].astype(np.float32)
    past_gmv = snap_val["lifetime_gmv"].to_numpy()[order].astype(np.float32)
    was_act = (past_gmv > 0).astype(int)
    will_buy = (y_rub > 0).astype(int)
    z_true = np.log1p(np.maximum(0.0, y_rub))

    # Load candidate components
    router_df = pl.read_parquet("artifacts/s1_s2_router/router_val_predictions.parquet")
    u_map_r = {u: i for i, u in enumerate(router_df["user_id"].to_list())}
    order_r = [u_map_r[u] for u in users_100k]
    z_r3 = router_df["z_r3"].to_numpy()[order_r]

    ett_df = pl.read_parquet("artifacts/ett_optimization/OPT_LR0/validation_predictions.parquet")
    z_ett = ett_df["pred_factorized_z"].to_numpy()

    cb_df = pl.read_parquet("artifacts/val_predictions_cv3.parquet")
    u_map_cb = {u: i for i, u in enumerate(cb_df["user_id"].to_list())}
    order_cb = [u_map_cb[u] for u in users_100k]
    z_cb = np.log1p(np.maximum(0.0, cb_df["pred_hurdle"].to_numpy()[order_cb]))

    blends = {
        "CatBoost + R3 + ETT (Equal)": (z_cb + z_r3 + z_ett) / 3.0,
        "R3 + ETT (Equal)": (z_r3 + z_ett) / 2.0,
        "CatBoost + ETT (Equal)": (z_cb + z_ett) / 2.0,
        "Constrained Blend (35% R3 + 65% ETT)": 0.35 * z_r3 + 0.65 * z_ett,
    }

    blend_records = []
    for name, z in blends.items():
        gmv_pred = np.expm1(np.maximum(0.0, z))
        rmsle = float(np.sqrt(np.mean((np.log1p(gmv_pred) - np.log1p(y_rub)) ** 2)))

        st_0_0 = (was_act == 0) & (will_buy == 0)
        st_0_1 = (was_act == 0) & (will_buy == 1)
        st_1_0 = (was_act == 1) & (will_buy == 0)
        st_1_1 = (was_act == 1) & (will_buy == 1)

        mse_0_0 = float(np.mean((z[st_0_0] - z_true[st_0_0]) ** 2))
        mse_0_1 = float(np.mean((z[st_0_1] - z_true[st_0_1]) ** 2))
        mse_1_0 = float(np.mean((z[st_1_0] - z_true[st_1_0]) ** 2))
        mse_1_1 = float(np.mean((z[st_1_1] - z_true[st_1_1]) ** 2))

        print(f"[{name:38s}] RMSLE: {rmsle:.5f} | 0->0: {mse_0_0:.4f} | 0->1: {mse_0_1:.4f} | 1->0: {mse_1_0:.4f} | 1->1: {mse_1_1:.4f}")

        blend_records.append({
            "blend_name": name,
            "january_rmsle": rmsle,
            "mse_0_0": mse_0_0,
            "mse_0_1": mse_0_1,
            "mse_1_0": mse_1_0,
            "mse_1_1": mse_1_1,
        })

    df_blends = pl.DataFrame(blend_records)
    out_csv = reports_dir / "blends_comparison_january.csv"
    df_blends.write_csv(out_csv)
    print(f"\n[+] Saved blends comparison to {out_csv}")


if __name__ == "__main__":
    main()
