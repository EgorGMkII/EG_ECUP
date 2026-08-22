"""Evaluate Constrained Blending: CatBoost B1 + Shallow Router R3 + Optimized ETT1.

Evaluates weights on 100k canonical users across:
- Equal weighting
- Grid search over simplex (w_cb, w_router, w_ett) with w_i >= 0, sum(w) = 1
- Transition state breakdown (0->0, 0->1, 1->0, 1->1)
"""

import numpy as np
import polars as pl

# 1. Load validation ground truth
snap_val = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
users_100k = pl.read_parquet("artifacts/selected_users_100k.parquet")["user_id"].to_numpy()

# Align snap_val with users_100k
u_map = {u: i for i, u in enumerate(snap_val["user_id"].to_list())}
order = [u_map[u] for u in users_100k]

y_rub = snap_val["target"].to_numpy()[order].astype(np.float32)
lifetime_gmv = snap_val["lifetime_gmv"].to_numpy()[order].astype(np.float32)
was_act = (lifetime_gmv > 0).astype(int)
y_buy = (y_rub > 0).astype(int)
y_z_true = np.log1p(np.maximum(0.0, y_rub))

# 2. Load Optimized ETT predictions
ett_lr0_df = pl.read_parquet("artifacts/ett_optimization/OPT_LR0/validation_predictions.parquet")
z_ett_lr0 = ett_lr0_df["pred_factorized_z"].to_numpy()

ett_max256_df = pl.read_parquet("artifacts/ett_optimization/OPT_MAX256/validation_predictions.parquet")
z_ett_max256 = ett_max256_df["pred_factorized_z"].to_numpy()

# 3. Load Shallow Router R3 predictions
router_df = pl.read_parquet("artifacts/s1_s2_router/router_val_predictions.parquet")
# Align router_df with users_100k
u_map_r = {u: i for i, u in enumerate(router_df["user_id"].to_list())}
order_r = [u_map_r[u] for u in users_100k]
z_router = router_df["z_r3"].to_numpy()[order_r]

# 4. Load CatBoost B1 predictions
cb_df = pl.read_parquet("artifacts/val_predictions_cv3.parquet")
u_map_cb = {u: i for i, u in enumerate(cb_df["user_id"].to_list())}
order_cb = [u_map_cb[u] for u in users_100k]
z_cb = np.log1p(np.maximum(0.0, cb_df["pred_hurdle"].to_numpy()[order_cb]))

def calc_rmsle(z_pred, y_true_rub):
    gmv_pred = np.expm1(np.maximum(0.0, z_pred))
    return float(np.sqrt(np.mean((np.log1p(gmv_pred) - np.log1p(y_true_rub)) ** 2)))

def calc_state_breakdown(z_pred, y_true_z, was_active, y_buy):
    metrics = {}
    for a in [0, 1]:
        for b in [0, 1]:
            mask = (was_active == a) & (y_buy == b)
            if mask.sum() > 0:
                metrics[f"{a}->{b}"] = float(np.mean((z_pred[mask] - y_true_z[mask]) ** 2))
    return metrics

print("=" * 80)
print("INDIVIDUAL MODEL PERFORMANCE (Factorized / Log-space Z)")
print("=" * 80)

models = {
    "CatBoost B1": z_cb,
    "Shallow Router R3": z_router,
    "Optimized ETT1 (OPT_LR0, 128 tok)": z_ett_lr0,
    "Optimized ETT1 (OPT_MAX256, 256 tok)": z_ett_max256,
}

for name, z in models.items():
    rmsle = calc_rmsle(z, y_rub)
    st = calc_state_breakdown(z, y_z_true, was_act, y_buy)
    print(f"{name:38s} | RMSLE = {rmsle:.5f} | 0->0: {st.get('0->0', 0):.4f} | 0->1: {st.get('0->1', 0):.4f} | 1->0: {st.get('1->0', 0):.4f} | 1->1: {st.get('1->1', 0):.4f}")

# ETT Seed/Length Ensemble
z_ett_ens = 0.5 * z_ett_lr0 + 0.5 * z_ett_max256
rmsle_ett_ens = calc_rmsle(z_ett_ens, y_rub)
st_ett_ens = calc_state_breakdown(z_ett_ens, y_z_true, was_act, y_buy)
print(f"{'ETT 2-Model Ensemble (128 + 256)':38s} | RMSLE = {rmsle_ett_ens:.5f} | 0->0: {st_ett_ens.get('0->0', 0):.4f} | 0->1: {st_ett_ens.get('0->1', 0):.4f} | 1->0: {st_ett_ens.get('1->0', 0):.4f} | 1->1: {st_ett_ens.get('1->1', 0):.4f}")

print("\n" + "=" * 80)
print("CONSTRAINED BLENDING GRID SEARCH (w_cb + w_router + w_ett = 1.0)")
print("=" * 80)

best_blend_rmsle = 999.0
best_blend_weights = None
best_blend_z = None

# Grid search with step 0.05
for w_cb in np.linspace(0.0, 0.5, 11):
    for w_r in np.linspace(0.0, 0.8, 17):
        w_ett = 1.0 - w_cb - w_r
        if w_ett < 0.0:
            continue
        z_blend = w_cb * z_cb + w_r * z_router + w_ett * z_ett_ens
        rmsle = calc_rmsle(z_blend, y_rub)
        if rmsle < best_blend_rmsle:
            best_blend_rmsle = rmsle
            best_blend_weights = (w_cb, w_r, w_ett)
            best_blend_z = z_blend

w_cb, w_r, w_ett = best_blend_weights
st_blend = calc_state_breakdown(best_blend_z, y_z_true, was_act, y_buy)

print(f"Optimal Weights: CatBoost={w_cb:.2f}, Shallow Router={w_r:.2f}, ETT={w_ett:.2f}")
print(f"BEST BLEND RMSLE = {best_blend_rmsle:.5f}")
print(f"State Breakdown: 0->0: {st_blend.get('0->0', 0):.4f} | 0->1: {st_blend.get('0->1', 0):.4f} | 1->0: {st_blend.get('1->0', 0):.4f} | 1->1: {st_blend.get('1->1', 0):.4f}")

# Save blend results
blend_summary = [{
    "blend_name": "CatBoost_Router_ETT_Optimal_Blend",
    "w_catboost": w_cb,
    "w_router": w_r,
    "w_ett": w_ett,
    "best_rmsle": best_blend_rmsle,
    "mse_0_0": st_blend.get("0->0", 0.0),
    "mse_0_1": st_blend.get("0->1", 0.0),
    "mse_1_0": st_blend.get("1->0", 0.0),
    "mse_1_1": st_blend.get("1->1", 0.0),
}]
pl.DataFrame(blend_summary).write_csv("artifacts/ett_optimization/optimal_blend_summary.csv")
print("\n[+] Saved optimal blend summary to artifacts/ett_optimization/optimal_blend_summary.csv")
