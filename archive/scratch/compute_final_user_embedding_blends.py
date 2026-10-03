"""Compute honest ensembles, transitions, calibration, and generate all diagnostic plots."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path("artifacts/user_embedding")
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load ground truth from snapshot
val_snap = pl.read_parquet("data/snapshots/snapshot_2026-01-14.parquet")
y_true_rub = val_snap["target"].to_numpy().astype(np.float64)
y_true_log = np.log1p(y_true_rub)
past_buyer = (val_snap["gmv_sum_30d"].to_numpy() > 0).astype(np.int32)
fut_buyer = (y_true_rub > 0).astype(np.int32)

# Load CatBoost predictions
cb_c4 = pl.read_parquet("artifacts/btyd_audit/predictions_B0.parquet")["factorized_z"].to_numpy().astype(np.float64)
cb_b1 = pl.read_parquet("artifacts/btyd_audit/predictions_B1.parquet")["factorized_z"].to_numpy().astype(np.float64)

c4_rmsle = float(np.sqrt(np.mean((cb_c4 - y_true_log) ** 2)))
b1_rmsle = float(np.sqrt(np.mean((cb_b1 - y_true_log) ** 2)))
print(f"[*] CatBoost C4 Solo RMSLE: {c4_rmsle:.5f}")
print(f"[*] CatBoost B1 Solo RMSLE: {b1_rmsle:.5f}")

# Load GRU predictions for all variants and seeds
variants = ["E0", "E1", "E2", "E3", "E2_shuffled"]
seeds = [42, 43, 44]

preds = {}
for var in variants:
    for s in seeds:
        if var == "E2_shuffled" and s != 42:
            continue
        p_path = ROOT / f"{var}_seed{s}" / "validation_predictions.parquet"
        df = pl.read_parquet(p_path)
        preds[f"{var}_seed{s}"] = {
            "z_fact": df["factorized_z"].to_numpy().astype(np.float64),
            "p_react": df["p_react"].to_numpy().astype(np.float64),
            "p_churn": df["p_churn"].to_numpy().astype(np.float64),
            "p_buy": df["p_buy"].to_numpy().astype(np.float64),
            "z_cond": df["conditional_z"].to_numpy().astype(np.float64),
        }

# Compute 3-seed ensemble predictions
ens_preds = {}
for var in ["E0", "E1", "E2", "E3"]:
    z_stack = np.stack([preds[f"{var}_seed{s}"]["z_fact"] for s in seeds], axis=0)
    p_react_stack = np.stack([preds[f"{var}_seed{s}"]["p_react"] for s in seeds], axis=0)
    p_churn_stack = np.stack([preds[f"{var}_seed{s}"]["p_churn"] for s in seeds], axis=0)
    p_buy_stack = np.stack([preds[f"{var}_seed{s}"]["p_buy"] for s in seeds], axis=0)
    z_cond_stack = np.stack([preds[f"{var}_seed{s}"]["z_cond"] for s in seeds], axis=0)

    ens_preds[var] = {
        "z_fact": np.mean(z_stack, axis=0),
        "p_react": np.mean(p_react_stack, axis=0),
        "p_churn": np.mean(p_churn_stack, axis=0),
        "p_buy": np.mean(p_buy_stack, axis=0),
        "z_cond": np.mean(z_cond_stack, axis=0),
    }

# Compute multi-seed summary and blends
seed_summary_rows = []
for var in ["E0", "E1", "E2", "E3"]:
    solo_rmsles = [float(np.sqrt(np.mean((preds[f"{var}_seed{s}"]["z_fact"] - y_true_log) ** 2))) for s in seeds]
    e0_solo_rmsles = [float(np.sqrt(np.mean((preds[f"E0_seed{s}"]["z_fact"] - y_true_log) ** 2))) for s in seeds]
    deltas = [solo_rmsles[i] - e0_solo_rmsles[i] for i in range(len(seeds))]

    ens_solo = float(np.sqrt(np.mean((ens_preds[var]["z_fact"] - y_true_log) ** 2)))
    
    # 50/50 blend with CatBoost C4 and B1
    blend_c4 = float(np.sqrt(np.mean(((0.5 * cb_c4 + 0.5 * ens_preds[var]["z_fact"]) - y_true_log) ** 2)))
    blend_b1 = float(np.sqrt(np.mean(((0.5 * cb_b1 + 0.5 * ens_preds[var]["z_fact"]) - y_true_log) ** 2)))

    # Optimal weight search (w_cb * cb + (1 - w_cb) * gru)
    weights = np.linspace(0.0, 1.0, 101)
    best_w_c4, best_score_c4 = 0.5, 999.0
    for w in weights:
        sc = float(np.sqrt(np.mean(((w * cb_c4 + (1 - w) * ens_preds[var]["z_fact"]) - y_true_log) ** 2)))
        if sc < best_score_c4:
            best_score_c4 = sc
            best_w_c4 = w

    best_w_b1, best_score_b1 = 0.5, 999.0
    for w in weights:
        sc = float(np.sqrt(np.mean(((w * cb_b1 + (1 - w) * ens_preds[var]["z_fact"]) - y_true_log) ** 2)))
        if sc < best_score_b1:
            best_score_b1 = sc
            best_w_b1 = w

    react_aucs = [float(roc_auc_score(fut_buyer[past_buyer == 0], preds[f"{var}_seed{s}"]["p_react"][past_buyer == 0])) for s in seeds]
    churn_aucs = [float(roc_auc_score((1 - fut_buyer)[past_buyer == 1], preds[f"{var}_seed{s}"]["p_churn"][past_buyer == 1])) for s in seeds]
    briers = [float(brier_score_loss(fut_buyer, preds[f"{var}_seed{s}"]["p_buy"])) for s in seeds]

    row = {
        "variant": var,
        "mean_solo_rmsle": float(np.mean(solo_rmsles)),
        "std_solo_rmsle": float(np.std(solo_rmsles)),
        "mean_delta_vs_e0": float(np.mean(deltas)),
        "delta_seed42": deltas[0],
        "delta_seed43": deltas[1],
        "delta_seed44": deltas[2],
        "ensemble_3seed_rmsle": ens_solo,
        "ensemble_blend_c4_rmsle_50_50": blend_c4,
        "optimal_blend_c4_rmsle": best_score_c4,
        "optimal_w_cb_c4": best_w_c4,
        "ensemble_blend_b1_rmsle_50_50": blend_b1,
        "optimal_blend_b1_rmsle": best_score_b1,
        "optimal_w_cb_b1": best_w_b1,
        "mean_react_auc": float(np.mean(react_aucs)),
        "mean_churn_auc": float(np.mean(churn_aucs)),
        "mean_overall_brier": float(np.mean(briers)),
    }
    seed_summary_rows.append(row)

pl.DataFrame(seed_summary_rows).write_csv(ROOT / "seed_summary.csv")
print("[+] Updated seed_summary.csv with exact CatBoost blends")

# Save blend summary table
blend_table = [
    {"Model / Blend": "CatBoost C4 Solo", "Validation RMSLE": c4_rmsle, "Delta vs C4 Solo": 0.0},
    {"Model / Blend": "CatBoost B1 (BTYD) Solo", "Validation RMSLE": b1_rmsle, "Delta vs C4 Solo": b1_rmsle - c4_rmsle},
    {"Model / Blend": "GRU E0 (3-seed)", "Validation RMSLE": ens_preds["E0_rmsle"] if "E0_rmsle" in ens_preds else seed_summary_rows[0]["ensemble_3seed_rmsle"], "Delta vs C4 Solo": seed_summary_rows[0]["ensemble_3seed_rmsle"] - c4_rmsle},
    {"Model / Blend": "GRU E1 (3-seed)", "Validation RMSLE": seed_summary_rows[1]["ensemble_3seed_rmsle"], "Delta vs C4 Solo": seed_summary_rows[1]["ensemble_3seed_rmsle"] - c4_rmsle},
    {"Model / Blend": "GRU E2 (3-seed)", "Validation RMSLE": seed_summary_rows[2]["ensemble_3seed_rmsle"], "Delta vs C4 Solo": seed_summary_rows[2]["ensemble_3seed_rmsle"] - c4_rmsle},
    {"Model / Blend": "GRU E3 (3-seed)", "Validation RMSLE": seed_summary_rows[3]["ensemble_3seed_rmsle"], "Delta vs C4 Solo": seed_summary_rows[3]["ensemble_3seed_rmsle"] - c4_rmsle},
    {"Model / Blend": "CatBoost C4 + GRU E0 (50/50)", "Validation RMSLE": seed_summary_rows[0]["ensemble_blend_c4_rmsle_50_50"], "Delta vs C4 Solo": seed_summary_rows[0]["ensemble_blend_c4_rmsle_50_50"] - c4_rmsle},
    {"Model / Blend": "CatBoost C4 + GRU E2 (Optimal)", "Validation RMSLE": seed_summary_rows[2]["optimal_blend_c4_rmsle"], "Delta vs C4 Solo": seed_summary_rows[2]["optimal_blend_c4_rmsle"] - c4_rmsle},
    {"Model / Blend": "CatBoost B1 + GRU E2 (Optimal)", "Validation RMSLE": seed_summary_rows[2]["optimal_blend_b1_rmsle"], "Delta vs C4 Solo": seed_summary_rows[2]["optimal_blend_b1_rmsle"] - c4_rmsle},
    {"Model / Blend": "CatBoost B1 + GRU E3 (Optimal)", "Validation RMSLE": seed_summary_rows[3]["optimal_blend_b1_rmsle"], "Delta vs C4 Solo": seed_summary_rows[3]["optimal_blend_b1_rmsle"] - c4_rmsle},
]
pl.DataFrame(blend_table).write_csv(ROOT / "blend_summary.csv")
print("[+] Updated blend_summary.csv")

# -----------------------------------------------------------------------------
# PLOT 1: RMSLE Across Variants and Seeds
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=150)
labels = ["E0 (Base)", "E1 (Biases)", "E2 (UserEmb)", "E3 (FullEmb)", "E2 Shuffled"]
s42 = [float(np.sqrt(np.mean((preds[f"{v}_seed42"]["z_fact"] - y_true_log) ** 2))) for v in variants]
s43 = [float(np.sqrt(np.mean((preds[f"{v}_seed43"]["z_fact"] - y_true_log) ** 2))) if v != "E2_shuffled" else np.nan for v in variants]
s44 = [float(np.sqrt(np.mean((preds[f"{v}_seed44"]["z_fact"] - y_true_log) ** 2))) if v != "E2_shuffled" else np.nan for v in variants]
ens_3s = [seed_summary_rows[i]["ensemble_3seed_rmsle"] if i < 4 else np.nan for i in range(5)]

x = np.arange(len(labels))
width = 0.2
plt.bar(x - 1.5 * width, s42, width, label="Seed 42", color="#4285F4", alpha=0.85)
plt.bar(x - 0.5 * width, s43, width, label="Seed 43", color="#34A853", alpha=0.85)
plt.bar(x + 0.5 * width, s44, width, label="Seed 44", color="#FBBC05", alpha=0.85)
plt.bar(x + 1.5 * width, ens_3s, width, label="3-Seed Ensemble", color="#EA4335", alpha=0.9)

plt.axhline(c4_rmsle, color="black", linestyle="--", linewidth=1.5, label=f"CatBoost C4 Solo ({c4_rmsle:.4f})")
plt.axhline(b1_rmsle, color="purple", linestyle=":", linewidth=1.5, label=f"CatBoost B1 Solo ({b1_rmsle:.4f})")

plt.ylabel("Validation RMSLE (Lower is better)", fontsize=12)
plt.title("User Embedding GRU-180: Solo Performance by Architecture Variant & Seed", fontsize=14, fontweight="bold")
plt.xticks(x, labels, fontsize=11)
plt.ylim(1.58, 1.86)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(frameon=True, fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rmsle_variants_and_seeds.png")
plt.close()

# -----------------------------------------------------------------------------
# PLOT 2: Transition MSE Decomposition
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=150)
m00 = (past_buyer == 0) & (fut_buyer == 0)
m01 = (past_buyer == 0) & (fut_buyer == 1)
m10 = (past_buyer == 1) & (fut_buyer == 0)
m11 = (past_buyer == 1) & (fut_buyer == 1)

def get_trans_mse(z_arr):
    diff = (z_arr - y_true_log) ** 2
    return [np.mean(diff[m00]), np.mean(diff[m01]), np.mean(diff[m10]), np.mean(diff[m11])]

mse_e0 = get_trans_mse(ens_preds["E0"]["z_fact"])
mse_e1 = get_trans_mse(ens_preds["E1"]["z_fact"])
mse_e2 = get_trans_mse(ens_preds["E2"]["z_fact"])
mse_e3 = get_trans_mse(ens_preds["E3"]["z_fact"])
mse_shuf = get_trans_mse(preds["E2_shuffled_seed42"]["z_fact"])

t_labels = ["0 -> 0 (Dormant)", "0 -> 1 (Reactivation)", "1 -> 0 (Churn)", "1 -> 1 (Retention)"]
xt = np.arange(len(t_labels))
w = 0.16

plt.bar(xt - 2*w, mse_e0, w, label="E0 Base", color="#9E9E9E")
plt.bar(xt - w, mse_e1, w, label="E1 Biases", color="#42A5F5")
plt.bar(xt, mse_e2, w, label="E2 UserEmb", color="#2E7D32")
plt.bar(xt + w, mse_e3, w, label="E3 FullEmb", color="#E65100")
plt.bar(xt + 2*w, mse_shuf, w, label="E2 Shuffled", color="#D32F2F", hatch="//")

plt.ylabel("MSE on log(1 + Y)", fontsize=12)
plt.title("Transition State Error Decomposition (E0 vs E1 vs E2 vs E3 vs Shuffled)", fontsize=13, fontweight="bold")
plt.xticks(xt, t_labels, fontsize=11)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "transition_mse_decomposition.png")
plt.close()

# -----------------------------------------------------------------------------
# PLOT 3: Calibration Curves
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 5), dpi=150)

# Reactivation calibration
plt.subplot(1, 2, 1)
bins = np.linspace(0, 1, 11)
for name, p_dict, col in [("E0 Base", ens_preds["E0"], "#9E9E9E"), ("E2 UserEmb", ens_preds["E2"], "#2E7D32"), ("E3 FullEmb", ens_preds["E3"], "#E65100")]:
    p_r = p_dict["p_react"][past_buyer == 0]
    y_r = fut_buyer[past_buyer == 0]
    b_centers, b_actual = [], []
    for b_l, b_r in zip(bins[:-1], bins[1:]):
        mask = (p_r >= b_l) & (p_r < b_r)
        if np.sum(mask) > 50:
            b_centers.append(np.mean(p_r[mask]))
            b_actual.append(np.mean(y_r[mask]))
    plt.plot(b_centers, b_actual, "o-", label=name, color=col, linewidth=2)

plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
plt.xlabel("Predicted Reactivation Probability", fontsize=11)
plt.ylabel("Observed Purchase Rate (0 -> 1)", fontsize=11)
plt.title("Reactivation Calibration (Dormant Users)", fontsize=12, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# Churn calibration
plt.subplot(1, 2, 2)
for name, p_dict, col in [("E0 Base", ens_preds["E0"], "#9E9E9E"), ("E2 UserEmb", ens_preds["E2"], "#2E7D32"), ("E3 FullEmb", ens_preds["E3"], "#E65100")]:
    p_c = p_dict["p_churn"][past_buyer == 1]
    y_c = (1 - fut_buyer)[past_buyer == 1]
    b_centers, b_actual = [], []
    for b_l, b_r in zip(bins[:-1], bins[1:]):
        mask = (p_c >= b_l) & (p_c < b_r)
        if np.sum(mask) > 50:
            b_centers.append(np.mean(p_c[mask]))
            b_actual.append(np.mean(y_c[mask]))
    plt.plot(b_centers, b_actual, "s-", label=name, color=col, linewidth=2)

plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
plt.xlabel("Predicted Churn Probability", fontsize=11)
plt.ylabel("Observed Churn Rate (1 -> 0)", fontsize=11)
plt.title("Churn Calibration (Active Users)", fontsize=12, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "calibration_curves.png")
plt.close()

# -----------------------------------------------------------------------------
# PLOT 4: Scatter Predict vs GT
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 6), dpi=150)
np.random.seed(42)
sample_idx = np.random.choice(len(y_true_log), size=10000, replace=False)
plt.scatter(y_true_log[sample_idx], ens_preds["E2"]["z_fact"][sample_idx], alpha=0.15, s=8, color="#1E88E5", label="GRU E2 Predictions")
plt.plot([0, 12], [0, 12], "r--", linewidth=1.5, label="y_pred = y_true")
plt.xlabel("Ground Truth ln(1 + GMV)", fontsize=11)
plt.ylabel("Predicted ln(1 + GMV) (GRU E2)", fontsize=11)
plt.title("Validation Out-of-Time Predictions vs Ground Truth (10k sample)", fontsize=12, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scatter_predict_vs_gt.png")
plt.close()

print("[+] All diagnostic plots successfully generated in artifacts/user_embedding/plots/")
