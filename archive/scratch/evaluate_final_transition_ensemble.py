"""Unified Evaluation of Long-Sequence Encoders + CatBoost Transitions Ensemble."""

from pathlib import Path
import numpy as np
import polars as pl

from src.transitions.metrics import decompose_mse_by_transitions

TRANSITIONS_ARTIFACTS = Path("artifacts/transitions")


def main():
    print("===================================================================")
    print("=== UNIFIED ENSEMBLE EVALUATION ON TRANSITION STATES ===")
    print("===================================================================")

    # 1. Load Baseline Canonical Audit
    audit_df = pl.read_parquet(TRANSITIONS_ARTIFACTS / "baseline_cv3_audit.parquet")
    y_true = audit_df["target"].to_numpy().astype(np.float32)
    past_buyer = audit_df["past_buyer_30d"].to_numpy().astype(np.int32)
    z_baseline_ens = audit_df["z_pred_ensemble_v4"].to_numpy().astype(np.float32)
    z_catboost_base = audit_df["z_pred_catboost"].to_numpy().astype(np.float32)

    # 2. Load CatBoost Transitions A3
    exp_a_df = pl.read_parquet(TRANSITIONS_ARTIFACTS / "experiment_A_predictions.parquet")
    z_cb_trans = exp_a_df["z_factorized_a3"].to_numpy().astype(np.float32)

    # 3. Load Long Sequence Predictions
    exp_seq_df = pl.read_parquet(TRANSITIONS_ARTIFACTS / "experiment_long_seq_predictions.parquet")
    z_gru90 = exp_seq_df["z_gru90_fact"].to_numpy().astype(np.float32)
    z_gru365 = exp_seq_df["z_gru365_fact"].to_numpy().astype(np.float32)
    z_tf365 = exp_seq_df["z_tf365_fact"].to_numpy().astype(np.float32)
    z_hier = exp_seq_df["z_hier_fact"].to_numpy().astype(np.float32)

    # Decompositions
    decomp_base = decompose_mse_by_transitions(y_true, np.clip(np.expm1(z_baseline_ens), 0, None), past_buyer)
    decomp_gru365 = decompose_mse_by_transitions(y_true, np.clip(np.expm1(z_gru365), 0, None), past_buyer)
    decomp_hier = decompose_mse_by_transitions(y_true, np.clip(np.expm1(z_hier), 0, None), past_buyer)

    # Ensembles:
    # Ensemble 1: CatBoost + GRU-365 (35% / 65%)
    z_ens_365 = 0.35 * z_cb_trans + 0.65 * z_gru365
    decomp_ens_365 = decompose_mse_by_transitions(y_true, np.clip(np.expm1(z_ens_365), 0, None), past_buyer)

    # Ensemble 2: CatBoost + Hierarchical GRU (35% / 65%)
    z_ens_hier = 0.35 * z_cb_trans + 0.65 * z_hier
    decomp_ens_hier = decompose_mse_by_transitions(y_true, np.clip(np.expm1(z_ens_hier), 0, None), past_buyer)

    # Ensemble 3: Tri-Ensemble (30% CatBoost + 50% Hierarchical GRU + 20% Transformer)
    z_ens_tri = 0.30 * z_cb_trans + 0.50 * z_hier + 0.20 * z_tf365
    decomp_ens_tri = decompose_mse_by_transitions(y_true, np.clip(np.expm1(z_ens_tri), 0, None), past_buyer)

    print("\n--- [1] OVERALL VALIDATION RMSLE COMPARISON ---")
    tbl_comp = pl.DataFrame({
        "Model / Ensemble": [
            "Baseline Clean Ensemble v4 (CB + GRU-90)",
            "Solo GRU-365 (1-Year Daily)",
            "Solo Hierarchical GRU (90d Daily + 275d Weekly)",
            "Ensemble: CatBoost Transitions + GRU-365",
            "Ensemble: CatBoost Transitions + Hierarchical GRU",
            "Tri-Ensemble: CatBoost + Hier-GRU + Transformer",
        ],
        "Validation_RMSLE": [
            decomp_base["total_rmsle"],
            decomp_gru365["total_rmsle"],
            decomp_hier["total_rmsle"],
            decomp_ens_365["total_rmsle"],
            decomp_ens_hier["total_rmsle"],
            decomp_ens_tri["total_rmsle"],
        ],
        "Validation_MSE": [
            decomp_base["total_mse"],
            decomp_gru365["total_mse"],
            decomp_hier["total_mse"],
            decomp_ens_365["total_mse"],
            decomp_ens_hier["total_mse"],
            decomp_ens_tri["total_mse"],
        ],
        "Delta_MSE_vs_Baseline": [
            0.0,
            decomp_gru365["total_mse"] - decomp_base["total_mse"],
            decomp_hier["total_mse"] - decomp_base["total_mse"],
            decomp_ens_365["total_mse"] - decomp_base["total_mse"],
            decomp_ens_hier["total_mse"] - decomp_base["total_mse"],
            decomp_ens_tri["total_mse"] - decomp_base["total_mse"],
        ],
    })
    print(tbl_comp)

    print("\n--- [2] DETAILED 4-STATE SSE COMPARISON (Tri-Ensemble vs Baseline) ---")
    t_base = decomp_base["decomposition_table"]
    t_tri = decomp_ens_tri["decomposition_table"]

    sse_comp = pl.DataFrame({
        "Transition State": t_base["State"].to_list(),
        "User Count": t_base["Count"].to_list(),
        "Baseline SSE": t_base["Group_SSE"].to_list(),
        "Tri-Ensemble SSE": t_tri["Group_SSE"].to_list(),
        "SSE Reduction (Delta)": (t_tri["Group_SSE"] - t_base["Group_SSE"]).to_list(),
        "SSE Reduction (%)": ((t_tri["Group_SSE"] - t_base["Group_SSE"]) / t_base["Group_SSE"] * 100.0).to_list(),
        "Baseline Group RMSLE": t_base["Group_RMSLE"].to_list(),
        "Tri-Ensemble Group RMSLE": t_tri["Group_RMSLE"].to_list(),
    })
    print(sse_comp)


if __name__ == "__main__":
    main()
