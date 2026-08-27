"""Full 250k training and submission generation for Next-Generation Quad Stack."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.direct_temporal_cv_v1.adapters.catboost_direct import DirectCatBoostAdapter
from src.direct_temporal_cv_v1.adapters.delta_regressor import DirectDeltaRegressorAdapter
from src.direct_temporal_cv_v1.adapters.frequency_specialist import DirectFrequencySpecialistAdapter
from src.direct_temporal_cv_v1.adapters.two_tower_adapter import DirectTwoTowerAdapter
from src.direct_temporal_cv_v1.base import FoldContext
from src.direct_temporal_cv_v1.contracts import TemporalFold
from src.direct_temporal_cv_v1.features.coles import extract_coles_embeddings
from src.direct_temporal_cv_v1.features.sparse_agg import SparseAggregateFeatureProvider
from src.ssl_temporal_stack_v1.stores import build_event_memmap_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.parquet")
    parser.add_argument("--sample-submit", default="sample_submit.csv")
    parser.add_argument("--output-dir", default="artifacts/direct_temporal_cv_v1/submissions/next_gen_quad_v1")
    parser.add_argument("--job-id", default="local")
    parser.add_argument("--pre-run-sha", default="unknown")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading raw logs and sample submit...")
    raw = pl.read_parquet(args.train_data)
    sample_sub = pl.read_csv(args.sample_submit)
    users = sample_sub["user_id"].to_list()
    user_array = np.asarray(users)

    # Full Train Anchor: 2025-12-15 -> Target: 2025-12-16..2026-01-14
    # Full Inference Anchor: 2026-01-14 -> Predict Test GMV
    train_anchor = date(2025, 12, 15)
    inf_anchor = date(2026, 1, 14)
    fold_obj = TemporalFold("FULL_250K", inf_anchor)

    # 1. Tabular features
    logging.info("Extracting 194 sparse tabular features for train and inference anchors...")
    provider = SparseAggregateFeatureProvider()
    train_tab = provider.extract_features(raw, users, train_anchor)
    val_tab = provider.extract_features(raw, users, inf_anchor)

    # Compute targets for train anchor
    train_target_df = (
        raw.filter(
            (pl.col("action_type") == 2)
            & (pl.col("timestamp") >= "2025-12-16")
            & (pl.col("timestamp") <= "2026-01-14")
        )
        .group_by("user_id")
        .agg(pl.col("item_price").sum().alias("gmv"))
    )
    user_df = pl.DataFrame({"user_id": users})
    train_target_z = (
        user_df.join(train_target_df, on="user_id", how="left")
        .with_columns(pl.col("gmv").fill_null(0.0).log1p().alias("target_z"))["target_z"]
        .to_numpy()
    )

    # 2. CoLES Embeddings
    logging.info("Training CoLES embeddings on 90d window...")
    train_coles, val_coles = extract_coles_embeddings(raw, users, train_anchor, inf_anchor)
    train_tab = train_tab.join(train_coles, on="user_id", how="left")
    val_tab = val_tab.join(val_coles, on="user_id", how="left")
    logging.info(f"Total tabular features: {len(train_tab.columns) - 1}")

    # 3. Event Sequences Store
    logging.info("Building event memmap store for Two-Tower network...")
    store_root = Path("artifacts/direct_temporal_cv_v1/temp_stores")
    event_store = build_event_memmap_store(
        raw, users, (train_anchor.isoformat(), inf_anchor.isoformat()), store_root / "events"
    )
    train_events = event_store.get_anchor_slice(train_anchor.isoformat())
    val_events = event_store.get_anchor_slice(inf_anchor.isoformat())

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context = FoldContext(
        fold=fold_obj,
        users=user_array,
        train_target_z=train_target_z,
        validation_target_z=np.zeros_like(train_target_z),
        train_tabular=train_tab,
        validation_tabular=val_tab,
        train_daily=None,
        validation_daily=None,
        train_events=train_events,
        validation_events=val_events,
        device=dev,
        output_dir=out_dir,
        root_seed=42,
    )

    # Model 1: CatBoost Direct
    logging.info("Training Full 250k CatBoost Direct...")
    cb_ad = DirectCatBoostAdapter()
    cb_cfg = cb_ad.validate_config({"iterations": 450, "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 5.0})
    res_cb = cb_ad.fit_predict_fold(context, cb_cfg)
    cb_z = res_cb.prediction_z

    # Model 2: Frequency Specialist
    logging.info("Training Full 250k Frequency Specialist...")
    freq_ad = DirectFrequencySpecialistAdapter()
    freq_cfg = freq_ad.validate_config({
        "min_orders_90d": 2,
        "frequent_iterations": 450,
        "frequent_depth": 8,
        "frequent_lr": 0.04,
        "frequent_l2": 5.0,
        "dormant_iterations": 450,
        "dormant_depth": 7,
        "dormant_lr": 0.035,
        "dormant_l2": 5.0,
    })
    res_freq = freq_ad.fit_predict_fold(context, freq_cfg)
    freq_z = res_freq.prediction_z

    # Model 3: Delta Regressor
    logging.info("Training Full 250k Delta Regressor...")
    delta_ad = DirectDeltaRegressorAdapter()
    delta_cfg = delta_ad.validate_config({
        "baseline_feature": "gmv_sum_30d",
        "iterations": 450,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
    })
    res_delta = delta_ad.fit_predict_fold(context, delta_cfg)
    delta_z = res_delta.prediction_z

    # Model 4: Two-Tower Network
    logging.info("Training Full 250k Two-Tower Event Network...")
    two_ad = DirectTwoTowerAdapter()
    two_cfg = two_ad.validate_config({"epochs": 2, "batch_size": 512, "learning_rate": 0.0005, "churn_weight": 0.2})
    res_two = two_ad.fit_predict_fold(context, two_cfg)
    two_z = res_two.prediction_z

    # Quad Next-Gen Synergy Blend:
    # 0.40 FreqSpecialist + 0.28 DeltaReg + 0.17 TwoTower + 0.15 CatBoost
    logging.info("Ensembling Quad Next-Gen Synergy Blend...")
    quad_z = 0.40 * freq_z + 0.28 * delta_z + 0.17 * two_z + 0.15 * cb_z
    quad_gmv = np.expm1(np.clip(quad_z, 0.0, 15.0))

    sub_quad = pl.DataFrame({"user_id": users, "target": quad_gmv})
    sub_quad_path = out_dir / "submission_direct_next_gen_quad_stack_v1.csv"
    sub_quad.write_csv(sub_quad_path)
    logging.info(f"Saved Quad Stack submission: {sub_quad_path}")
    logging.info(f"Quad GMV Mean: {quad_gmv.mean():.2f}, Median: {np.median(quad_gmv):.2f}, Max: {quad_gmv.max():.2f}")

    # Grand Meta Blend with Champion (1.6559):
    # 0.60 Champions Meta Blend + 0.40 Next-Gen Quad Stack
    champ_path = Path("submission_meta_blend_champions_v1.csv")
    if champ_path.exists():
        logging.info("Blending with Record Champion Submission (1.6559)...")
        champ_df = pl.read_csv(champ_path)
        champ_gmv = champ_df["target"].to_numpy()
        champ_z = np.log1p(np.clip(champ_gmv, 0.0, None))

        # Ensembling in Z space
        grand_z = 0.60 * champ_z + 0.40 * quad_z
        grand_gmv = np.expm1(np.clip(grand_z, 0.0, 15.0))

        sub_grand = pl.DataFrame({"user_id": users, "target": grand_gmv})
        sub_grand_path = out_dir / "submission_grand_next_gen_champion_blend_v1.csv"
        sub_grand.write_csv(sub_grand_path)
        logging.info(f"Saved Grand Champion Blend submission: {sub_grand_path}")
        logging.info(f"Grand GMV Mean: {grand_gmv.mean():.2f}, Median: {np.median(grand_gmv):.2f}, Max: {grand_gmv.max():.2f}")

    logging.info("FULL 250K TRAINING AND SUBMISSION GENERATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
