"""Full 250k training and submission generation for Next-Generation Quad Stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.direct_temporal_cv_v1.adapters.catboost_direct import DirectCatBoostAdapter
from src.direct_temporal_cv_v1.adapters.delta_regressor import DirectDeltaRegressorAdapter
from src.direct_temporal_cv_v1.adapters.frequency_specialist import DirectFrequencySpecialistAdapter
from src.direct_temporal_cv_v1.adapters.two_tower_adapter import DirectTwoTowerAdapter
from src.direct_temporal_cv_v1.base import FoldContext
from src.direct_temporal_cv_v1.coles import CoLESEncoder, EventSequenceDataset, FullEventSequenceDataset, info_nce_loss
from src.direct_temporal_cv_v1.contracts import TemporalFold
from src.direct_temporal_cv_v1.datasets import build_target_z
from src.direct_temporal_cv_v1.features import SparseAggregateFeatureProvider
from src.ssl_temporal_stack_v1.stores import build_event_memmap_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--sample-submit", type=Path, default=Path("sample_submit.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/direct_temporal_cv_v1/submissions/next_gen_quad_v1"))
    parser.add_argument("--job-id", default="local")
    parser.add_argument("--pre-run-sha", default="unknown")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading raw logs and sample submit...")
    raw = pl.read_parquet(args.train_data)
    sample_sub = pl.read_csv(args.sample_submit)
    users = sample_sub["user_id"].to_numpy().astype(np.int64)

    # Full Train Anchor: 2026-01-14 -> Target: 2026-01-15..2026-02-13
    # Full Inference Anchor: 2026-02-13 -> Predict Test GMV
    train_anchor = date(2026, 1, 14)
    inf_anchor = date(2026, 2, 13)
    fold_obj = TemporalFold("FULL_250K", inf_anchor)

    # 1. 194 Tabular features
    logging.info(f"Extracting 194 sparse tabular features for train={train_anchor} and inf={inf_anchor}...")
    provider = SparseAggregateFeatureProvider()
    train_snap = provider.build_snapshot(raw, users.tolist(), train_anchor)
    inf_snap = provider.build_snapshot(raw, users.tolist(), inf_anchor)

    target_start = date(2026, 1, 15)
    target_end = date(2026, 2, 13)
    train_target_z = build_target_z(raw, users.tolist(), target_start, target_end)

    # 2. Event Sequences Store on SSD
    logging.info("Building event memmap store for CoLES & Two-Tower network...")
    store_root = Path(tempfile.mkdtemp(prefix="direct_next_gen_store_"))
    anchors = (train_anchor.isoformat(), inf_anchor.isoformat())
    events = build_event_memmap_store(raw, users.tolist(), anchors, store_root / "events")
    train_memmap = events.get(train_anchor.isoformat())
    inf_memmap = events.get(inf_anchor.isoformat())

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. CoLES 32 Dense Features
    logging.info("Training CoLES embeddings on GPU...")
    dataset = EventSequenceDataset(train_memmap[0])
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    coles_model = CoLESEncoder(in_features=train_memmap[0].shape[-1], hidden_dim=64, out_dim=32).to(dev)
    opt = torch.optim.AdamW(coles_model.parameters(), lr=1e-3, weight_decay=1e-4)
    coles_model.train()
    for ep in range(2):
        for w1, w2 in loader:
            w1, w2 = w1.to(dev), w2.to(dev)
            opt.zero_grad()
            loss = info_nce_loss(coles_model(w1), coles_model(w2))
            loss.backward()
            opt.step()

    coles_model.eval()
    def extract_coles(memmap_tuple):
        ds = FullEventSequenceDataset(memmap_tuple[0])
        dl = DataLoader(ds, batch_size=512, shuffle=False)
        res = []
        with torch.no_grad():
            for b in dl:
                res.append(coles_model(b.to(dev)).cpu().numpy())
        mat = np.concatenate(res, axis=0)
        cols = {f"coles_{i}": mat[:, i].astype(np.float32) for i in range(32)}
        cols["user_id"] = users
        return pl.DataFrame(cols)

    train_coles = extract_coles(train_memmap)
    inf_coles = extract_coles(inf_memmap)

    train_tab = train_snap.join(train_coles, on="user_id")
    val_tab = inf_snap.join(inf_coles, on="user_id")
    logging.info(f"Total tabular features: {len(train_tab.columns) - 1}")

    context = FoldContext(
        fold=fold_obj,
        users=users,
        train_target_z=train_target_z,
        validation_target_z=np.zeros_like(train_target_z),
        train_tabular=train_tab,
        validation_tabular=val_tab,
        train_daily=None,
        validation_daily=None,
        train_events=train_memmap,
        validation_events=inf_memmap,
        device=dev,
        output_dir=out_dir,
        root_seed=42,
    )

    # Model 1: CatBoost Direct (226f)
    logging.info("[1/4] Training Full 250k CatBoost Direct...")
    cb_ad = DirectCatBoostAdapter()
    cb_cfg = cb_ad.validate_config({"iterations": 500, "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 5.0})
    res_cb = cb_ad.fit_predict_fold(context, cb_cfg)
    cb_z = res_cb.prediction_z

    # Model 2: Frequency Specialist (Frequent CB + Dormant LGB)
    logging.info("[2/4] Training Full 250k Frequency Specialist...")
    freq_ad = DirectFrequencySpecialistAdapter()
    freq_cfg = freq_ad.validate_config({
        "min_orders_90d": 2,
        "frequent_iterations": 500,
        "frequent_depth": 8,
        "frequent_lr": 0.04,
        "frequent_l2": 5.0,
        "dormant_iterations": 500,
        "dormant_depth": 7,
        "dormant_lr": 0.035,
        "dormant_l2": 5.0,
    })
    res_freq = freq_ad.fit_predict_fold(context, freq_cfg)
    freq_z = res_freq.prediction_z

    # Model 3: Delta Regressor
    logging.info("[3/4] Training Full 250k Delta Regressor...")
    delta_ad = DirectDeltaRegressorAdapter()
    delta_cfg = delta_ad.validate_config({
        "baseline_feature": "gmv_sum_30d",
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
    })
    res_delta = delta_ad.fit_predict_fold(context, delta_cfg)
    delta_z = res_delta.prediction_z

    # Model 4: Two-Tower Sequential Network
    logging.info("[4/4] Training Full 250k Two-Tower Sequential Network on GPU...")
    two_ad = DirectTwoTowerAdapter()
    two_cfg = two_ad.validate_config({"epochs": 2, "batch_size": 512, "learning_rate": 0.0005, "churn_weight": 0.2})
    res_two = two_ad.fit_predict_fold(context, two_cfg)
    two_z = res_two.prediction_z

    # Quad Next-Gen Synergy Blend:
    # 0.40 FreqSpecialist + 0.28 DeltaReg + 0.17 TwoTower + 0.15 CatBoost
    logging.info("Ensembling Quad Next-Gen Synergy Blend (0.40 Freq + 0.28 Delta + 0.17 TwoTower + 0.15 CB)...")
    quad_z = 0.40 * freq_z + 0.28 * delta_z + 0.17 * two_z + 0.15 * cb_z
    quad_gmv = np.expm1(np.clip(quad_z, 0.0, 15.0))

    sub_quad = pl.DataFrame({"user_id": users, "predict": quad_gmv})
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
        col = "predict" if "predict" in champ_df.columns else champ_df.columns[1]
        champ_gmv = champ_df[col].to_numpy()
        champ_z = np.log1p(np.clip(champ_gmv, 0.0, None))

        grand_z = 0.60 * champ_z + 0.40 * quad_z
        grand_gmv = np.expm1(np.clip(grand_z, 0.0, 15.0))

        sub_grand = pl.DataFrame({"user_id": users, "predict": grand_gmv})
        sub_grand_path = out_dir / "submission_grand_next_gen_champion_blend_v1.csv"
        sub_grand.write_csv(sub_grand_path)
        logging.info(f"Saved Grand Champion Blend submission: {sub_grand_path}")
        logging.info(f"Grand GMV Mean: {grand_gmv.mean():.2f}, Median: {np.median(grand_gmv):.2f}, Max: {grand_gmv.max():.2f}")

    logging.info("FULL 250K TRAINING AND SUBMISSION GENERATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
