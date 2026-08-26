"""Full 250k Submission Builder: Unified Hurdle (0.70 CatBoost + 0.30 Causal GRU Churn) + 0.30 ETT Direct."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import date
from pathlib import Path
import shutil
import tempfile

import numpy as np
import polars as pl
import torch

from src.direct_temporal_cv_v1.adapters.catboost_cohort_specialist import CatBoostCohortSpecialistAdapter
from src.direct_temporal_cv_v1.adapters.hybrid_cohort_specialist import HybridCohortSpecialistAdapter
from src.direct_temporal_cv_v1.adapters.direct_ett import DirectETTAdapter
from src.direct_temporal_cv_v1.base import FoldContext
from src.direct_temporal_cv_v1.contracts import TemporalFold
from src.direct_temporal_cv_v1.datasets import build_daily_tensor_store, build_target_z
from src.direct_temporal_cv_v1.features import SparseAggregateFeatureProvider
from src.ssl_temporal_stack_v1.stores import build_event_memmap_store


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    ap.add_argument("--sample-submit", type=Path, default=Path("sample_submit.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/direct_unified_hurdle_ett_submission_v1"))
    ap.add_argument("--pre-run-sha", required=True)
    ap.add_argument("--job-id")
    args = ap.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)

    users = pl.read_csv(args.sample_submit)["user_id"].to_numpy().astype(np.int64)
    raw = pl.read_parquet(args.train)

    fold = TemporalFold("FINAL", date(2026, 2, 13))
    print(f"[*] Building 106 tabular features for train={fold.train_anchor} and inference={fold.inference_anchor}...", flush=True)
    provider = SparseAggregateFeatureProvider()
    snaps = provider.build_pair(raw, users.tolist(), fold.train_anchor, fold.inference_anchor)

    print(f"[*] Extracting target z for train period [{fold.train_target_start} to {fold.train_target_end}]...", flush=True)
    target = build_target_z(raw, users.tolist(), fold.train_target_start, fold.train_target_end)

    store_root = Path(tempfile.mkdtemp(prefix="direct_final_stores_"))
    print("[*] Building daily tensor store on SSD...", flush=True)
    daily_store = build_daily_tensor_store(
        raw,
        users.tolist(),
        (fold.train_anchor.isoformat(), fold.inference_anchor.isoformat()),
        store_root / "daily",
    )

    print("[*] Building event sequences store on SSD...", flush=True)
    event_store = build_event_memmap_store(
        raw,
        users.tolist(),
        (fold.train_anchor.isoformat(), fold.inference_anchor.isoformat()),
        store_root / "events",
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = FoldContext(
        fold=fold,
        users=users,
        train_target_z=target,
        validation_target_z=np.zeros_like(target),
        train_tabular=snaps.train,
        validation_tabular=snaps.validation,
        train_daily=daily_store.get(fold.train_anchor.isoformat()),
        validation_daily=daily_store.get(fold.inference_anchor.isoformat()),
        train_events=event_store.get(fold.train_anchor.isoformat()),
        validation_events=event_store.get(fold.inference_anchor.isoformat()),
        device=dev,
        output_dir=out,
        root_seed=42,
    )

    # 1. Train CatBoost Cohort Specialist
    print("\n[1/3] Training CatBoost Cohort Specialist (Price Tier Natural)...", flush=True)
    cb_adapter = CatBoostCohortSpecialistAdapter()
    cb_cfg = cb_adapter.validate_config({
        "activity_window_days": 90,
        "churn_iterations": 600,
        "churn_depth": 6,
        "churn_learning_rate": 0.04,
        "churn_l2_leaf_reg": 3.0,
        "amount_iterations": 400,
        "amount_depth": 8,
        "amount_learning_rate": 0.05,
        "amount_l2_leaf_reg": 5.0,
        "inactive_iterations": 300,
        "inactive_depth": 8,
        "inactive_learning_rate": 0.05,
        "inactive_l2_leaf_reg": 5.0,
        "p_power": 1.0,
        "thread_count": 8,
        "random_seed": 42,
    })
    cb_res = cb_adapter.fit_predict_fold(ctx, cb_cfg)

    # 2. Train Hybrid Cohort Specialist (Causal Sequential GRU Churn + CatBoost Amount)
    print("\n[2/3] Training Hybrid Cohort Specialist (Causal GRU Churn + CatBoost Amount)...", flush=True)
    gru_adapter = HybridCohortSpecialistAdapter()
    gru_cfg = gru_adapter.validate_config({
        "activity_window_days": 90,
        "gru_epochs": 3,
        "gru_batch_size": 512,
        "gru_learning_rate": 0.001,
        "gru_hidden_dim": 64,
        "gru_num_layers": 2,
        "gru_dropout": 0.1,
        "gru_weight_decay": 0.0001,
        "amount_iterations": 400,
        "amount_depth": 8,
        "amount_learning_rate": 0.05,
        "amount_l2_leaf_reg": 5.0,
        "inactive_iterations": 300,
        "inactive_depth": 8,
        "inactive_learning_rate": 0.05,
        "inactive_l2_leaf_reg": 5.0,
        "thread_count": 8,
        "random_seed": 42,
    })
    gru_res = gru_adapter.fit_predict_fold(ctx, gru_cfg)

    # 3. Train ETT Direct Transformer
    print("\n[3/3] Training ETT Direct Transformer on GPU...", flush=True)
    ett_adapter = DirectETTAdapter()
    ett_cfg = ett_adapter.validate_config({
        "epochs": 2,
        "batch_size": 512,
        "learning_rate": 0.0003,
        "scheduler": "cosine",
        "warmup_fraction": 0.1,
        "weight_decay": 0.0001,
        "dropout": 0.1,
        "history_days": 180,
        "gradient_accumulation": 1,
    })
    ett_res = ett_adapter.fit_predict_fold(ctx, ett_cfg)

    # 4. Assemble Unified Hurdle & Final Synergy Blend
    print("\n[*] Assembling Unified Hurdle and Final Synergy Blend...", flush=True)
    z_cb = cb_res.prediction_z
    z_gru = gru_res.prediction_z
    z_ett = ett_res.prediction_z

    # Stage 1: Unified Hurdle (0.70 CB Hurdle + 0.30 GRU Hurdle)
    z_hurdle = 0.70 * z_cb + 0.30 * z_gru

    # Stage 2: Final Blend with ETT (0.70 Unified Hurdle + 0.30 ETT)
    z_final = 0.70 * z_hurdle + 0.30 * z_ett
    pred = np.maximum(np.expm1(z_final), 0.0)

    # Save submission
    sub_path = out / "submission.csv"
    pl.DataFrame({"user_id": users, "predict": pred}).write_csv(sub_path)
    print(f"[+] Submission successfully saved to {sub_path}", flush=True)

    report = {
        "experiment_id": "direct_unified_hurdle_ett_submission_v1",
        "pre_run_sha": args.pre_run_sha,
        "job_id": args.job_id,
        "train_sha256": sha256_file(args.train),
        "sample_submit_sha256": sha256_file(args.sample_submit),
        "submission_sha256": sha256_file(sub_path),
        "inference_anchor": "2026-02-13",
        "ensemble_formula": "0.70 * (0.70 * CatBoost_Hurdle + 0.30 * GRU_Hurdle) + 0.30 * ETT_Direct",
        "effective_weights": {
            "catboost_cohort_specialist": 0.49,
            "hybrid_gru_cohort_specialist": 0.21,
            "ett_direct": 0.30,
        },
        "models": {
            "catboost_cohort_specialist": cb_res.training_report,
            "hybrid_gru_cohort_specialist": gru_res.training_report,
            "ett_direct": ett_res.training_report,
        },
        "rows": len(users),
        "prediction_min": float(pred.min()),
        "prediction_median": float(np.median(pred)),
        "prediction_mean": float(pred.mean()),
        "prediction_max": float(pred.max()),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Clean up transient memmap stores
    event_store.close()
    shutil.rmtree(store_root, ignore_errors=True)
    gc.collect()

    print("\n[+] FINAL SUBMISSION SUMMARY:\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
