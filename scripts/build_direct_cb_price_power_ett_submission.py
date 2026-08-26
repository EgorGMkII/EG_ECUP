"""Fresh full-history submission builder: CatBoost Price+Power Specialist (0.85) + ETT Direct (0.15)."""

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
from src.direct_temporal_cv_v1.adapters.direct_ett import DirectETTAdapter
from src.direct_temporal_cv_v1.base import FoldContext
from src.direct_temporal_cv_v1.contracts import TemporalFold
from src.direct_temporal_cv_v1.datasets import build_target_z
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
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/direct_cb_price_power_ett_submission"))
    ap.add_argument("--pre-run-sha", required=True)
    ap.add_argument("--job-id")
    args = ap.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)

    users = pl.read_csv(args.sample_submit)["user_id"].to_numpy().astype(np.int64)
    raw = pl.read_parquet(args.train)

    fold = TemporalFold("FINAL", date(2026, 2, 13))
    print(f"[*] Building tabular features for train={fold.train_anchor} and inference={fold.inference_anchor}...", flush=True)
    provider = SparseAggregateFeatureProvider()
    snaps = provider.build_pair(raw, users.tolist(), fold.train_anchor, fold.inference_anchor)

    print(f"[*] Extracting target z for train period [{fold.train_target_start} to {fold.train_target_end}]...", flush=True)
    target = build_target_z(raw, users.tolist(), fold.train_target_start, fold.train_target_end)

    print("[*] Building event sequences store on SSD...", flush=True)
    store_root = Path(tempfile.mkdtemp(prefix="direct_final_events_"))
    events = build_event_memmap_store(
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
        train_daily=None,
        validation_daily=None,
        train_events=events.get(fold.train_anchor.isoformat()),
        validation_events=events.get(fold.inference_anchor.isoformat()),
        device=dev,
        output_dir=out,
        root_seed=42,
    )

    print("[*] Training CatBoost Price Tier + Power 1.15 Specialist...", flush=True)
    cb_adapter = CatBoostCohortSpecialistAdapter()
    cb_cfg = cb_adapter.validate_config({
        "activity_window_days": 90,
        "churn_iterations": 600,
        "churn_depth": 6,
        "churn_learning_rate": 0.04,
        "churn_l2_leaf_reg": 3.0,
        "amount_iterations": 350,
        "amount_depth": 8,
        "amount_learning_rate": 0.05,
        "amount_l2_leaf_reg": 5.0,
        "inactive_iterations": 300,
        "inactive_depth": 8,
        "inactive_learning_rate": 0.05,
        "inactive_l2_leaf_reg": 5.0,
        "p_power": 1.15,
        "thread_count": 8,
        "random_seed": 42,
    })
    cb_res = cb_adapter.fit_predict_fold(ctx, cb_cfg)

    print("[*] Training ETT Direct Regressor on GPU...", flush=True)
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

    # Blend 85% CatBoost Price+Power Specialist + 15% ETT Direct
    print("[*] Blending predictions (0.85 CB Price+Power + 0.15 ETT Direct)...", flush=True)
    z_blend = 0.85 * cb_res.prediction_z + 0.15 * ett_res.prediction_z
    pred = np.maximum(np.expm1(z_blend), 0.0)

    # Save submission
    sub_path = out / "submission.csv"
    pl.DataFrame({"user_id": users, "predict": pred}).write_csv(sub_path)
    print(f"[+] Submission saved to {sub_path}", flush=True)

    report = {
        "experiment_id": "direct_cb_price_power_ett_submission_v1",
        "pre_run_sha": args.pre_run_sha,
        "job_id": args.job_id,
        "train_sha256": sha256_file(args.train),
        "sample_submit_sha256": sha256_file(args.sample_submit),
        "submission_sha256": sha256_file(sub_path),
        "inference_anchor": "2026-02-13",
        "weights": {
            "catboost_cohort_specialist_price_power": 0.85,
            "ett_direct": 0.15,
        },
        "models": {
            "catboost_cohort_specialist_price_power": cb_res.training_report,
            "ett_direct": ett_res.training_report,
        },
        "rows": len(users),
        "prediction_min": float(pred.min()),
        "prediction_median": float(np.median(pred)),
        "prediction_mean": float(pred.mean()),
        "prediction_max": float(pred.max()),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Clean up transient events memmap
    events.close()
    shutil.rmtree(store_root, ignore_errors=True)
    gc.collect()

    print("\n[+] FINAL SUBMISSION SUMMARY:\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
