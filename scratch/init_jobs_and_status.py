"""Script to generate job_manifest.csv and initialize run_status.json."""

import json
from pathlib import Path
import polars as pl

out_dir = Path("artifacts/specialized_hurdle")
logs_dir = out_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

models = ["catboost", "s1_gru", "s2_gru", "ett_180", "t5_patch"]
tasks = ["react", "churn", "amount"]
folds = ["fold_00", "fold_01", "fold_02", "fold_03", "fold_04", "fold_05", "fold_06", "january_holdout"]

jobs = []
job_idx = 0

for fold in folds:
    # 1. Base pretraining jobs for neural models
    for m in ["s1_gru", "s2_gru", "ett_180"]:
        job_idx += 1
        jobs.append({
            "job_id": f"job_{job_idx:03d}",
            "model": m,
            "fold": fold,
            "stage": "Stage_A_Base",
            "task": "multitask_gmv30",
            "requires_gpu": True,
            "estimated_time_min": 12.0,
            "actual_time_min": None,
            "peak_memory_gb": None,
            "status": "PENDING",
            "artifact_path": f"artifacts/specialized_hurdle/base_checkpoints/{fold}/{m}_base.pt"
        })

    # T5 screening on fold_00
    if fold == "fold_00":
        job_idx += 1
        jobs.append({
            "job_id": f"job_{job_idx:03d}",
            "model": "t5_patch",
            "fold": fold,
            "stage": "Stage_A_Base",
            "task": "screening_base",
            "requires_gpu": True,
            "estimated_time_min": 15.0,
            "actual_time_min": None,
            "peak_memory_gb": None,
            "status": "PENDING",
            "artifact_path": f"artifacts/specialized_hurdle/base_checkpoints/{fold}/t5_base.pt"
        })

    # 2. Specialist jobs (CatBoost, S1, S2, ETT)
    for m in ["catboost", "s1_gru", "s2_gru", "ett_180"]:
        for t in tasks:
            job_idx += 1
            req_gpu = m != "catboost"
            jobs.append({
                "job_id": f"job_{job_idx:03d}",
                "model": m,
                "fold": fold,
                "stage": "Stage_B_Specialist",
                "task": t,
                "requires_gpu": req_gpu,
                "estimated_time_min": 4.0 if m == "catboost" else 6.0,
                "actual_time_min": None,
                "peak_memory_gb": None,
                "status": "PENDING",
                "artifact_path": f"artifacts/specialized_hurdle/specialist_checkpoints/{fold}/{m}_{t}.pt" if req_gpu else f"artifacts/specialized_hurdle/specialist_checkpoints/{fold}/{m}_{t}.cbm"
            })

df_jobs = pl.DataFrame(jobs)
df_jobs.write_csv(out_dir / "job_manifest.csv")
print(f"[+] Saved {len(df_jobs)} jobs to {out_dir / 'job_manifest.csv'}")

# Initialize run_status.json
status = {
    "current_stage": "INITIALIZATION",
    "completed_jobs": 0,
    "total_jobs": len(df_jobs),
    "active_fold": "fold_00",
    "last_checkpoint": None,
    "integrity_gates_passed": True,
    "start_time": "2026-08-21T01:41:00",
    "updated_at": "2026-08-21T01:41:00",
    "errors": []
}

with open(logs_dir / "run_status.json", "w", encoding="utf-8") as f:
    json.dump(status, f, indent=2)
print(f"[+] Initialized {logs_dir / 'run_status.json'}")
