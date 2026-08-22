"""Logging and Experiment Registry Infrastructure for MultiTask GRU Sweep."""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import polars as pl
import numpy as np

ARTIFACTS_ROOT = Path("artifacts/gru_sweep")
MODELS_ROOT = Path("models/gru_sweep")
REGISTRY_PATH = ARTIFACTS_ROOT / "experiment_registry.csv"

REGISTRY_COLUMNS = [
    "run_id",
    "status",
    "timestamp",
    "sequence_length",
    "anchor_set",
    "n_anchors",
    "n_users",
    "n_samples",
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "lambda_cls",
    "lambda_reg",
    "alpha",
    "seed",
    "best_epoch",
    "train_time_sec",
    "peak_gpu_mb",
    "rmsle_direct",
    "rmsle_factorized",
    "rmsle_blend_cb",
    "reactivation_auc",
    "reactivation_brier",
    "churn_auc",
    "churn_brier",
    "mse_00_sleep",
    "mse_01_react",
    "mse_10_churn",
    "mse_11_retention",
    "cb_error_correlation",
    "checkpoint_path",
    "predictions_path",
    "notes",
]


class GRUExperimentLogger:
    """Manages run directories, registries, metrics, and manifest outputs."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = ARTIFACTS_ROOT / run_id
        self.model_dir = MODELS_ROOT / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / "run.log"

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        print(formatted)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    def save_config(self, config: Dict[str, Any]):
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save_anchor_manifest(self, anchors: List[str], fold_type: str = "train"):
        df = pl.DataFrame({"anchor_index": list(range(len(anchors))), "anchor_date": anchors, "type": [fold_type] * len(anchors)})
        df.write_csv(self.run_dir / "anchor_manifest.csv")

    def save_training_history(self, history: Dict[str, List[float]]):
        df = pl.DataFrame(history)
        df.write_csv(self.run_dir / "training_history.csv")

    def save_metrics(self, metrics: Dict[str, Any]):
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def save_transition_metrics(self, df_transitions: pl.DataFrame):
        df_transitions.write_csv(self.run_dir / "metrics_by_transition.csv")

    def save_prediction_distribution(self, dist: Dict[str, Any]):
        with open(self.run_dir / "prediction_distribution.json", "w", encoding="utf-8") as f:
            json.dump(dist, f, indent=2, ensure_ascii=False)

    def save_predictions_parquet(self, df_preds: pl.DataFrame) -> Path:
        p = self.run_dir / "predictions_validation.parquet"
        df_preds.write_parquet(p)
        return p

    def get_checkpoint_path(self) -> Path:
        return self.model_dir / "best.pt"


def initialize_registry():
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
            writer.writeheader()


def append_registry_record(record: Dict[str, Any]):
    initialize_registry()
    # Read existing
    existing = []
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if r.get("run_id") != record.get("run_id")]

    # Clean record
    clean_rec = {}
    for col in REGISTRY_COLUMNS:
        val = record.get(col, "")
        if isinstance(val, float):
            clean_rec[col] = f"{val:.5f}" if abs(val) < 1000 else f"{val:.2f}"
        else:
            clean_rec[col] = str(val) if val is not None else ""

    existing.append(clean_rec)
    with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
