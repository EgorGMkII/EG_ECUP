"""Checkpoint Lineage and Leakage Verification for Specialized Hurdle Stack."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointMetadata:
    model_family: str          # "s1_gru", "s2_gru", "ett", "t5", "catboost"
    task: str                  # "base_multitask", "reactivation", "churn", "amount"
    checkpoint_path: str
    checkpoint_hash: str
    encoder_architecture: str
    head_architecture: str
    train_anchors: List[str]
    latest_input_date: str
    latest_target_date: str
    validation_anchor: str
    feature_order: List[str]
    scaler: Optional[str]
    loss: str
    optimizer: str
    best_metric_name: str
    best_metric_value: float
    seed: int


def compute_file_hash(file_path: Path) -> str:
    if not file_path.exists():
        return "none"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def is_checkpoint_fold_safe(metadata: CheckpointMetadata, outer_anchor: str) -> bool:
    """Verifies that the checkpoint did NOT train on supervised targets extending past outer_anchor."""
    outer_dt = datetime.strptime(outer_anchor, "%Y-%m-%d").date()
    target_end_dt = datetime.strptime(metadata.latest_target_date, "%Y-%m-%d").date()
    return target_end_dt <= outer_dt


def save_checkpoint_metadata(metadata: CheckpointMetadata, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)


def load_checkpoint_metadata(json_path: Path) -> CheckpointMetadata:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CheckpointMetadata(**data)
