"""Fold-Safe Lineage Validator and Metadata Management."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelArtifactMetadata:
    artifact_path: str
    model: str
    task: str
    fold: str
    train_anchors: List[str]
    validation_anchor: str
    maximum_input_date: str
    maximum_target_date: str
    feature_hash: str
    config_hash: str
    checkpoint_hash: str
    seed: int
    created_at: str


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def validate_fold_safety(
    train_anchors: List[str],
    validation_anchor: str,
    target_horizon_days: int = 30,
) -> Tuple[bool, Optional[str]]:
    """Strictly verifies that no training target window overlaps with the validation window."""
    v_dt = datetime.strptime(validation_anchor, "%Y-%m-%d").date()
    val_target_start = v_dt + timedelta(days=1)

    for a in train_anchors:
        a_dt = datetime.strptime(a, "%Y-%m-%d").date()
        train_target_end = a_dt + timedelta(days=target_horizon_days)

        # Violation if training target ends AFTER outer validation anchor V
        if train_target_end > v_dt:
            err = f"Leakage violation: train anchor {a} target ends {train_target_end} > validation anchor {v_dt}"
            return False, err

        # Violation if training target overlaps with validation target
        if train_target_end >= val_target_start:
            err = f"Overlap violation: train anchor {a} target ends {train_target_end} >= val target start {val_target_start}"
            return False, err

    return True, None


def save_artifact_metadata(meta: ModelArtifactMetadata, out_json_path: Path):
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)


def load_artifact_metadata(json_path: Path) -> ModelArtifactMetadata:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelArtifactMetadata(**data)
