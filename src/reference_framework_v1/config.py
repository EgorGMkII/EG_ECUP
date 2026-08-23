"""Strict YAML loading for a single immutable reference experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .profiles import TemporalProfile, get_profile


STAGES = frozenset({"screen", "full", "final"})
MODEL_IDS = frozenset({"catboost", "s1", "s2", "ett", "tcn", "residual_mlp"})


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str
    experiment_id: str
    profile: TemporalProfile
    stage: str
    root_seed: int
    enabled_models: tuple[str, ...]
    output_root: Path
    train_path: Path
    sample_submit_path: Path
    cohort_path: Path


def _digest(raw: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_keys(raw: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields at {location}: {sorted(unknown)}")


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a mapping")
    _require_keys(raw, {"experiment_id", "profile", "stage", "root_seed", "enabled_models", "output_root", "inputs", "models", "anchor_sampling", "loss_weights", "final"}, "root")
    stage = str(raw["stage"])
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    enabled = tuple(raw["enabled_models"])
    if not enabled or len(enabled) != len(set(enabled)) or not set(enabled) <= MODEL_IDS:
        raise ValueError("enabled_models must be unique registered model IDs")
    inputs = raw["inputs"]
    _require_keys(inputs, {"train", "sample_submit", "cohort"}, "inputs")
    cohort_key = {"screen": "screen", "full": "full", "final": "universe"}[stage]
    cohort = inputs["cohort"]
    if cohort_key not in cohort:
        raise ValueError(f"inputs.cohort.{cohort_key} is required for {stage}")
    config = ExperimentConfig(path, raw, _digest(raw), str(raw["experiment_id"]), get_profile(str(raw["profile"])), stage, int(raw["root_seed"]), enabled, Path(raw["output_root"]), Path(inputs["train"]), Path(inputs["sample_submit"]), Path(cohort[cohort_key]))
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    if not config.experiment_id or config.root_seed != 42:
        raise ValueError("experiment_id and root_seed are invalid")
    if config.output_root.parts[:3] != ("artifacts", "reference_v1", "experiments"):
        raise ValueError("output_root must be isolated below artifacts/reference_v1/experiments")
    models = config.raw["models"]
    if set(models) != set(config.enabled_models):
        raise ValueError("models keys must equal enabled_models")
    for model_id, values in models.items():
        if not isinstance(values, dict):
            raise ValueError(f"models.{model_id} must be a mapping")
    sampling = config.raw.get("anchor_sampling", {"mode": "uniform"})
    _require_keys(sampling, {"mode", "tickets"}, "anchor_sampling")
    if sampling["mode"] not in {"uniform", "weighted_round_robin"}:
        raise ValueError("Unsupported anchor_sampling mode")
    if sampling["mode"] == "weighted_round_robin":
        tickets = sampling.get("tickets", {})
        if not tickets or any(not isinstance(value, int) or value <= 0 for value in tickets.values()):
            raise ValueError("anchor tickets must be positive integers")
    loss = config.raw.get("loss_weights", {})
    _require_keys(loss, {"factorized", "direct_amount", "conditional_amount", "react", "churn"}, "loss_weights")
    if loss and any(float(value) < 0 for value in loss.values()):
        raise ValueError("loss weights must be non-negative")
    if config.stage == "final":
        final = config.raw.get("final")
        if not isinstance(final, dict) or not {"frozen_meta_path", "frozen_meta_sha256", "source_full_config_sha256"} <= set(final):
            raise ValueError("final config must pin a full meta package and source config hash")


def resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    return {**config.raw, "config_sha256": config.sha256, "resolved_profile": {
        "run_a_anchors": list(config.profile.run_a_anchors), "meta_anchor": config.profile.meta_anchor,
        "run_b_anchors": list(config.profile.run_b_anchors), "validation_anchor": config.profile.validation_anchor,
        "final_train_anchors": list(config.profile.final_train_anchors), "final_inference_anchor": config.profile.final_inference_anchor,
    }}
