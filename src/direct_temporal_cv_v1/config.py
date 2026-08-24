"""Strict immutable experiment configuration for direct four-fold CV."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .contracts import TemporalFold, get_protocol


MODEL_IDS = frozenset({"catboost_direct", "ett_direct", "tcn_direct"})
_ROOT_KEYS = {"experiment_id", "protocol", "root_seed", "enabled_models", "features", "models", "inputs", "output_root", "blend"}


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str
    experiment_id: str
    folds: tuple[TemporalFold, ...]
    root_seed: int
    enabled_models: tuple[str, ...]
    output_root: Path
    train_path: Path
    sample_submit_path: Path


def _canonical_sha(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields at {location}: {sorted(unknown)}")


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "root")
    _reject_unknown(raw, _ROOT_KEYS, "root")
    required = {"experiment_id", "protocol", "root_seed", "enabled_models", "features", "models", "inputs", "output_root"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing root config fields: {sorted(missing)}")
    enabled = tuple(str(item) for item in raw["enabled_models"])
    if not enabled or len(enabled) != len(set(enabled)) or not set(enabled) <= MODEL_IDS:
        raise ValueError("enabled_models must be unique registered direct model IDs")
    models = _require_mapping(raw["models"], "models")
    if set(models) != set(enabled):
        raise ValueError("models keys must exactly equal enabled_models")
    inputs = _require_mapping(raw["inputs"], "inputs")
    _reject_unknown(inputs, {"train", "sample_submit"}, "inputs")
    if {"train", "sample_submit"} - set(inputs):
        raise ValueError("inputs.train and inputs.sample_submit are required")
    output_root = Path(str(raw["output_root"]))
    if output_root.parts[:3] != ("artifacts", "direct_temporal_cv_v1", "experiments"):
        raise ValueError("output_root must be below artifacts/direct_temporal_cv_v1/experiments")
    if int(raw["root_seed"]) != 42:
        raise ValueError("root_seed is pinned to 42 for the baseline audit")
    return ExperimentConfig(
        path=path,
        raw=raw,
        sha256=_canonical_sha(raw),
        experiment_id=str(raw["experiment_id"]),
        folds=get_protocol(str(raw["protocol"])),
        root_seed=42,
        enabled_models=enabled,
        output_root=output_root,
        train_path=Path(str(inputs["train"])),
        sample_submit_path=Path(str(inputs["sample_submit"])),
    )


def resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    return {**config.raw, "config_sha256": config.sha256, "resolved_folds": [fold.as_dict() for fold in config.folds]}
