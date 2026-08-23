"""Load and strictly validate the machine-readable SSL V1 config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .contract import EXPERIMENT, NeuralBudget, load_feature_order, validate_experiment


DEFAULT_CONFIG_PATH = Path("configs/ssl_temporal_stack_v1/experiment.yaml")


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str
    train_path: Path
    cohort_path: Path
    output_root: Path


def _canonical_sha(raw: dict[str, Any]) -> str:
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LoadedConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    config = LoadedConfig(
        path=path,
        raw=raw,
        sha256=_canonical_sha(raw),
        train_path=Path(raw["train"]["path"]),
        cohort_path=Path(raw["cohort"]["path"]),
        output_root=Path(raw["output_root"]),
    )
    validate_loaded_config(config)
    return config


def validate_loaded_config(config: LoadedConfig) -> None:
    validate_experiment(EXPERIMENT)
    raw = config.raw
    if raw.get("experiment_id") != EXPERIMENT.experiment_id:
        raise ValueError("YAML experiment_id differs from Python contract")
    if raw.get("root_seed") != EXPERIMENT.root_seed:
        raise ValueError("YAML root_seed differs from Python contract")
    if tuple(raw["run_a"]["anchors"]) != EXPERIMENT.run_a_anchors:
        raise ValueError("YAML RUN A anchors differ from Python contract")
    if raw["run_a"]["holdout_anchor"] != EXPERIMENT.meta_anchor:
        raise ValueError("YAML M differs from Python contract")
    if tuple(raw["run_b"]["anchors"]) != EXPERIMENT.run_b_anchors:
        raise ValueError("YAML RUN B anchors differ from Python contract")
    if raw["run_b"]["holdout_anchor"] != EXPERIMENT.validation_anchor:
        raise ValueError("YAML V differs from Python contract")
    if tuple(raw["models"]["order"]) != EXPERIMENT.model_ids:
        raise ValueError("YAML model order differs from Python contract")
    for model_id, budget in EXPERIMENT.budgets.items():
        values = raw["models"][model_id]
        observed = NeuralBudget(
            ssl_steps=int(values["ssl_steps"]),
            base_steps=int(values["base_steps"]),
            specialist_head_steps=int(values["specialist_head_steps"]),
            specialist_finetune_steps=int(values["specialist_finetune_steps"]),
            batch_size=int(values.get("batch_size", values.get("effective_batch_size"))),
        )
        if observed != budget:
            raise ValueError(f"YAML {model_id} budget differs from Python contract")
    feature_config = raw["catboost_features"]
    if int(feature_config["count"]) != 374:
        raise ValueError("YAML CatBoost feature count must be 374")
    if feature_config["ordered_features_sha256"] != EXPERIMENT.feature_order_sha256:
        raise ValueError("YAML CatBoost feature hash differs from Python contract")
    feature_path = Path(feature_config["path"])
    if feature_path != EXPERIMENT.feature_order_path:
        raise ValueError("YAML CatBoost feature path differs from Python contract")
    load_feature_order(feature_path)
    if raw["meta"] != {
        "optimizer": "SLSQP",
        "maxiter": 1000,
        "ftol": 1.0e-10,
        "random_starts": 8,
        "alpha": 1.1,
        "fit_anchor": EXPERIMENT.meta_anchor,
        "apply_anchor": EXPERIMENT.validation_anchor,
    }:
        raise ValueError("YAML meta contract differs from frozen SSL V1")
    if config.output_root != Path("artifacts/ssl_temporal_stack_v1/post_ny_public_proxy"):
        raise ValueError("Unexpected output root")


def resolved_contract(config: LoadedConfig) -> dict[str, Any]:
    """Return the auditable contract embedded into every run manifest."""

    return {
        "experiment_id": EXPERIMENT.experiment_id,
        "config_path": config.path.as_posix(),
        "config_sha256": config.sha256,
        "root_seed": EXPERIMENT.root_seed,
        "cohort_sha256": EXPERIMENT.cohort_sha256,
        "run_a_anchors": list(EXPERIMENT.run_a_anchors),
        "meta_anchor": EXPERIMENT.meta_anchor,
        "run_b_anchors": list(EXPERIMENT.run_b_anchors),
        "validation_anchor": EXPERIMENT.validation_anchor,
        "model_order": list(EXPERIMENT.model_ids),
        "budgets": {name: asdict(value) for name, value in EXPERIMENT.budgets.items()},
        "catboost_feature_order_sha256": EXPERIMENT.feature_order_sha256,
        "output_root": config.output_root.as_posix(),
    }
