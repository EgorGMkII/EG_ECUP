"""Base Model Registry for Specialized Hurdle Stack."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from src.specialized_hurdle.checkpoint_lineage import CheckpointMetadata


@dataclass
class BaseModelSpec:
    model_name: str
    family: str
    checkpoint_path: Path
    config_path: Optional[Path]
    validation_predictions_path: Optional[Path]
    d_model: int
    architecture: str
    expected_january_rmsle: float


BASE_MODEL_SPECS: Dict[str, BaseModelSpec] = {
    "catboost_b1": BaseModelSpec(
        model_name="catboost_b1",
        family="catboost",
        checkpoint_path=Path("artifacts/catboost_cadence_audit/cb_model_14d.cbm"),
        config_path=None,
        validation_predictions_path=Path("artifacts/val_predictions_cv3.parquet"),
        d_model=0,
        architecture="CatBoostRegressor+Classifier (41 features)",
        expected_january_rmsle=1.71983,
    ),
    "s1_masked_gru": BaseModelSpec(
        model_name="s1_masked_gru",
        family="s1_gru",
        checkpoint_path=Path("artifacts/gru_hurdle_research/H1/best_model.pt"),
        config_path=Path("artifacts/gru_hurdle_research/canonical_gru180_config.json"),
        validation_predictions_path=Path("artifacts/s1_s2_router/router_val_predictions.parquet"),
        d_model=128,
        architecture="2-Layer Masked Behavior GRU (180 days)",
        expected_january_rmsle=1.68496,
    ),
    "s2_dense_gru": BaseModelSpec(
        model_name="s2_dense_gru",
        family="s2_gru",
        checkpoint_path=Path("artifacts/gru_hurdle_research/H2/best_model.pt"),
        config_path=Path("artifacts/gru_hurdle_research/canonical_gru180_config.json"),
        validation_predictions_path=Path("artifacts/s1_s2_router/router_val_predictions.parquet"),
        d_model=128,
        architecture="2-Layer Dense-Supervised GRU (180 days)",
        expected_january_rmsle=1.68756,
    ),
    "ett_opt_lr0": BaseModelSpec(
        model_name="ett_opt_lr0",
        family="ett",
        checkpoint_path=Path("artifacts/ett_optimization/OPT_LR0/best_model.pt"),
        config_path=Path("artifacts/ett_optimization/canonical_ett1_config.json"),
        validation_predictions_path=Path("artifacts/ett_optimization/OPT_LR0/validation_predictions.parquet"),
        d_model=128,
        architecture="2-Layer Event-Time Transformer (128 event tokens, tau=30d)",
        expected_january_rmsle=1.67722,
    ),
    "ett_opt_max256": BaseModelSpec(
        model_name="ett_opt_max256",
        family="ett",
        checkpoint_path=Path("artifacts/ett_optimization/OPT_MAX256/best_model.pt"),
        config_path=Path("artifacts/ett_optimization/canonical_ett1_config.json"),
        validation_predictions_path=Path("artifacts/ett_optimization/OPT_MAX256/validation_predictions.parquet"),
        d_model=128,
        architecture="2-Layer Event-Time Transformer (256 event tokens, tau=30d)",
        expected_january_rmsle=1.67784,
    ),
}
