"""Direct TCN extension point for causal daily tensors."""

from __future__ import annotations

from typing import Any, Mapping

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class DirectTCNAdapter(DirectModelAdapter):
    model_id = "tcn_direct"
    requirements = ModelRequirements(daily_tensor=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {"epochs", "batch_size", "learning_rate", "scheduler", "warmup_fraction", "weight_decay", "dropout", "history_days"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown tcn_direct fields: {sorted(unknown)}")
        values = {"epochs": int(raw.get("epochs", 2)), "batch_size": int(raw.get("batch_size", 512)), "learning_rate": float(raw.get("learning_rate", 3e-4)), "scheduler": str(raw.get("scheduler", "cosine")), "warmup_fraction": float(raw.get("warmup_fraction", 0.1)), "weight_decay": float(raw.get("weight_decay", 1e-4)), "dropout": float(raw.get("dropout", 0.1)), "history_days": int(raw.get("history_days", 180))}
        if values["epochs"] not in {1, 2, 3}:
            raise ValueError("tcn_direct epochs must be 1, 2, or 3")
        if values["history_days"] != 180:
            raise ValueError("tcn_direct history_days is pinned to 180 in v1")
        if values["scheduler"] not in {"constant", "linear", "cosine"}:
            raise ValueError("Unsupported TCN scheduler")
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        raise NotImplementedError("Direct TCN adapter is prepared but not enabled until causal scalar-head parity tests pass")
