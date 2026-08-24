"""Direct TCN extension point for causal daily tensors."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements
from src.reference_framework_v1.candidates.tcn import TCNRecipe, TCNTransitionBase


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
        if context.train_daily is None or context.validation_daily is None:
            raise ValueError("tcn_direct requires daily tensors")
        device = context.device
        recipe = TCNRecipe(dropout=config.values["dropout"])
        backbone = TCNTransitionBase(recipe).to(device)
        head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(config.values["dropout"]), nn.Linear(64, 1)).to(device)
        optimizer = torch.optim.AdamW([*backbone.parameters(), *head.parameters()], lr=config.values["learning_rate"], weight_decay=config.values["weight_decay"])
        batch_size = config.values["batch_size"]
        for epoch in range(config.values["epochs"]):
            order = np.arange(len(context.users))
            np.random.default_rng(context.root_seed + epoch).shuffle(order)
            backbone.train(); head.train()
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                daily = torch.as_tensor(context.train_daily[idx], device=device).float()
                target = torch.as_tensor(context.train_target_z[idx], device=device).float()
                prediction = head(backbone.encode(daily)).squeeze(-1)
                loss = nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        backbone.eval(); head.eval(); predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(context.users), batch_size):
                daily = torch.as_tensor(context.validation_daily[start:start + batch_size], device=device).float()
                predictions.append(head(backbone.encode(daily)).squeeze(-1).cpu().numpy())
        prediction = np.concatenate(predictions).astype(np.float64, copy=False)
        return FoldPrediction(self.model_id, np.asarray(context.users), prediction, {"model_id": self.model_id, "epochs": config.values["epochs"], "fresh_model_per_fold": True, "direct_target": "log1p_gmv_30d", "history_days": 180})
