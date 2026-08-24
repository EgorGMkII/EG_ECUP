"""Direct ETT extension point for the four-fold protocol.

The implementation must wrap the existing EventTimeTransformer without
introducing SSL, transition specialists, pooled anchors, or meta weights.
Keeping the adapter explicit prevents an incomplete neural path from being
launched accidentally.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements
from src.ssl_temporal_stack_v1.models import EventTimeTransformer


class DirectETTAdapter(DirectModelAdapter):
    model_id = "ett_direct"
    requirements = ModelRequirements(event_sequences=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {"epochs", "batch_size", "gradient_accumulation", "learning_rate", "scheduler", "warmup_fraction", "weight_decay", "dropout", "history_days"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown ett_direct fields: {sorted(unknown)}")
        values = {"epochs": int(raw.get("epochs", 2)), "batch_size": int(raw.get("batch_size", 512)), "gradient_accumulation": int(raw.get("gradient_accumulation", 1)), "learning_rate": float(raw.get("learning_rate", 3e-4)), "scheduler": str(raw.get("scheduler", "cosine")), "warmup_fraction": float(raw.get("warmup_fraction", 0.1)), "weight_decay": float(raw.get("weight_decay", 1e-4)), "dropout": float(raw.get("dropout", 0.1)), "history_days": int(raw.get("history_days", 180))}
        if values["epochs"] not in {1, 2, 3}:
            raise ValueError("ett_direct epochs must be 1, 2, or 3")
        if values["history_days"] != 180:
            raise ValueError("ett_direct history_days is pinned to 180 in v1")
        if values["scheduler"] not in {"constant", "linear", "cosine"}:
            raise ValueError("Unsupported ETT scheduler")
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_events is None or context.validation_events is None:
            raise ValueError("ett_direct requires event sequences")
        device = context.device
        model = EventTimeTransformer(transformer_dropout=config.values["dropout"], head_dropout=config.values["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.values["learning_rate"], weight_decay=config.values["weight_decay"])
        train_arrays = context.train_events
        valid_arrays = context.validation_events
        model.train()
        batch_size = config.values["batch_size"]
        for epoch in range(config.values["epochs"]):
            order = np.arange(len(context.users))
            rng = np.random.default_rng(context.root_seed + epoch)
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                inputs = tuple(torch.as_tensor(array[idx], device=device) for array in train_arrays)
                inputs = (inputs[0].float(), inputs[1].float(), inputs[2].long(), inputs[3].bool(), inputs[4].bool())
                target = torch.as_tensor(context.train_target_z[idx], device=device).float()
                prediction = model(*inputs)["direct_z"]
                loss = nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(context.users), batch_size):
                idx = slice(start, start + batch_size)
                inputs = tuple(torch.as_tensor(array[idx], device=device) for array in valid_arrays)
                inputs = (inputs[0].float(), inputs[1].float(), inputs[2].long(), inputs[3].bool(), inputs[4].bool())
                predictions.append(model(*inputs)["direct_z"].detach().cpu().numpy())
        prediction = np.concatenate(predictions).astype(np.float64, copy=False)
        return FoldPrediction(self.model_id, np.asarray(context.users), prediction, {"model_id": self.model_id, "epochs": config.values["epochs"], "fresh_model_per_fold": True, "direct_target": "log1p_gmv_30d"})
