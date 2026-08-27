"""Two-Tower Sequential Network Adapter for direct temporal CV."""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements
from ..models.two_tower import TwoTowerEventNet


class EventTupleDataset(Dataset):
    def __init__(self, memmap_tuple: tuple[np.ndarray, ...], targets: np.ndarray | None = None):
        self.counts = memmap_tuple[0]
        self.totals = memmap_tuple[1]
        self.recency = memmap_tuple[2]
        self.mask = memmap_tuple[3]
        self.static = memmap_tuple[4]
        self.targets = targets

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, idx: int):
        c = torch.from_numpy(self.counts[idx].astype(np.float32))
        t = torch.from_numpy(self.totals[idx].astype(np.float32))
        r = torch.from_numpy(self.recency[idx].astype(np.float32))
        m = torch.from_numpy(self.mask[idx])
        s = torch.from_numpy(self.static[idx:idx + 1]).squeeze(0)

        if self.targets is not None:
            return c, t, r, m, s, torch.tensor(self.targets[idx], dtype=torch.float32)
        return c, t, r, m, s


class DirectTwoTowerAdapter(DirectModelAdapter):
    model_id = "two_tower_direct"
    requirements = ModelRequirements(event_sequences=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "churn_weight",
            "latent_dim",
            "head_dropout",
            "seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {self.model_id} fields: {sorted(unknown)}")

        values = {
            "epochs": int(raw.get("epochs", 2)),
            "batch_size": int(raw.get("batch_size", 512)),
            "learning_rate": float(raw.get("learning_rate", 5e-4)),
            "weight_decay": float(raw.get("weight_decay", 1e-4)),
            "churn_weight": float(raw.get("churn_weight", 0.2)),
            "latent_dim": int(raw.get("latent_dim", 32)),
            "head_dropout": float(raw.get("head_dropout", 0.1)),
            "seed": int(raw.get("seed", 42)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_events is None or context.validation_events is None:
            raise ValueError("Two-Tower adapter requires event sequences store")

        torch.manual_seed(config.values["seed"])
        dev = context.device if context.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_memmap = context.train_events
        val_memmap = context.validation_events

        train_dataset = EventTupleDataset(train_memmap, targets=context.train_target_z)
        val_dataset = EventTupleDataset(val_memmap, targets=None)

        bs = config.values["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

        count_dim = train_memmap[0].shape[-1]
        total_dim = train_memmap[1].shape[-1]
        static_dim = train_memmap[4].shape[-1]

        model = TwoTowerEventNet(
            count_features=count_dim,
            total_features=total_dim,
            static_features=static_dim,
            latent_dim=config.values["latent_dim"],
            head_dropout=config.values["head_dropout"],
        ).to(dev)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.values["learning_rate"],
            weight_decay=config.values["weight_decay"],
        )

        epochs = config.values["epochs"]
        churn_w = config.values["churn_weight"]

        t0 = time.perf_counter()
        model.train()
        for ep in range(epochs):
            for counts, totals, recency, mask, static, target in train_loader:
                counts = counts.to(dev)
                totals = totals.to(dev)
                recency = recency.to(dev)
                mask = mask.to(dev)
                static = static.to(dev)
                target = target.to(dev)

                optimizer.zero_grad()
                out = model(counts, totals, recency, mask, static)

                loss_z = F.mse_loss(out["direct_z"], target)
                churn_label = (target <= 0.0).float()
                loss_churn = F.binary_cross_entropy_with_logits(out["churn_logit"], churn_label)

                loss = loss_z + churn_w * loss_churn
                loss.backward()
                optimizer.step()

        fit_dur = time.perf_counter() - t0

        # Predict
        model.eval()
        t1 = time.perf_counter()
        preds_list = []
        with torch.no_grad():
            for counts, totals, recency, mask, static in val_loader:
                counts = counts.to(dev)
                totals = totals.to(dev)
                recency = recency.to(dev)
                mask = mask.to(dev)
                static = static.to(dev)

                out = model(counts, totals, recency, mask, static)
                preds_list.append(out["direct_z"].cpu().numpy())

        pred_z = np.clip(np.concatenate(preds_list, axis=0), 0.0, 15.0)

        report = {
            "model_id": self.model_id,
            "fold_id": context.fold.fold_id,
            "epochs": epochs,
            "latent_dim": config.values["latent_dim"],
            "churn_weight": churn_w,
            "elapsed_seconds": fit_dur,
        }

        return FoldPrediction(self.model_id, np.asarray(context.users), pred_z, report)
