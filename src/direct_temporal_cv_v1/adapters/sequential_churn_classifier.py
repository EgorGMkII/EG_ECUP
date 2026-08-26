"""Causal Sequential GRU + Attention Churn Classifier for Active Cohort."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = torch.softmax(self.query(x), dim=1) # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1) # (B, D)
        return pooled


class CausalSequentialChurnNet(nn.Module):
    def __init__(self, in_features: int = 15, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttentionPooling(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features)
        h = self.input_proj(x)
        out, _ = self.gru(h)
        pooled = self.attention(out)
        logits = self.head(pooled).squeeze(-1)
        return logits


class SequentialChurnClassifierAdapter(DirectModelAdapter):
    model_id = "sequential_churn_classifier"
    requirements = ModelRequirements(daily_tensor=True, tabular_features=True)

    def validate_config(self, raw: dict[str, Any]) -> ModelConfig:
        allowed = {
            "epochs",
            "batch_size",
            "learning_rate",
            "hidden_dim",
            "num_layers",
            "dropout",
            "weight_decay",
            "history_days",
            "random_seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown sequential_churn_classifier fields: {sorted(unknown)}")
        values = {
            "epochs": int(raw.get("epochs", 3)),
            "batch_size": int(raw.get("batch_size", 512)),
            "learning_rate": float(raw.get("learning_rate", 0.001)),
            "hidden_dim": int(raw.get("hidden_dim", 64)),
            "num_layers": int(raw.get("num_layers", 2)),
            "dropout": float(raw.get("dropout", 0.1)),
            "weight_decay": float(raw.get("weight_decay", 0.0001)),
            "history_days": int(raw.get("history_days", 180)),
            "random_seed": int(raw.get("random_seed", 42)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_daily is None or context.validation_daily is None:
            raise ValueError("sequential_churn_classifier requires daily tensor stores")
        
        v = config.values
        device = context.device
        torch.manual_seed(v["random_seed"])
        np.random.seed(v["random_seed"])

        started = time.perf_counter()

        # Direct daily causal arrays (shape: N, 180, 15)
        x_train_daily = context.train_daily
        x_val_daily = context.validation_daily

        # Activity definition: GMV or orders in last 90d (channels 0 and 3)
        train_active = (x_train_daily[:, -90:, 0].sum(axis=1) > 0) | (x_train_daily[:, -90:, 3].sum(axis=1) > 0)
        val_active = (x_val_daily[:, -90:, 0].sum(axis=1) > 0) | (x_val_daily[:, -90:, 3].sum(axis=1) > 0)

        train_z = context.train_target_z
        train_will_buy = (train_z > 0).astype(np.float32)

        x_train_active = torch.from_numpy(x_train_daily[train_active]).float()
        y_train_active = torch.from_numpy(train_will_buy[train_active]).float()

        n_active_train = int(train_active.sum())
        buy_rate = float(y_train_active.mean())
        print(f"  [SEQ_CHURN] Active train N={n_active_train}, buy_rate={buy_rate:.4f}", flush=True)

        model = CausalSequentialChurnNet(
            in_features=x_train_daily.shape[-1],
            hidden_dim=v["hidden_dim"],
            num_layers=v["num_layers"],
            dropout=v["dropout"],
        ).to(device)

        dataset = TensorDataset(x_train_active, y_train_active)
        loader = DataLoader(dataset, batch_size=v["batch_size"], shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=v["learning_rate"], weight_decay=v["weight_decay"])
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(v["epochs"]):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
            print(f"  [SEQ_CHURN] Epoch {epoch+1}/{v['epochs']} Loss: {total_loss/len(dataset):.4f}", flush=True)

        # In-sample & Validation evaluation
        model.eval()
        with torch.no_grad():
            # In-sample Train proba
            train_preds = []
            for batch_x, _ in DataLoader(dataset, batch_size=v["batch_size"]*2, shuffle=False):
                logits = model(batch_x.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                train_preds.append(probs)
            train_probs = np.concatenate(train_preds)
            train_auc = float(roc_auc_score(y_train_active.numpy(), train_probs))
            print(f"  [SEQ_CHURN] Train AUC: {train_auc:.6f}", flush=True)

            # Validation proba on Active cohort
            x_val_active = torch.from_numpy(x_val_daily[val_active]).float()
            val_dataset = TensorDataset(x_val_active)
            val_preds = []
            for (batch_x,) in DataLoader(val_dataset, batch_size=v["batch_size"]*2, shuffle=False):
                logits = model(batch_x.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
            val_probs = np.concatenate(val_preds)

        val_auc = None
        if context.validation_target_z is not None and (context.validation_target_z > 0).any():
            val_will_buy = (context.validation_target_z[val_active] > 0).astype(np.int32)
            val_auc = float(roc_auc_score(val_will_buy, val_probs))
            print(f"  [SEQ_CHURN] >>> Validation Active Churn AUC: {val_auc:.6f} <<<", flush=True)

        # Fit conditional regressor on buyers to produce prediction_z
        from catboost import CatBoostRegressor
        train_tab = context.train_tabular
        val_tab = context.validation_tabular
        feature_order = tuple(c for c in train_tab.columns if c != "user_id")
        x_train_tab = train_tab.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_val_tab = val_tab.select(feature_order).to_numpy().astype(np.float32, copy=False)

        active_buyers_mask = train_active & (train_will_buy.astype(bool))
        cb_amount = CatBoostRegressor(iterations=350, depth=8, learning_rate=0.05, loss_function="RMSE", verbose=False, allow_writing_files=False)
        cb_amount.fit(x_train_tab[active_buyers_mask], train_z[active_buyers_mask], verbose=False)
        cond_z_val = np.maximum(cb_amount.predict(x_val_tab[val_active]), 0.0)

        prediction_z = np.zeros(len(context.users), dtype=np.float64)
        val_active_idx = np.where(val_active)[0]
        prediction_z[val_active_idx] = val_probs * cond_z_val

        # Inactive cohort baseline
        cb_inact = CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05, loss_function="RMSE", verbose=False, allow_writing_files=False)
        cb_inact.fit(x_train_tab[~train_active], train_z[~train_active], verbose=False)
        val_inact_idx = np.where(~val_active)[0]
        prediction_z[val_inact_idx] = np.maximum(cb_inact.predict(x_val_tab[~val_active]), 0.0)

        elapsed = time.perf_counter() - started
        return FoldPrediction(
            model_id=self.model_id,
            user_ids=np.asarray(context.users),
            prediction_z=prediction_z,
            training_report={
                "model_id": self.model_id,
                "fold_id": context.fold.fold_id,
                "elapsed_seconds": elapsed,
                "fresh_model_per_fold": True,
                "train_auc": train_auc,
                "val_auc": val_auc,
            },
        )
