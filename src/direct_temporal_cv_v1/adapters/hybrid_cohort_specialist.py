"""Hybrid Cohort Specialist: Causal Sequential GRU Churn + CatBoost Amount Regressors."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostRegressor

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.query(x), dim=1)
        pooled = torch.sum(x * weights, dim=1)
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
        h = self.input_proj(x)
        out, _ = self.gru(h)
        pooled = self.attention(out)
        logits = self.head(pooled).squeeze(-1)
        return logits


class HybridCohortSpecialistAdapter(DirectModelAdapter):
    model_id = "hybrid_cohort_specialist"
    requirements = ModelRequirements(daily_tensor=True, tabular_features=True)

    _VALID_WINDOWS = (30, 60, 90)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "activity_window_days",
            "gru_epochs",
            "gru_batch_size",
            "gru_learning_rate",
            "gru_hidden_dim",
            "gru_num_layers",
            "gru_dropout",
            "gru_weight_decay",
            "amount_iterations",
            "amount_depth",
            "amount_learning_rate",
            "amount_l2_leaf_reg",
            "inactive_iterations",
            "inactive_depth",
            "inactive_learning_rate",
            "inactive_l2_leaf_reg",
            "thread_count",
            "random_seed",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {self.model_id} fields: {sorted(unknown)}")
        activity_window = int(raw.get("activity_window_days", 90))
        if activity_window not in self._VALID_WINDOWS:
            raise ValueError(f"activity_window_days must be one of {self._VALID_WINDOWS}, got {activity_window}")
        values = {
            "activity_window_days": activity_window,
            "gru_epochs": int(raw.get("gru_epochs", 3)),
            "gru_batch_size": int(raw.get("gru_batch_size", 512)),
            "gru_learning_rate": float(raw.get("gru_learning_rate", 0.001)),
            "gru_hidden_dim": int(raw.get("gru_hidden_dim", 64)),
            "gru_num_layers": int(raw.get("gru_num_layers", 2)),
            "gru_dropout": float(raw.get("gru_dropout", 0.1)),
            "gru_weight_decay": float(raw.get("gru_weight_decay", 1e-4)),
            "amount_iterations": int(raw.get("amount_iterations", 400)),
            "amount_depth": int(raw.get("amount_depth", 8)),
            "amount_learning_rate": float(raw.get("amount_learning_rate", 0.05)),
            "amount_l2_leaf_reg": float(raw.get("amount_l2_leaf_reg", 5.0)),
            "inactive_iterations": int(raw.get("inactive_iterations", 300)),
            "inactive_depth": int(raw.get("inactive_depth", 8)),
            "inactive_learning_rate": float(raw.get("inactive_learning_rate", 0.05)),
            "inactive_l2_leaf_reg": float(raw.get("inactive_l2_leaf_reg", 5.0)),
            "thread_count": int(raw.get("thread_count", 8)),
            "random_seed": int(raw.get("random_seed", 42)),
        }
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_tabular is None or context.validation_tabular is None:
            raise ValueError("hybrid_cohort_specialist requires tabular features")
        if context.train_daily is None or context.validation_daily is None:
            raise ValueError("hybrid_cohort_specialist requires daily tensor stores")

        v = config.values
        device = context.device
        seed = v["random_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        started = time.perf_counter()

        # Tabular arrays
        train_tab = context.train_tabular
        val_tab = context.validation_tabular
        feature_order = tuple(c for c in train_tab.columns if c != "user_id")
        x_train_tab = train_tab.select(feature_order).to_numpy().astype(np.float32, copy=False)
        x_val_tab = val_tab.select(feature_order).to_numpy().astype(np.float32, copy=False)

        # Daily tensor arrays
        x_train_daily = context.train_daily
        x_val_daily = context.validation_daily

        # Activity filter: gmv > 0 or orders > 0 in last 90d
        act_col = f"gmv_sum_{v['activity_window_days']}d"
        train_active = (train_tab[act_col].to_numpy() > 0.0) | (x_train_daily[:, -90:, 0].sum(axis=1) > 0)
        val_active = (val_tab[act_col].to_numpy() > 0.0) | (x_val_daily[:, -90:, 0].sum(axis=1) > 0)

        train_z = context.train_target_z
        train_will_buy = (train_z > 0).astype(np.float32)

        n_active_train = int(train_active.sum())
        buy_rate = float(train_will_buy[train_active].mean())
        print(f"  [HYBRID] Active train N={n_active_train} ({n_active_train/len(train_z):.1%}), buy_rate={buy_rate:.4f}", flush=True)

        # ── Stage 1: Causal GRU + Attention Churn Classifier ────────────
        x_train_act_tensor = torch.from_numpy(x_train_daily[train_active]).float()
        y_train_act_tensor = torch.from_numpy(train_will_buy[train_active]).float()

        gru_model = CausalSequentialChurnNet(
            in_features=x_train_daily.shape[-1],
            hidden_dim=v["gru_hidden_dim"],
            num_layers=v["gru_num_layers"],
            dropout=v["gru_dropout"],
        ).to(device)

        dataset = TensorDataset(x_train_act_tensor, y_train_act_tensor)
        loader = DataLoader(dataset, batch_size=v["gru_batch_size"], shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(gru_model.parameters(), lr=v["gru_learning_rate"], weight_decay=v["gru_weight_decay"])
        criterion = nn.BCEWithLogitsLoss()

        gru_model.train()
        for epoch in range(v["gru_epochs"]):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = gru_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
            print(f"  [HYBRID] GRU Epoch {epoch+1}/{v['gru_epochs']} Loss: {total_loss/len(dataset):.4f}", flush=True)

        gru_model.eval()
        with torch.no_grad():
            x_val_act_tensor = torch.from_numpy(x_val_daily[val_active]).float()
            val_preds = []
            for (batch_x,) in DataLoader(TensorDataset(x_val_act_tensor), batch_size=v["gru_batch_size"]*2, shuffle=False):
                logits = gru_model(batch_x.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
            p_buy_active = np.concatenate(val_preds)

        val_active_idx = np.where(val_active)[0]
        val_auc = None
        if context.validation_target_z is not None and (context.validation_target_z > 0).any():
            val_will_buy = (context.validation_target_z[val_active] > 0).astype(np.int32)
            val_auc = float(roc_auc_score(val_will_buy, p_buy_active))
            print(f"  [HYBRID] >>> Causal GRU Validation Active Churn AUC: {val_auc:.6f} <<<", flush=True)

        prediction_z = np.zeros(len(context.users), dtype=np.float64)

        # ── Stage 2: CatBoost Conditional Amount Regressor on Real Buyers ──
        active_buyers_mask = train_active & (train_will_buy.astype(bool))
        x_train_buyers = x_train_tab[active_buyers_mask]
        z_train_buyers = train_z[active_buyers_mask]
        n_buyers = int(active_buyers_mask.sum())
        print(f"  [HYBRID] Active buyers train N={n_buyers}", flush=True)

        cb_amount = CatBoostRegressor(
            iterations=v["amount_iterations"],
            depth=v["amount_depth"],
            learning_rate=v["amount_learning_rate"],
            l2_leaf_reg=v["amount_l2_leaf_reg"],
            loss_function="RMSE",
            thread_count=v["thread_count"],
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        cb_amount.fit(x_train_buyers, z_train_buyers, verbose=False)
        cond_z_val = np.maximum(cb_amount.predict(x_val_tab[val_active]), 0.0)

        # Active cohort final prediction
        prediction_z[val_active_idx] = p_buy_active * cond_z_val

        # ── Stage 3: Inactive Cohort Regressor ─────────────────────────────
        x_train_inact = x_train_tab[~train_active]
        z_train_inact = train_z[~train_active]
        n_inact = int((~train_active).sum())
        val_inact_idx = np.where(~val_active)[0]
        print(f"  [HYBRID] Inactive train N={n_inact}", flush=True)

        cb_inact = CatBoostRegressor(
            iterations=v["inactive_iterations"],
            depth=v["inactive_depth"],
            learning_rate=v["inactive_learning_rate"],
            l2_leaf_reg=v["inactive_l2_leaf_reg"],
            loss_function="RMSE",
            thread_count=v["thread_count"],
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        cb_inact.fit(x_train_inact, z_train_inact, verbose=False)
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
                "val_auc": val_auc,
            },
        )
