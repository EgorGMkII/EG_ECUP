"""ETT binary classification adapter with focal/asymmetric loss options.

Predicts P(will_buy_30d) for temporal four-fold validation.
Can be evaluated standalone on AUC/Logloss or combined with CatBoost/amount regressors.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import torch
from torch import nn
from torch.nn import functional as F

from ..base import DirectModelAdapter, FoldContext, FoldPrediction, ModelConfig, ModelRequirements
from src.ssl_temporal_stack_v1.models import EventTimeTransformer


class FocalLoss(nn.Module):
    """Focal Loss with optional positive class weighting and asymmetric hardness."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_factor = (1.0 - p_t).clamp(min=1e-6, max=1.0).pow(self.gamma)
        
        # Binary cross entropy with logits
        pos_weight_tensor = torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor, reduction="none")
        return (alpha_t * focal_factor * bce).mean()


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon, Khosla et al.) for binary active classification."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=features.device)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=features.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        norm_features = F.normalize(features, dim=-1)
        anchor_dot_contrast = torch.div(torch.matmul(norm_features, norm_features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        valid_pos = mask.sum(1) > 0
        if not valid_pos.any():
            return torch.tensor(0.0, device=features.device)

        return -mean_log_prob_pos[valid_pos].mean()


class ETTClassifierAdapter(DirectModelAdapter):
    model_id = "ett_classifier"
    requirements = ModelRequirements(event_sequences=True, tabular_features=True)

    def validate_config(self, raw: Mapping[str, Any]) -> ModelConfig:
        allowed = {
            "epochs", "batch_size", "learning_rate", "scheduler",
            "warmup_fraction", "weight_decay", "dropout", "history_days",
            "loss_type", "focal_gamma", "focal_alpha", "pos_weight",
            "active_only", "activity_window_days", "supcon_weight", "temperature",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown ett_classifier fields: {sorted(unknown)}")
        loss_type = str(raw.get("loss_type", "bce"))
        if loss_type not in {"bce", "focal"}:
            raise ValueError("loss_type must be 'bce' or 'focal'")
        values = {
            "epochs": int(raw.get("epochs", 2)),
            "batch_size": int(raw.get("batch_size", 512)),
            "learning_rate": float(raw.get("learning_rate", 3e-4)),
            "scheduler": str(raw.get("scheduler", "cosine")),
            "warmup_fraction": float(raw.get("warmup_fraction", 0.1)),
            "weight_decay": float(raw.get("weight_decay", 1e-4)),
            "dropout": float(raw.get("dropout", 0.1)),
            "history_days": int(raw.get("history_days", 180)),
            "loss_type": loss_type,
            "focal_gamma": float(raw.get("focal_gamma", 2.0)),
            "focal_alpha": float(raw.get("focal_alpha", 0.5)),
            "pos_weight": float(raw.get("pos_weight", 1.0)),
            "active_only": bool(raw.get("active_only", False)),
            "activity_window_days": int(raw.get("activity_window_days", 90)),
            "supcon_weight": float(raw.get("supcon_weight", 0.0)),
            "temperature": float(raw.get("temperature", 0.1)),
        }
        if values["epochs"] not in {1, 2, 3, 4, 5}:
            raise ValueError("ett_classifier epochs must be between 1 and 5")
        if values["history_days"] != 180:
            raise ValueError("ett_classifier history_days is pinned to 180 in v1")
        if values["scheduler"] not in {"constant", "linear", "cosine"}:
            raise ValueError("Unsupported ETT scheduler")
        return ModelConfig(self.model_id, values)

    def fit_predict_fold(self, context: FoldContext, config: ModelConfig) -> FoldPrediction:
        if context.train_events is None or context.validation_events is None:
            raise ValueError("ett_classifier requires event sequences")

        v = config.values
        device = context.device
        model = EventTimeTransformer(
            transformer_dropout=v["dropout"],
            head_dropout=v["dropout"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=v["learning_rate"],
            weight_decay=v["weight_decay"]
        )

        train_arrays = context.train_events
        valid_arrays = context.validation_events
        train_z = context.train_target_z
        train_will_buy = (train_z > 0).astype(np.float32)

        # Optional active-only filter
        if v["active_only"]:
            if context.train_tabular is None or context.validation_tabular is None:
                raise ValueError("active_only requires tabular context for activity masking")
            act_col = f"gmv_sum_{v['activity_window_days']}d"
            train_active = context.train_tabular[act_col].to_numpy() > 0
            val_active = context.validation_tabular[act_col].to_numpy() > 0
            train_indices = np.where(train_active)[0]
        else:
            train_indices = np.arange(len(context.users))
            val_active = np.ones(len(context.users), dtype=bool)

        # Loss function
        if v["loss_type"] == "focal":
            criterion: nn.Module = FocalLoss(
                gamma=v["focal_gamma"],
                alpha=v["focal_alpha"],
                pos_weight=v["pos_weight"]
            )
        else:
            pos_weight_t = torch.tensor(v["pos_weight"], device=device) if v["pos_weight"] != 1.0 else None
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

        supcon_loss_fn = SupervisedContrastiveLoss(temperature=v["temperature"]) if v["supcon_weight"] > 0 else None

        started = time.perf_counter()
        batch_size = v["batch_size"]
        total_steps = v["epochs"] * (len(train_indices) // batch_size + 1)
        warmup_steps = int(total_steps * v["warmup_fraction"])

        def get_lr_factor(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / max(1, warmup_steps)
            if v["scheduler"] == "constant":
                return 1.0
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            if v["scheduler"] == "linear":
                return max(0.0, 1.0 - progress)
            # cosine
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        global_step = 0
        model.train()
        for epoch in range(v["epochs"]):
            order = train_indices.copy()
            rng = np.random.default_rng(context.root_seed + epoch)
            rng.shuffle(order)

            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                inputs = tuple(torch.as_tensor(array[idx], device=device) for array in train_arrays)
                inputs = (inputs[0].float(), inputs[1].float(), inputs[2].long(), inputs[3].bool(), inputs[4].bool())
                target = torch.as_tensor(train_will_buy[idx], device=device).float()

                # Use reactivation head as binary classification logit
                embedding, _ = model.encode(*inputs)
                logits = model.reactivation(embedding).squeeze(-1)
                loss = criterion(logits, target)

                if supcon_loss_fn is not None:
                    loss_sc = supcon_loss_fn(embedding, target)
                    loss = loss + v["supcon_weight"] * loss_sc

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update lr
                lr_factor = get_lr_factor(global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = v["learning_rate"] * lr_factor
                global_step += 1

        # Inference on validation set
        model.eval()
        val_logits_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(context.users), batch_size):
                idx = slice(start, start + batch_size)
                inputs = tuple(torch.as_tensor(array[idx], device=device) for array in valid_arrays)
                inputs = (inputs[0].float(), inputs[1].float(), inputs[2].long(), inputs[3].bool(), inputs[4].bool())
                embedding, _ = model.encode(*inputs)
                logits = model.reactivation(embedding).squeeze(-1)
                val_logits_list.append(logits.detach().cpu().numpy())

        val_logits = np.concatenate(val_logits_list).astype(np.float64)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        # Metrics evaluation
        val_z = context.validation_target_z
        val_will_buy = (val_z > 0).astype(np.int32)

        val_mask = val_active
        y_true_eval = val_will_buy[val_mask]
        p_eval = val_probs[val_mask]

        val_auc = float(roc_auc_score(y_true_eval, p_eval))
        val_logloss = float(log_loss(y_true_eval, p_eval))
        val_brier = float(brier_score_loss(y_true_eval, p_eval))

        elapsed = time.perf_counter() - started
        print(f"  [ETT_CLASSIFIER] val_auc={val_auc:.6f} val_logloss={val_logloss:.6f} brier={val_brier:.6f} (elapsed {elapsed:.1f}s)", flush=True)

        report = {
            "model_id": self.model_id,
            "fold_id": context.fold.fold_id,
            "epochs": v["epochs"],
            "loss_type": v["loss_type"],
            "val_auc": val_auc,
            "val_logloss": val_logloss,
            "val_brier": val_brier,
            "elapsed_seconds": elapsed,
            "active_only": v["active_only"],
        }

        # For prediction_z in direct pipeline: return calibrated probability as logit-transformed or probability
        # In hurdle stack, this prediction bank provides the P(buy) probabilities
        return FoldPrediction(self.model_id, np.asarray(context.users), val_probs, report)
