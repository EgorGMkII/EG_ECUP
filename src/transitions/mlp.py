"""PyTorch Multi-Task MLP with Masked Lifecycle Losses."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class MultiTaskDataset(Dataset):
    """Dataset for multi-task tabular training with lifecycle masks."""

    def __init__(
        self,
        X: np.ndarray,
        target: np.ndarray,
        past_buyer_30d: np.ndarray,
    ):
        self.X = torch.from_numpy(X).float()
        self.target = torch.from_numpy(target).float()
        self.target_log = torch.log1p(torch.clamp(self.target, min=0.0))
        self.past_buyer_30d = torch.from_numpy(past_buyer_30d).float()
        self.future_buyer_30d = (self.target > 0).float()
        # Churn target (1 if churned to 0)
        self.y_churn = (1.0 - self.future_buyer_30d)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.target_log[idx],
            self.future_buyer_30d[idx],
            self.past_buyer_30d[idx],
            self.y_churn[idx],
        )


class MultiTaskMLP(nn.Module):
    """Multi-Task Feed-Forward Neural Network with shared representations and specialized transition heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, embed_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-Task Heads
        self.head_reactivation = nn.Linear(embed_dim, 1) # P(reactivation) for past=0
        self.head_churn = nn.Linear(embed_dim, 1)        # P(churn) for past=1
        self.head_buy30d = nn.Linear(embed_dim, 1)      # General P(buy)
        self.head_conditional = nn.Linear(embed_dim, 1) # E[log1p(GMV) | buy]
        self.head_direct = nn.Linear(embed_dim, 1)      # Direct log1p(GMV)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.shared_encoder(x)
        logit_react = self.head_reactivation(feat).squeeze(-1)
        logit_churn = self.head_churn(feat).squeeze(-1)
        logit_buy = self.head_buy30d(feat).squeeze(-1)
        pred_cond = torch.relu(self.head_conditional(feat).squeeze(-1))
        pred_dir = torch.relu(self.head_direct(feat).squeeze(-1))
        return logit_react, logit_churn, logit_buy, pred_cond, pred_dir


def train_multitask_mlp(
    model: MultiTaskMLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Trains the MultiTaskMLP with masked task-specific losses."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss_fn = nn.MSELoss(reduction="none")

    history = {"train_loss": [], "val_rmsle": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for x, y_log, fut_buy, past_buy, y_churn in train_loader:
            x = x.to(device)
            y_log = y_log.to(device)
            fut_buy = fut_buy.to(device)
            past_buy = past_buy.to(device)
            y_churn = y_churn.to(device)

            optimizer.zero_grad()
            logit_react, logit_churn, logit_buy, pred_cond, pred_dir = model(x)

            # Mask 1: Reactivation loss only where past_buyer == 0
            mask_dormant = (past_buy == 0)
            if mask_dormant.sum() > 0:
                loss_react = bce_loss_fn(logit_react[mask_dormant], fut_buy[mask_dormant]).mean()
            else:
                loss_react = torch.tensor(0.0, device=device)

            # Mask 2: Churn loss only where past_buyer == 1
            mask_active = (past_buy == 1)
            if mask_active.sum() > 0:
                loss_churn = bce_loss_fn(logit_churn[mask_active], y_churn[mask_active]).mean()
            else:
                loss_churn = torch.tensor(0.0, device=device)

            # General Buy 30d
            loss_buy = bce_loss_fn(logit_buy, fut_buy).mean()

            # Mask 3: Conditional GMV only on actual buyers (future_buy == 1)
            mask_buyers = (fut_buy == 1)
            if mask_buyers.sum() > 0:
                loss_cond = mse_loss_fn(pred_cond[mask_buyers], y_log[mask_buyers]).mean()
            else:
                loss_cond = torch.tensor(0.0, device=device)

            # Direct GMV on all
            loss_dir = mse_loss_fn(pred_dir, y_log).mean()

            # Combined Multi-Task Loss
            loss = (
                1.0 * loss_dir
                + 0.5 * loss_cond
                + 0.5 * loss_react
                + 0.5 * loss_churn
                + 0.2 * loss_buy
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        if verbose:
            print(f"  MLP Epoch [{ep:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f}")

    return history
