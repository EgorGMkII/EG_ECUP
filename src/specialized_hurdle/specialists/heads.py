"""Specialized Neural Heads for Reactivation, Churn, and Positive Amount."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactivationHead(nn.Module):
    """Outputs raw logit for reactivation probability P(will_buy | was_active == 0)."""
    def __init__(self, d_in: int = 128, d_hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


class ChurnHead(nn.Module):
    """Outputs raw logit for churn probability P(not will_buy | was_active == 1)."""
    def __init__(self, d_in: int = 128, d_hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)


class AmountHead(nn.Module):
    """Outputs positive conditional magnitude conditional_z >= 0."""
    def __init__(self, d_in: int = 128, d_hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(emb).squeeze(-1))
