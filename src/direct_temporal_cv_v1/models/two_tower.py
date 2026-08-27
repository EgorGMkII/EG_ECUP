"""Two-Tower Sequential Event Network for Direct LTV prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivityTower(nn.Module):
    """Processes search/cart/order counts and temporal intervals."""

    def __init__(self, in_features: int = 12, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C), mask: (B, L) where True indicates valid event (or padding)
        # mask is True for valid events or not mask for valid
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L, hidden)
        out, _ = self.gru(h)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        return self.head(pooled)


class MonetaryTower(nn.Module):
    """Processes GMV, ticket sizes, and spend history."""

    def __init__(self, in_features: int = 12, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(h)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        return self.head(pooled)


class TwoTowerEventNet(nn.Module):
    """Two-Tower Architecture: Activity Tower + Monetary Tower + Fusion Head."""

    def __init__(
        self,
        content_features: int = 12,
        time_features: int = 12,
        latent_dim: int = 32,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.activity_tower = ActivityTower(in_features=time_features, hidden_dim=64, out_dim=latent_dim)
        self.monetary_tower = MonetaryTower(in_features=content_features, hidden_dim=64, out_dim=latent_dim)
        
        fusion_in = latent_dim * 2 + 1  # 2 towers + empty flag
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.head_z = nn.Linear(32, 1)
        self.head_churn = nn.Linear(32, 1)

    def forward(
        self,
        content: torch.Tensor,
        time_feat: torch.Tensor,
        mask: torch.Tensor,
        empty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # content: (B, L, 12), time_feat: (B, L, 12), mask: (B, L), empty: (B,)
        valid_mask = ~mask if mask.dtype == torch.bool else mask

        rep_act = self.activity_tower(time_feat, valid_mask)
        rep_mon = self.monetary_tower(content, valid_mask)
        empty_feat = empty.unsqueeze(-1).float()

        fused = torch.cat([rep_act, rep_mon, empty_feat], dim=-1)
        feat = self.fusion(fused)

        direct_z = self.head_z(feat).squeeze(-1)
        churn_logit = self.head_churn(feat).squeeze(-1)

        return {"direct_z": direct_z, "churn_logit": churn_logit}
