"""Two-Tower Sequential Event Network for Direct LTV prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivityTower(nn.Module):
    """Processes search/cart/order counts and recency intervals."""

    def __init__(self, in_features: int = 5, hidden_dim: int = 64, out_dim: int = 32):
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
        # x: (B, L, C), mask: (B, L)
        # Transpose for Conv1d: (B, C, L)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L, hidden)
        out, _ = self.gru(h)
        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        return self.head(pooled)


class MonetaryTower(nn.Module):
    """Processes GMV, ticket size, and monetary values."""

    def __init__(self, in_features: int = 3, hidden_dim: int = 64, out_dim: int = 32):
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
    """Two-Tower Architecture: Activity & Cadence Tower + Monetary Tower + Fusion Head."""

    def __init__(
        self,
        count_features: int = 5,
        total_features: int = 3,
        static_features: int = 7,
        latent_dim: int = 32,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.activity_tower = ActivityTower(in_features=count_features + 1, hidden_dim=64, out_dim=latent_dim)
        self.monetary_tower = MonetaryTower(in_features=total_features, hidden_dim=64, out_dim=latent_dim)
        self.static_proj = nn.Sequential(
            nn.Linear(static_features, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        fusion_in = latent_dim * 3
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
        counts: torch.Tensor,
        totals: torch.Tensor,
        recency: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Log-transform raw counts and totals for numerical stability
        counts_log = torch.log1p(torch.clamp(counts, min=0.0))
        recency_norm = (recency.unsqueeze(-1).float() / 90.0).clamp(0.0, 1.0)
        act_in = torch.cat([counts_log, recency_norm], dim=-1)

        totals_log = torch.log1p(torch.clamp(totals, min=0.0))
        static_log = torch.log1p(torch.clamp(static, min=0.0))

        rep_act = self.activity_tower(act_in, mask)
        rep_mon = self.monetary_tower(totals_log, mask)
        rep_stat = self.static_proj(static_log)

        fused = torch.cat([rep_act, rep_mon, rep_stat], dim=-1)
        feat = self.fusion(fused)

        direct_z = self.head_z(feat).squeeze(-1)
        churn_logit = self.head_churn(feat).squeeze(-1)

        return {"direct_z": direct_z, "churn_logit": churn_logit}
