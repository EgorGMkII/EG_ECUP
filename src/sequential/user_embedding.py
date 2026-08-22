"""User-Embedding Residual Architectures for Multi-Task Sequence Processing (E0, E1, E2, E3)."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sequential.models import TemporalAttention


class UserEmbeddingResidualGRU(nn.Module):
    """Multi-Task GRU with pluggable residual User Embedding and Bias branches.
    
    Supports:
    - 'E0': Canonical GRU-180 baseline (no user-specific parameters).
    - 'E1': Personal scalar biases for reactivation and churn classifiers only.
    - 'E2': 8-dim User Embedding with LayerNorm + Dropout for reactivation and churn classifiers only.
    - 'E3': 8-dim User Embedding for classifiers + bounded conditional regression residual.
    """

    def __init__(
        self,
        variant: str = "E0",
        num_users: int = 250000,
        embedding_dim: int = 8,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        embedding_dropout: float = 0.15,
        cond_delta_limit: float = 2.0,
    ):
        super().__init__()
        assert variant in ["E0", "E1", "E2", "E3"], f"Unknown variant: {variant}"
        self.variant = variant
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cond_delta_limit = cond_delta_limit

        # 1. Base Shared GRU Backbone
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.attention = TemporalAttention(hidden_dim)

        # 2. Base Multi-Task Heads
        self.head_reactivation = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_churn = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_buy = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_cond = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_dir = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

        # 3. User Specific Branches
        if variant == "E1":
            # Scalar biases for classifiers only (0-initialized)
            self.user_bias_react = nn.Embedding(num_users + 1, 1, padding_idx=0)
            self.user_bias_churn = nn.Embedding(num_users + 1, 1, padding_idx=0)
            nn.init.zeros_(self.user_bias_react.weight)
            nn.init.zeros_(self.user_bias_churn.weight)

        elif variant in ["E2", "E3"]:
            # User Embedding table: [250001, embedding_dim], padding_idx=0
            self.user_embedding = nn.Embedding(
                num_embeddings=num_users + 1,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
            # Small random init (std=0.02)
            nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.user_embedding.weight[0].fill_(0.0)

            self.emb_ln = nn.LayerNorm(embedding_dim)
            self.emb_dropout = nn.Dropout(embedding_dropout)

            # Residual MLP for Reactivation: (hidden_dim + embedding_dim) -> 32 -> GELU -> 1
            self.mlp_react = nn.Sequential(
                nn.Linear(hidden_dim + embedding_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            # Zero-init last layer so delta_react at start is exactly 0.0
            nn.init.zeros_(self.mlp_react[-1].weight)
            nn.init.zeros_(self.mlp_react[-1].bias)

            # Residual MLP for Churn: (hidden_dim + embedding_dim) -> 32 -> GELU -> 1
            self.mlp_churn = nn.Sequential(
                nn.Linear(hidden_dim + embedding_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            # Zero-init last layer so delta_churn at start is exactly 0.0
            nn.init.zeros_(self.mlp_churn[-1].weight)
            nn.init.zeros_(self.mlp_churn[-1].bias)

            if variant == "E3":
                # Residual Bounded MLP for Conditional Amount: (hidden_dim + embedding_dim) -> 32 -> GELU -> 1
                self.mlp_cond = nn.Sequential(
                    nn.Linear(hidden_dim + embedding_dim, 32),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                )
                nn.init.zeros_(self.mlp_cond[-1].weight)
                nn.init.zeros_(self.mlp_cond[-1].bias)

    def forward(
        self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, input_dim]
        rnn_out, _ = self.gru(x)
        emb = self.attention(rnn_out)  # [B, hidden_dim]

        base_react = self.head_reactivation(emb).squeeze(-1)
        base_churn = self.head_churn(emb).squeeze(-1)
        base_buy = self.head_buy(emb).squeeze(-1)
        base_cond = self.head_cond(emb).squeeze(-1)
        base_dir = self.head_dir(emb).squeeze(-1)

        if self.variant == "E0" or user_idx is None:
            return base_react, base_churn, base_buy, base_cond, base_dir, emb

        if self.variant == "E1":
            b_r = self.user_bias_react(user_idx).squeeze(-1)
            b_c = self.user_bias_churn(user_idx).squeeze(-1)
            return base_react + b_r, base_churn + b_c, base_buy, base_cond, base_dir, emb

        if self.variant in ["E2", "E3"]:
            raw_e = self.user_embedding(user_idx)  # [B, embedding_dim]
            e = self.emb_dropout(self.emb_ln(raw_e))
            he = torch.cat([emb, e], dim=-1)  # [B, hidden_dim + embedding_dim]

            delta_react = self.mlp_react(he).squeeze(-1)
            delta_churn = self.mlp_churn(he).squeeze(-1)

            final_react = base_react + delta_react
            final_churn = base_churn + delta_churn
            final_cond = base_cond

            if self.variant == "E3":
                raw_delta_cond = self.mlp_cond(he).squeeze(-1)
                delta_cond = self.cond_delta_limit * torch.tanh(raw_delta_cond)
                final_cond = base_cond + delta_cond

            return final_react, final_churn, base_buy, final_cond, base_dir, emb

        return base_react, base_churn, base_buy, base_cond, base_dir, emb

    def get_embedding_diagnostics(self) -> Dict[str, float]:
        """Calculates embedding norms and dimension stats for monitoring."""
        if self.variant not in ["E2", "E3"]:
            return {}
        with torch.no_grad():
            weights = self.user_embedding.weight[1:].detach().cpu().numpy()  # exclude UNK
            norms = np.linalg.norm(weights, axis=-1)
            dim_stds = np.std(weights, axis=0)
            zero_fraction = float(np.mean(norms < 1e-4))
            return {
                "norm_mean": float(np.mean(norms)),
                "norm_p50": float(np.median(norms)),
                "norm_p90": float(np.percentile(norms, 90)),
                "norm_p99": float(np.percentile(norms, 99)),
                "norm_max": float(np.max(norms)),
                "mean_dim_std": float(np.mean(dim_stds)),
                "zero_norm_fraction": zero_fraction,
            }
