"""Neural Architectures for Sequence Processing: Direct GRU & Multi-Task Hurdle GRU."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Computes learned attention weights across time steps [B, T, H] -> [B, H]."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: [B, T, H]
        scores = self.attn(rnn_out)  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        context = torch.sum(rnn_out * weights, dim=1)  # [B, H]
        return context


class DirectGRUModel(nn.Module):
    """Direct GRU Baseline mapping sequence of daily logs directly to z = log1p(GMV)."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        gru_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        eff_hidden = gru_hidden * 2 if bidirectional else gru_hidden
        self.attention = TemporalAttention(eff_hidden) if use_attention else None

        self.head = nn.Sequential(
            nn.Linear(eff_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        rnn_out, _ = self.gru(x)  # [B, T, H]

        if self.attention is not None:
            emb = self.attention(rnn_out)  # [B, H]
        else:
            emb = rnn_out[:, -1, :]  # [B, H]

        out = self.head(emb).squeeze(-1)  # [B]
        return out, emb


class MultiTaskGRUModel(nn.Module):
    """Multi-Task GRU Encoder with joint Classification (P(buy)) and Conditional Regression heads."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        gru_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        eff_hidden = gru_hidden * 2 if bidirectional else gru_hidden
        self.attention = TemporalAttention(eff_hidden) if use_attention else None

        # Head 1: Classification P(target > 0)
        self.head_clf = nn.Sequential(
            nn.Linear(eff_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Head 2: Conditional Regression E[ln(1 + GMV) | buy]
        self.head_reg = nn.Sequential(
            nn.Linear(eff_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        rnn_out, _ = self.gru(x)  # [B, T, H]

        if self.attention is not None:
            emb = self.attention(rnn_out)  # [B, H]
        else:
            emb = rnn_out[:, -1, :]  # [B, H]

        p_logits = self.head_clf(emb).squeeze(-1)  # [B]
        z_cond = self.head_reg(emb).squeeze(-1)  # [B]
        return p_logits, z_cond, emb


class MultiTaskTransitionGRUModel(nn.Module):
    """Multi-Task GRU Encoder with dedicated Transition Heads (Reactivation, Churn, Buy, Cond, Direct)."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)

        self.head_reactivation = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rnn_out, _ = self.gru(x)
        emb = self.attention(rnn_out)

        l_react = self.head_reactivation(emb).squeeze(-1)
        l_churn = self.head_churn(emb).squeeze(-1)
        l_buy = self.head_buy(emb).squeeze(-1)
        z_cond = self.head_cond(emb).squeeze(-1)
        z_dir = self.head_dir(emb).squeeze(-1)
        return l_react, l_churn, l_buy, z_cond, z_dir, emb


class PatchTransformer365Model(nn.Module):
    """Patch Transformer Encoder partitioning 365 daily steps into 7-day tokens with Temporal Attention."""

    def __init__(
        self,
        input_dim: int = 15,
        patch_size: int = 7,
        num_patches: int = 52,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model

        # Project patch of shape [B, 52, 7 * 15] -> [B, 52, d_model]
        self.patch_proj = nn.Linear(patch_size * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = TemporalAttention(d_model)

        # Multi-task heads
        self.head_reactivation = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 365, 15] -> take last 52*7 = 364 days
        x_trim = x[:, -self.num_patches * self.patch_size :, :]  # [B, 364, 15]
        B = x_trim.shape[0]
        x_patches = x_trim.reshape(B, self.num_patches, self.patch_size * x.shape[-1])  # [B, 52, 105]

        tokens = self.patch_proj(x_patches) + self.pos_embed  # [B, 52, 128]
        enc_out = self.transformer(tokens)  # [B, 52, 128]
        emb = self.attention(enc_out)  # [B, 128]

        l_react = self.head_reactivation(emb).squeeze(-1)
        l_churn = self.head_churn(emb).squeeze(-1)
        l_buy = self.head_buy(emb).squeeze(-1)
        z_cond = self.head_cond(emb).squeeze(-1)
        z_dir = self.head_dir(emb).squeeze(-1)
        return l_react, l_churn, l_buy, z_cond, z_dir, emb


class HierarchicalGRUModel(nn.Module):
    """Hierarchical Sequence Model: High-resolution Daily GRU (90d) + Low-resolution Weekly GRU (275d)."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_daily: int = 96,
        hidden_weekly: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.daily_gru = nn.GRU(input_dim, hidden_daily, num_layers=2, batch_first=True, dropout=dropout)
        self.daily_attn = TemporalAttention(hidden_daily)

        self.weekly_gru = nn.GRU(input_dim, hidden_weekly, num_layers=1, batch_first=True)
        self.weekly_attn = TemporalAttention(hidden_weekly)

        comb_dim = hidden_daily + hidden_weekly
        self.head_reactivation = nn.Sequential(nn.Linear(comb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_churn = nn.Sequential(nn.Linear(comb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy = nn.Sequential(nn.Linear(comb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(comb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(comb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 365, 15]
        x_recent = x[:, -90:, :]  # [B, 90, 15]
        x_older = x[:, :-90, :]   # [B, 275, 15]

        # Daily GRU on recent 90d
        out_recent, _ = self.daily_gru(x_recent)
        emb_recent = self.daily_attn(out_recent)

        # Weekly pooling on older 275d (take 39 weeks of 7 days)
        B = x_older.shape[0]
        x_older_trim = x_older[:, -39*7:, :]  # [B, 273, 15]
        x_weekly = x_older_trim.reshape(B, 39, 7, x.shape[-1]).mean(dim=2)  # [B, 39, 15]
        out_weekly, _ = self.weekly_gru(x_weekly)
        emb_weekly = self.weekly_attn(out_weekly)

        emb = torch.cat([emb_recent, emb_weekly], dim=-1)  # [B, 160]

        l_react = self.head_reactivation(emb).squeeze(-1)
        l_churn = self.head_churn(emb).squeeze(-1)
        l_buy = self.head_buy(emb).squeeze(-1)
        z_cond = self.head_cond(emb).squeeze(-1)
        z_dir = self.head_dir(emb).squeeze(-1)
        return l_react, l_churn, l_buy, z_cond, z_dir, emb


class MultiHorizonGRUModel(nn.Module):
    """Multi-Horizon GRU Encoder with joint auxiliary heads (7d, 14d, 30d) and Conditional Regressor."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)

        self.head_buy_7d = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy_14d = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_buy_30d = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.head_cond = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rnn_out, _ = self.gru(x)
        emb = self.attention(rnn_out)

        l_7d = self.head_buy_7d(emb).squeeze(-1)
        l_14d = self.head_buy_14d(emb).squeeze(-1)
        l_30d = self.head_buy_30d(emb).squeeze(-1)
        z_cond = self.head_cond(emb).squeeze(-1)
        return l_7d, l_14d, l_30d, z_cond, emb


class DiscreteHazardGRUModel(nn.Module):
    """Discrete Hazard GRU Model predicting conditional hazard across 4 weekly intervals (1-7d, 8-14d, 15-21d, 22-30d)."""

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)

        # Hazard logits for 4 discrete intervals
        self.head_hazard = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 4))
        self.head_cond = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rnn_out, _ = self.gru(x)
        emb = self.attention(rnn_out)

        hazard_logits = self.head_hazard(emb)  # [B, 4]
        z_cond = self.head_cond(emb).squeeze(-1)  # [B]
        return hazard_logits, z_cond, emb



