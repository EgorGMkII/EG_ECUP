"""Reserved causal TCN architecture for a future React/Churn adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TCNRecipe:
    input_features: int = 15
    history_days: int = 180
    channels: int = 128
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    dropout: float = 0.10

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._left_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, values: Tensor) -> Tensor:
        return super().forward(nn.functional.pad(values, (self._left_padding, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.first = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.second = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.dropout(nn.functional.gelu(self.norm1(self.first(values))))
        values = self.dropout(nn.functional.gelu(self.norm2(self.second(values))))
        return nn.functional.gelu(values + residual)


class TCNTransitionBase(nn.Module):
    """Future joint base; V1 exposes only React/Churn at adapter level."""

    implementation_id = "tcn_daily180x15_causal_residual_v1"

    def __init__(self, recipe: TCNRecipe = TCNRecipe()) -> None:
        super().__init__()
        if recipe.receptive_field < recipe.history_days:
            raise ValueError("TCN receptive field must cover full daily history")
        self.recipe = recipe
        self.stem = CausalConv1d(recipe.input_features, recipe.channels, 1)
        self.blocks = nn.ModuleList([TCNResidualBlock(recipe.channels, dilation, recipe.dropout) for dilation in recipe.dilations])
        self.attention = nn.Sequential(nn.Linear(recipe.channels, recipe.channels // 2), nn.GELU(), nn.Linear(recipe.channels // 2, 1))
        self.embedding = nn.Linear(recipe.channels * 2, 128)
        self.reactivation = nn.Linear(128, 1)
        self.churn = nn.Linear(128, 1)

    def forward(self, daily: Tensor) -> dict[str, Tensor]:
        if daily.ndim != 3 or daily.shape[1:] != (self.recipe.history_days, self.recipe.input_features):
            raise ValueError("Expected [batch, 180, 15] daily tensor")
        values = self.stem(daily.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        sequence = values.transpose(1, 2)
        attention = torch.softmax(self.attention(sequence), dim=1)
        pooled = (sequence * attention).sum(dim=1)
        representation = nn.functional.gelu(self.embedding(torch.cat((sequence[:, -1], pooled), dim=1)))
        return {"reactivation_logit": self.reactivation(representation).squeeze(-1), "churn_logit": self.churn(representation).squeeze(-1)}
