"""Executable V1 S1/S2/ETT models, objectives, and specialist copies."""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

    def forward(self, values: Tensor) -> Tensor:
        return (values * torch.softmax(self.layers(values), dim=1)).sum(dim=1)


class GRUBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(15, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = TemporalAttention(128)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        sequence, _ = self.gru(x)
        return sequence, self.attention(sequence)


def _head() -> nn.Sequential:
    return nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))


class S1MaskedPretrainer(nn.Module):
    implementation_id = "s1_random_day_mask12ch_gru180_attention_v1"
    def __init__(self) -> None:
        super().__init__(); self.encoder = GRUBackbone(); self.reconstruction = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 15))
    def forward(self, x: Tensor) -> Tensor:
        sequence, _ = self.encoder(x); return self.reconstruction(sequence)
    @staticmethod
    def corrupt(x: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        mask = torch.rand(x.shape[:2], device=x.device, generator=generator) < 0.20
        corrupted = x.clone(); corrupted[:, :, :12][mask] = 0.0
        return corrupted, mask
    @staticmethod
    def loss(reconstructed: Tensor, original: Tensor, mask: Tensor) -> Tensor:
        return F.smooth_l1_loss(reconstructed[mask], original[mask])


class S2MultiHorizonPretrainer(nn.Module):
    implementation_id = "s2_multihorizon_buy_gmv_gru180_attention_v1"
    def __init__(self) -> None:
        super().__init__(); self.encoder = GRUBackbone()
        self.buy = nn.ModuleList([_head() for _ in range(3)]); self.gmv = nn.ModuleList([_head() for _ in range(3)])
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        _, embedding = self.encoder(x)
        return {**{f"buy_{h}": head(embedding).squeeze(-1) for h, head in zip((7, 14, 30), self.buy)}, **{f"gmv_{h}": head(embedding).squeeze(-1) for h, head in zip((7, 14, 30), self.gmv)}}
    @staticmethod
    def loss(outputs: dict[str, Tensor], buy: dict[int, Tensor], gmv_z: dict[int, Tensor]) -> Tensor:
        return sum(F.binary_cross_entropy_with_logits(outputs[f"buy_{h}"], buy[h]) for h in (7, 14, 30)) * (0.55 / 3) + sum(F.smooth_l1_loss(outputs[f"gmv_{h}"], gmv_z[h]) for h in (7, 14, 30)) * (0.45 / 3)


class TransitionBase(nn.Module):
    def __init__(self, encoder: nn.Module, embedding: Callable[[Tensor], Tensor]) -> None:
        super().__init__(); self.encoder = encoder; self._embedding = embedding
        self.reactivation, self.churn, self.direct, self.conditional = _head(), _head(), _head(), _head()
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        emb = self._embedding(x)
        return {"reactivation_logit": self.reactivation(emb).squeeze(-1), "churn_logit": self.churn(emb).squeeze(-1), "direct_z": self.direct(emb).squeeze(-1), "conditional_z": self.conditional(emb).squeeze(-1)}


def transition_loss(outputs: dict[str, Tensor], z_target: Tensor, active: Tensor, will_buy: Tensor) -> Tensor:
    factorized = torch.where(active.bool(), torch.sigmoid(-outputs["churn_logit"]), torch.sigmoid(outputs["reactivation_logit"])) * torch.clamp(outputs["conditional_z"], min=0)
    loss = F.mse_loss(factorized, z_target) + .25 * F.mse_loss(outputs["direct_z"], z_target)
    positive, inactive, active_mask = z_target > 0, ~active.bool(), active.bool()
    if positive.any(): loss = loss + .25 * F.mse_loss(outputs["conditional_z"][positive], z_target[positive])
    if inactive.any(): loss = loss + .10 * F.binary_cross_entropy_with_logits(outputs["reactivation_logit"][inactive], will_buy[inactive])
    if active_mask.any(): loss = loss + .10 * F.binary_cross_entropy_with_logits(outputs["churn_logit"][active_mask], 1 - will_buy[active_mask])
    return loss


class EventTimeTransformer(nn.Module):
    implementation_id = "ett_record_builder_last_token_ff512_v1"
    def __init__(self) -> None:
        super().__init__()
        self.content = nn.Linear(12, 128); self.time = nn.Linear(12, 128); self.rank = nn.Embedding(181, 128); self.norm = nn.LayerNorm(128)
        layer = nn.TransformerEncoderLayer(128, 4, 512, 0.1, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2); self.reactivation, self.churn, self.direct, self.conditional = _head(), _head(), _head(), _head()
    def encode(self, content: Tensor, time_features: Tensor, ranks: Tensor, padding_mask: Tensor, empty: Tensor) -> tuple[Tensor, Tensor]:
        safe_mask = padding_mask.clone(); safe_mask[padding_mask.all(dim=1), 0] = False
        values = self.transformer(self.norm(self.content(content) + self.time(time_features) + self.rank(ranks.clamp(0, 180))), src_key_padding_mask=safe_mask)
        embedding = torch.where(empty.unsqueeze(1), torch.zeros_like(values[:, -1]), values[:, -1])
        return embedding, values
    def forward(self, content: Tensor, time_features: Tensor, ranks: Tensor, padding_mask: Tensor, empty: Tensor) -> dict[str, Tensor]:
        emb, _ = self.encode(content, time_features, ranks, padding_mask, empty)
        return {"reactivation_logit": self.reactivation(emb).squeeze(-1), "churn_logit": self.churn(emb).squeeze(-1), "direct_z": self.direct(emb).squeeze(-1), "conditional_z": self.conditional(emb).squeeze(-1)}


class Specialist(nn.Module):
    """Independent deep-copied encoder plus a fresh task head."""
    def __init__(self, encoder: nn.Module, task: str, kind: str) -> None:
        super().__init__(); self.encoder = copy.deepcopy(encoder); self.task, self.kind = task, kind; self.head = _head()
    def freeze_phase_h(self) -> None:
        for parameter in self.encoder.parameters(): parameter.requires_grad = False
    def unfreeze_phase_f(self) -> None:
        for parameter in self.encoder.parameters(): parameter.requires_grad = False
        if self.kind in {"s1", "s2"}:
            for name, parameter in self.encoder.gru.named_parameters():
                parameter.requires_grad = name.endswith("_l1")
            for parameter in self.encoder.attention.parameters(): parameter.requires_grad = True
        else:
            for parameter in self.encoder.transformer.layers[-1].parameters(): parameter.requires_grad = True
            for parameter in self.encoder.norm.parameters(): parameter.requires_grad = True

    def forward(self, *inputs: Tensor) -> Tensor:
        if self.kind in {"s1", "s2"}:
            _, embedding = self.encoder(inputs[0])
        else:
            embedding, _ = self.encoder.encode(*inputs)
        return self.head(embedding).squeeze(-1)
