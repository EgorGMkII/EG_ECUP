"""Specialized Neural Amount Specialist (E[log1p(future_gmv_30d) | future_gmv_30d > 0]).

Trained strictly on positive spenders (future_gmv_30d > 0) with MSE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralAmountSpecialist(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int = 128, dropout: float = 0.10):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model

        # Dedicated Conditional Amount Head (outputs softplus z)
        self.amount_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, *inputs) -> torch.Tensor:
        """Returns positive conditional magnitude conditional_z >= 0."""
        if hasattr(self.encoder, "extract_embedding"):
            emb = self.encoder.extract_embedding(*inputs)
        elif hasattr(self.encoder, "pooling_mlp"):
            content, time_feat, ranks, mask, empty = inputs
            b, s, _ = content.shape
            content_emb = self.encoder.content_projection(content)
            time_emb = self.encoder.time_mlp(time_feat)
            rank_emb = self.encoder.event_rank_embedding(ranks)
            event_token = self.encoder.input_layer_norm(content_emb + time_emb + rank_emb)
            empty_exp = self.encoder.empty_history_token.expand(b, s, -1)
            event_token = torch.where(empty.unsqueeze(1).unsqueeze(2), empty_exp, event_token)

            h = self.encoder.transformer_encoder(event_token, src_key_padding_mask=mask)
            last_token = h[:, -1, :]
            valid_mask = (~mask).unsqueeze(-1).float()
            sum_pooled = (h * valid_mask).sum(dim=1)
            mean_pooled = sum_pooled / valid_mask.sum(dim=1).clamp(min=1.0)
            h_masked = h.masked_fill(mask.unsqueeze(-1), -1e9)
            max_pooled = torch.where(empty.unsqueeze(-1), last_token, h_masked.max(dim=1).values)
            emb = self.encoder.pooling_mlp(torch.cat([last_token, mean_pooled, max_pooled], dim=-1))
        else:
            emb = self.encoder(*inputs)

        cond_z = F.softplus(self.amount_head(emb).squeeze(-1))
        return cond_z
