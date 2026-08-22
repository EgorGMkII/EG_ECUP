"""Loss functions for Direct Regression and Multi-Task Hurdle GRU."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectMSELoss(nn.Module):
    """Standard MSE loss on log1p space."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target_log)


class MultiTaskHurdleLoss(nn.Module):
    """Joint Classification (BCE) and Conditional Regression (MSE) Loss."""

    def __init__(self, lambda_cls: float = 0.5, lambda_reg: float = 1.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        p_logits: torch.Tensor,
        z_cond: torch.Tensor,
        is_buyer: torch.Tensor,
        y_log: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Classification loss across all samples
        cls_loss = self.bce(p_logits, is_buyer)

        # Regression loss strictly on active buyers
        buyer_mask = is_buyer > 0.5
        if buyer_mask.sum() > 0:
            reg_loss = self.mse(z_cond[buyer_mask], y_log[buyer_mask])
        else:
            reg_loss = torch.tensor(0.0, device=p_logits.device)

        total_loss = self.lambda_cls * cls_loss + self.lambda_reg * reg_loss
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_cls": float(cls_loss.item()),
            "loss_reg": float(reg_loss.item()),
        }
        return total_loss, metrics
