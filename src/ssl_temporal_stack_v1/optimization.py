"""Step-counted AMP optimizer control for SSL V1."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class StepOptimizer:
    """Own one exact optimizer schedule and reject skipped AMP updates."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        total_steps: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        device: torch.device,
        max_grad_norm: float = 1.0,
    ) -> None:
        params = [parameter for parameter in parameters if parameter.requires_grad]
        if not params:
            raise ValueError("Optimizer received no trainable parameters")
        if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
            raise ValueError("Invalid total/warmup step contract")
        self.parameters = params
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.completed_steps = 0
        self._prepared = False

    def _factor(self, step_number: int) -> float:
        if self.warmup_steps and step_number <= self.warmup_steps:
            return step_number / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        decay_position = step_number - self.warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * decay_position / decay_steps))

    def prepare(self) -> None:
        if self._prepared:
            raise RuntimeError("Optimizer step was prepared twice")
        if self.completed_steps >= self.total_steps:
            raise RuntimeError("Optimizer exceeded exact step budget")
        step_number = self.completed_steps + 1
        factor = self._factor(step_number)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * factor
        self.optimizer.zero_grad(set_to_none=True)
        self._prepared = True

    def backward(self, loss: Tensor) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before backward()")
        if not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite training loss")
        self.scaler.scale(loss).backward()

    def finish(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before finish()")
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradient norm")
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.amp_enabled and self.scaler.get_scale() < previous_scale:
            raise RuntimeError("AMP skipped an optimizer update; exact-step contract violated")
        self.completed_steps += 1
        self._prepared = False

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def assert_complete(self) -> None:
        if self._prepared:
            raise RuntimeError("Optimizer has unfinished accumulated gradients")
        if self.completed_steps != self.total_steps:
            raise RuntimeError(
                f"Completed {self.completed_steps} of {self.total_steps} optimizer steps"
            )
