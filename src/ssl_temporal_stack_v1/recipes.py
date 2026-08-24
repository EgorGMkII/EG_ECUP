"""Optional parameter recipes; legacy callers retain frozen defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerRecipe:
    learning_rate: float
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    scheduler: str = "cosine"
    steps: int = 0


@dataclass(frozen=True)
class LossWeights:
    factorized: float = 1.0
    direct_amount: float = 0.25
    conditional_amount: float = 0.25
    react: float = 0.10
    churn: float = 0.10


@dataclass(frozen=True)
class NeuralRecipe:
    ssl: OptimizerRecipe
    base: OptimizerRecipe
    specialist_head: OptimizerRecipe
    specialist_finetune: OptimizerRecipe
    encoder_dropout: float = 0.2
    head_dropout: float = 0.2
    transformer_dropout: float = 0.1
    loss_weights: LossWeights = LossWeights()
    synchronized_epochs: bool = False
