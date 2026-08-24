"""Independent exact-step adapters for TCN and Residual MLP candidates."""

from __future__ import annotations

from dataclasses import asdict
import gc
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from src.ssl_temporal_stack_v1.loaders import make_loader_factory
from src.ssl_temporal_stack_v1.optimization import StepOptimizer
from src.ssl_temporal_stack_v1.runtime import progress, seed_everything
from src.ssl_temporal_stack_v1.training import TrainingStats, round_robin_batches, train_exact_steps

from .base import ModelResult, RunContext
from .candidates.residual_mlp import ResidualMLPRecipe, ResidualMLPSpecialist, ResidualMLPTransitionBase, StreamingFeatureScaler
from .candidates.tcn import TCNRecipe, TCNSpecialist, TCNTransitionBase


def _autocast(device: torch.device):
    return torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def _to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")


def _progress(run: str, model: str, stage: str, task: str | None = None):
    return lambda step, total, anchor, loss: progress("TRAIN_PROGRESS", run=run, model=model, stage=stage, task=task, step=step, total=total, anchor=anchor, loss=loss)


def _fit_steps(*, model: torch.nn.Module, factories, steps: int, recipe: dict[str, Any], device: torch.device, run: str, model_id: str, stage: str, task: str | None, loss: Callable[[tuple[torch.Tensor, ...]], torch.Tensor], tickets: dict[str, int] | None) -> TrainingStats:
    stream = round_robin_batches(factories, tickets, synchronized_epochs="epoch_resolution" in recipe)
    control = StepOptimizer(model.parameters(), total_steps=steps, learning_rate=float(recipe["learning_rate"]), weight_decay=float(recipe.get("weight_decay", 1e-4)), warmup_steps=int(recipe.get("warmup_steps", 0)), scheduler=str(recipe.get("scheduler", "cosine")), device=device)
    model.train()
    progress("TRAIN_START", run=run, model=model_id, stage=stage, task=task, steps=steps)

    def train_step(batch: tuple[torch.Tensor, ...]) -> float:
        control.prepare()
        with _autocast(device):
            value = loss(batch)
        control.backward(value)
        control.finish()
        return float(value.detach().float().cpu())

    stats = train_exact_steps(batches=stream, requested_steps=steps, train_step=train_step, progress=_progress(run, model_id, stage, task))
    control.assert_complete()
    progress("TRAIN_DONE", run=run, model=model_id, stage=stage, task=task, **asdict(stats))
    return stats


def _dense_factories(
    context: RunContext,
    *,
    task: str | None,
    phase: str,
    batch_size: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    history_days: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for anchor in context.train_anchors:
        frame = context.stores.frames.get(anchor)
        if transform is None:
            values = context.stores.daily.get(anchor)
            if history_days is not None:
                if values.ndim != 3 or not 1 <= history_days <= values.shape[1]:
                    raise ValueError(f"Invalid causal history_days={history_days} for daily tensor {values.shape}")
                # The right edge is always the anchor.  This is a view into the
                # shared 180-day causal store, not a second feature store.
                values = values[:, -history_days:, :]
        else:
            values = transform(frame.select(context.stores.frames.feature_names).to_numpy())
        active = frame["was_active"].to_numpy().astype(np.float32, copy=False)
        buy = frame["will_buy"].to_numpy().astype(np.float32, copy=False)
        z = frame["z_target"].to_numpy().astype(np.float32, copy=False)
        if task == "react":
            indices, target = np.flatnonzero(active == 0), buy
        elif task == "churn":
            indices, target = np.flatnonzero(active == 1), 1.0 - buy
        elif task == "amount":
            indices, target = np.flatnonzero(z > 0), z
        else:
            indices, target = None, z
        arrays = (values, target) if task else (values, z, active, buy)
        result[anchor] = make_loader_factory(arrays=arrays, indices=indices, batch_size=batch_size, seed_parts=(context.run_name, "candidate", phase, task or "base", anchor))
    return result


@torch.no_grad()
def _predict(model: torch.nn.Module, values: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    factory = make_loader_factory(arrays=(values,), indices=None, batch_size=batch_size, seed_parts=("candidate", "inference"), shuffle=False)
    chunks: list[np.ndarray] = []
    model.eval()
    for batch, _ in factory(0):
        chunks.append(model(_to(batch[0], device)).float().cpu().numpy())
    result = np.concatenate(chunks).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("Candidate prediction is non-finite")
    return result


def fit_predict_tcn(context: RunContext, values: dict[str, Any]) -> ModelResult:
    if context.stores.daily is None:
        raise RuntimeError("TCN requires DailyTensorStore")
    history_days = int(values.get("history_days", 180))
    recipe = TCNRecipe(
        history_days=history_days,
        channels=int(values.get("channels", 128)),
        dropout=float(values.get("dropout", 0.10)),
    )
    batch_size = int(values["batch_size"])
    seed_everything(context.root_seed, context.run_name, "tcn", "base")
    base = TCNTransitionBase(recipe).to(context.device)
    factories = _dense_factories(context, task=None, phase="base", batch_size=batch_size, history_days=history_days)

    def base_loss(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        daily, _z, active, buy = (_to(value, context.device) for value in batch)
        out = base(daily)
        react_mask, churn_mask = active == 0, active == 1
        react = F.binary_cross_entropy_with_logits(out["reactivation_logit"][react_mask], buy[react_mask])
        churn = F.binary_cross_entropy_with_logits(out["churn_logit"][churn_mask], 1.0 - buy[churn_mask])
        return react + churn

    report: dict[str, Any] = {"base": asdict(_fit_steps(model=base, factories=factories, steps=int(values["base"]["steps"]), recipe=values["base"], device=context.device, run=context.run_name, model_id="tcn", stage="base", task=None, loss=base_loss, tickets=context.anchor_tickets))}
    base.to("cpu")
    torch.cuda.empty_cache()
    predictions: dict[str, np.ndarray] = {}
    report["specialists"] = {}
    for task in ("react", "churn"):
        seed_everything(context.root_seed, context.run_name, "tcn", task)
        model = TCNSpecialist(base, task, float(values.get("head_dropout", values.get("dropout", 0.10)))).to(context.device)
        report["specialists"][task] = {}
        for phase in ("H", "F"):
            getattr(model, "freeze_phase_h" if phase == "H" else "unfreeze_phase_f")()
            factories = _dense_factories(context, task=task, phase=phase, batch_size=batch_size, history_days=history_days)
            def specialist_loss(batch: tuple[torch.Tensor, ...], candidate=model) -> torch.Tensor:
                return F.binary_cross_entropy_with_logits(candidate(_to(batch[0], context.device)), _to(batch[1], context.device))
            report["specialists"][task][phase] = asdict(_fit_steps(model=model, factories=factories, steps=int(values["specialists"][task][phase]["steps"]), recipe=values["specialists"][task][phase], device=context.device, run=context.run_name, model_id="tcn", stage=phase, task=task, loss=specialist_loss, tickets=context.anchor_tickets))
        holdout_daily = context.stores.daily.get(context.holdout_anchor)[:, -history_days:, :]
        predictions[f"tcn_{task}_logit"] = _predict(model, holdout_daily, device=context.device, batch_size=batch_size)
        del model
        gc.collect(); torch.cuda.empty_cache()
    del base
    return ModelResult("tcn", predictions, {**report, "resolved_recipe": values, "history_source": "shared_daily_180d_right_aligned_view", "implementation_id": TCNTransitionBase.implementation_id})


def fit_predict_residual_mlp(context: RunContext, values: dict[str, Any]) -> ModelResult:
    recipe = ResidualMLPRecipe(input_features=len(context.stores.frames.feature_names), hidden=int(values.get("hidden", 512)), blocks=int(values.get("blocks", 4)), dropout=float(values.get("dropout", 0.10)))
    batch_size = int(values["batch_size"])
    scaler = StreamingFeatureScaler()
    for anchor in context.train_anchors:
        scaler.partial_fit(context.stores.frames.get(anchor).select(context.stores.frames.feature_names).to_numpy())
    transform = scaler.transform
    seed_everything(context.root_seed, context.run_name, "residual_mlp", "base")
    base = ResidualMLPTransitionBase(recipe).to(context.device)
    factories = _dense_factories(context, task=None, phase="base", batch_size=batch_size, transform=transform)

    def base_loss(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        features, z, active, buy = (_to(value, context.device) for value in batch)
        out = base(features)
        react_mask, churn_mask, amount_mask = active == 0, active == 1, z > 0
        return (F.binary_cross_entropy_with_logits(out["reactivation_logit"][react_mask], buy[react_mask]) + F.binary_cross_entropy_with_logits(out["churn_logit"][churn_mask], 1.0 - buy[churn_mask]) + F.mse_loss(out["amount_z"][amount_mask], z[amount_mask]))

    report: dict[str, Any] = {"scaler": scaler.report(), "base": asdict(_fit_steps(model=base, factories=factories, steps=int(values["base"]["steps"]), recipe=values["base"], device=context.device, run=context.run_name, model_id="residual_mlp", stage="base", task=None, loss=base_loss, tickets=context.anchor_tickets)), "specialists": {}}
    base.to("cpu")
    torch.cuda.empty_cache()
    predictions: dict[str, np.ndarray] = {}
    for task in ("react", "churn", "amount"):
        seed_everything(context.root_seed, context.run_name, "residual_mlp", task)
        model = ResidualMLPSpecialist(base, task, float(values.get("head_dropout", values.get("dropout", 0.10)))).to(context.device)
        report["specialists"][task] = {}
        for phase in ("H", "F"):
            getattr(model, "freeze_phase_h" if phase == "H" else "unfreeze_phase_f")()
            factories = _dense_factories(context, task=task, phase=phase, batch_size=batch_size, transform=transform)
            def specialist_loss(batch: tuple[torch.Tensor, ...], candidate=model, current_task=task) -> torch.Tensor:
                prediction, target = candidate(_to(batch[0], context.device)), _to(batch[1], context.device)
                return F.mse_loss(prediction, target) if current_task == "amount" else F.binary_cross_entropy_with_logits(prediction, target)
            report["specialists"][task][phase] = asdict(_fit_steps(model=model, factories=factories, steps=int(values["specialists"][task][phase]["steps"]), recipe=values["specialists"][task][phase], device=context.device, run=context.run_name, model_id="residual_mlp", stage=phase, task=task, loss=specialist_loss, tickets=context.anchor_tickets))
        frame = context.stores.frames.get(context.holdout_anchor)
        column = f"mlp_{task}_logit" if task != "amount" else "mlp_amount_z"
        predictions[column] = _predict(model, transform(frame.select(context.stores.frames.feature_names).to_numpy()), device=context.device, batch_size=batch_size)
        del model
        gc.collect(); torch.cuda.empty_cache()
    del base
    return ModelResult("residual_mlp", predictions, {**report, "resolved_recipe": values, "implementation_id": ResidualMLPTransitionBase.implementation_id})
