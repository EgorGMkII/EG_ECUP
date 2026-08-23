"""First-level fit/predict adapters for the isolated SSL stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl
import torch

from .config import LoadedConfig
from .contract import EXPERIMENT, NeuralBudget
from .loaders import make_loader_factory
from .models import EventTimeTransformer, Specialist, TransitionBase
from .predictions import AMOUNT_COLUMNS, CHURN_COLUMNS, REACT_COLUMNS
from .runtime import derive_seed, progress
from .recipes import NeuralRecipe, OptimizerRecipe
from .stores import StoreRegistry
from .trainers import (
    fit_ett_base,
    fit_ett_specialist,
    fit_gru_base,
    fit_gru_pretrainer,
    fit_gru_specialist,
)


@dataclass
class AdapterResult:
    model_id: str
    predictions: dict[str, np.ndarray]
    training_report: dict[str, Any]


def resolved_catboost_params(
    config: LoadedConfig,
    *,
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = config.raw["models"]["catboost"]
    result = {
        "iterations": int(raw["iterations"]),
        "learning_rate": float(raw["learning_rate"]),
        "depth": int(raw["depth"]),
        "l2_leaf_reg": float(raw["l2_leaf_reg"]),
        "random_seed": int(seed),
        "task_type": raw["task_type"],
        "devices": str(raw["devices"]),
        "boosting_type": raw["boosting_type"],
        "grow_policy": raw["grow_policy"],
        "bootstrap_type": raw["bootstrap_type"],
        "bagging_temperature": float(raw["bagging_temperature"]),
        "random_strength": float(raw["random_strength"]),
        "border_count": int(raw["border_count"]),
        "nan_mode": raw["nan_mode"],
        "allow_writing_files": False,
        "verbose": 250,
    }
    if overrides:
        result.update(overrides)
    if result["task_type"] == "CPU":
        result.pop("devices", None)
    return result


def fit_predict_catboost(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    holdout_anchor: str,
    *,
    run: str,
    config: LoadedConfig,
    parameter_overrides: dict[str, Any] | None = None,
) -> AdapterResult:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool, __version__ as catboost_version

    features = stores.frames.feature_names
    train_frames = [stores.frames.get(anchor) for anchor in anchors]
    pooled = pl.concat(train_frames)
    holdout = stores.frames.get(holdout_anchor)
    holdout_pool = Pool(holdout.select(features).to_pandas())
    predictions: dict[str, np.ndarray] = {}
    tasks: dict[str, Any] = {}
    definitions = (
        ("react", pooled["was_active"].to_numpy() == 0, pooled["will_buy"].to_numpy(), CatBoostClassifier, "Logloss"),
        ("churn", pooled["was_active"].to_numpy() == 1, 1 - pooled["will_buy"].to_numpy(), CatBoostClassifier, "Logloss"),
        ("amount", pooled["future_gmv_30d"].to_numpy() > 0, pooled["z_target"].to_numpy(), CatBoostRegressor, "RMSE"),
    )
    for task, mask, full_target, model_class, loss_function in definitions:
        started = time.perf_counter()
        subset = pooled.filter(pl.Series(mask))
        target = np.asarray(full_target)[mask]
        seed = derive_seed(EXPERIMENT.root_seed, run, "catboost", task)
        params = resolved_catboost_params(config, seed=seed, overrides=parameter_overrides)
        params["loss_function"] = loss_function
        progress("CATBOOST_START", run=run, task=task, rows=subset.height, params=params)
        train_pool = Pool(subset.select(features).to_pandas(), label=target)
        model = model_class(**params)
        model.fit(train_pool)
        if task == "amount":
            values = np.clip(model.predict(holdout_pool), 0.0, None)
            column = "cb_amount_z"
        else:
            values = model.predict(holdout_pool, prediction_type="RawFormulaVal")
            column = f"cb_{task}_logit"
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (holdout.height,) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid CatBoost {task} predictions")
        predictions[column] = values
        tasks[task] = {
            "rows": subset.height,
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - started,
            "tree_count": int(model.tree_count_),
            "resolved_parameters": model.get_all_params(),
        }
        progress("CATBOOST_DONE", run=run, task=task, **tasks[task])
        del subset, target, train_pool, model
        gc.collect()
    del train_frames, pooled, holdout_pool
    gc.collect()
    return AdapterResult(
        model_id="catboost",
        predictions=predictions,
        training_report={"catboost_version": catboost_version, "tasks": tasks},
    )


@torch.no_grad()
def predict_dense_specialist(
    model: Specialist,
    stores: StoreRegistry,
    holdout_anchor: str,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    values: list[np.ndarray] = []
    factory = make_loader_factory(
        arrays=(stores.daily.get(holdout_anchor),), indices=None, batch_size=batch_size,
        seed_parts=("inference", model.kind, model.task, holdout_anchor),
        shuffle=False,
    )
    model.eval()
    for batch, _ in factory(0):
        prediction = model(batch[0].to(device, dtype=torch.float32, non_blocking=device.type == "cuda"))
        values.append(prediction.float().cpu().numpy())
    result = np.concatenate(values).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError(f"Non-finite {model.kind}/{model.task} inference")
    return result


@torch.no_grad()
def predict_ett_specialist(
    model: Specialist,
    stores: StoreRegistry,
    holdout_anchor: str,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    values: list[np.ndarray] = []
    factory = make_loader_factory(
        arrays=stores.events.get(holdout_anchor), indices=None, batch_size=batch_size,
        seed_parts=("inference", "ett", model.task, holdout_anchor), shuffle=False,
    )
    model.eval()
    for batch, _ in factory(0):
        inputs = (
            batch[0].to(device, dtype=torch.float32, non_blocking=device.type == "cuda"),
            batch[1].to(device, dtype=torch.float32, non_blocking=device.type == "cuda"),
            batch[2].to(device, dtype=torch.long, non_blocking=device.type == "cuda"),
            batch[3].to(device, dtype=torch.bool, non_blocking=device.type == "cuda"),
            batch[4].to(device, dtype=torch.bool, non_blocking=device.type == "cuda"),
        )
        values.append(model(*inputs).float().cpu().numpy())
    result = np.concatenate(values).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError(f"Non-finite ETT/{model.task} inference")
    return result


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, path)


def fit_predict_gru(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    holdout_anchor: str,
    *,
    run: str,
    model_id: str,
    device: torch.device,
    checkpoint_root: Path,
    budget: NeuralBudget | None = None,
    recipe: NeuralRecipe | None = None,
    specialist_recipes: dict[str, tuple[OptimizerRecipe, OptimizerRecipe]] | None = None,
) -> AdapterResult:
    budget = budget or EXPERIMENT.budgets[model_id]
    pretrainer, ssl_stats = fit_gru_pretrainer(
        stores, anchors, run=run, model_id=model_id, device=device, budget=budget, recipe=recipe
    )
    base_fit = fit_gru_base(
        pretrainer, stores, anchors, run=run, model_id=model_id, device=device, budget=budget, recipe=recipe
    )
    base = base_fit.model
    assert isinstance(base, TransitionBase)
    _save_checkpoint(
        checkpoint_root / f"{run}_{model_id}_base.pt", base,
        {"run": run, "model": model_id, "stage": "base", "stats": asdict(base_fit.stats)},
    )
    del pretrainer
    base.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    predictions: dict[str, np.ndarray] = {}
    specialist_report: dict[str, Any] = {}
    for task in ("react", "churn", "amount"):
        task_pair = specialist_recipes.get(task) if specialist_recipes else None
        task_budget = replace(
            budget,
            specialist_head_steps=task_pair[0].steps if task_pair else budget.specialist_head_steps,
            specialist_finetune_steps=task_pair[1].steps if task_pair else budget.specialist_finetune_steps,
        ) if task_pair else budget
        fit = fit_gru_specialist(
            base, stores, anchors, run=run, model_id=model_id, task=task,
            device=device, budget=task_budget, recipe=recipe,
            specialist_recipes=specialist_recipes,
        )
        values = predict_dense_specialist(fit.model, stores, holdout_anchor, device=device)
        column = f"{model_id}_{task}_logit" if task != "amount" else f"{model_id}_amount_z"
        predictions[column] = values
        specialist_report[task] = {phase: asdict(stats) for phase, stats in fit.stats.items()}
        _save_checkpoint(
            checkpoint_root / f"{run}_{model_id}_{task}.pt", fit.model,
            {"run": run, "model": model_id, "task": task, "stats": specialist_report[task]},
        )
        del fit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del base
    return AdapterResult(
        model_id=model_id,
        predictions=predictions,
        training_report={
            "budget": asdict(budget), "ssl": asdict(ssl_stats),
            "base": asdict(base_fit.stats), "specialists": specialist_report,
        },
    )


def fit_predict_ett(
    stores: StoreRegistry,
    anchors: tuple[str, ...],
    holdout_anchor: str,
    *,
    run: str,
    device: torch.device,
    checkpoint_root: Path,
    budget: NeuralBudget | None = None,
    micro_batch_size: int = 128,
    accumulation_steps: int = 4,
    recipe: NeuralRecipe | None = None,
    specialist_recipes: dict[str, tuple[OptimizerRecipe, OptimizerRecipe]] | None = None,
) -> AdapterResult:
    budget = budget or EXPERIMENT.budgets["ett"]
    base_fit = fit_ett_base(
        stores, anchors, run=run, device=device, budget=budget,
        micro_batch_size=micro_batch_size, accumulation_steps=accumulation_steps, recipe=recipe,
    )
    base = base_fit.model
    assert isinstance(base, EventTimeTransformer)
    _save_checkpoint(
        checkpoint_root / f"{run}_ett_base.pt", base,
        {"run": run, "model": "ett", "stage": "base", "stats": asdict(base_fit.stats)},
    )
    base.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    predictions: dict[str, np.ndarray] = {}
    specialist_report: dict[str, Any] = {}
    for task in ("react", "churn", "amount"):
        task_pair = specialist_recipes.get(task) if specialist_recipes else None
        task_budget = replace(
            budget,
            specialist_head_steps=task_pair[0].steps if task_pair else budget.specialist_head_steps,
            specialist_finetune_steps=task_pair[1].steps if task_pair else budget.specialist_finetune_steps,
        ) if task_pair else budget
        fit = fit_ett_specialist(
            base, stores, anchors, run=run, task=task, device=device, budget=task_budget,
            micro_batch_size=micro_batch_size, accumulation_steps=accumulation_steps, recipe=recipe,
            specialist_recipes=specialist_recipes,
        )
        values = predict_ett_specialist(fit.model, stores, holdout_anchor, device=device)
        column = f"ett_{task}_logit" if task != "amount" else "ett_amount_z"
        predictions[column] = values
        specialist_report[task] = {phase: asdict(stats) for phase, stats in fit.stats.items()}
        _save_checkpoint(
            checkpoint_root / f"{run}_ett_{task}.pt", fit.model,
            {"run": run, "model": "ett", "task": task, "stats": specialist_report[task]},
        )
        del fit
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del base
    return AdapterResult(
        model_id="ett",
        predictions=predictions,
        training_report={
            "budget": asdict(budget), "base": asdict(base_fit.stats),
            "specialists": specialist_report,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
        },
    )


def expected_columns_for_adapter(model_id: str) -> tuple[str, str, str]:
    aliases = {"catboost": "cb", "s1": "s1", "s2": "s2", "ett": "ett"}
    if model_id not in aliases:
        raise KeyError(model_id)
    prefix = aliases[model_id]
    return (
        REACT_COLUMNS[("cb", "s1", "s2", "ett").index(prefix)],
        CHURN_COLUMNS[("cb", "s1", "s2", "ett").index(prefix)],
        AMOUNT_COLUMNS[("cb", "s1", "s2", "ett").index(prefix)],
    )
