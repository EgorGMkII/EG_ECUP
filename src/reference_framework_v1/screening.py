"""Independent model screening and incremental ensemble gates."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from scipy.special import expit

from src.ssl_temporal_stack_v1.diagnostics import _binary_report, build_validation_report
from src.ssl_temporal_stack_v1.runtime import gpu_info, require_cuda, sha256_file, write_json
from src.ssl_temporal_stack_v1.stores import build_store_registry_for_anchors

from .base import ModelConfig, ModelResult, RunContext
from .config import ExperimentConfig, resolved_config
from .epochs import resolve_epoch_recipe
from .meta import apply_meta_components, fit_meta
from .pipeline import _load_inputs
from .predictions import PredictionSchema, bank_arrays, make_prediction_bank, schema_from_specs
from .registry import build_adapters, collect_required_stores


INDIVIDUAL_FILES = ("screen_manifest.json", "resolved_config.yaml", "run_a_target_prediction_bank.parquet", "run_b_target_prediction_bank.parquet", "individual_report.json", "model_training_report.json", "artifact_sha256.json")
GATE_FILES = ("gate_manifest.json", "run_a_incremental_prediction_bank.parquet", "frozen_incremental_meta_package.json", "run_b_incremental_prediction_bank.parquet", "incremental_validation_report.json", "artifact_sha256.json")


def _refuse(root: Path, names: tuple[str, ...]) -> None:
    if any((root / name).exists() for name in names):
        raise FileExistsError(f"Refusing to overwrite screening outputs below {root}")
    root.mkdir(parents=True, exist_ok=True)


def _target_bank(frame: pl.DataFrame, anchor: str, predictions: dict[str, np.ndarray]) -> pl.DataFrame:
    base = frame.select("user_id", "was_active", "will_buy", "future_gmv_30d", "z_target").with_columns(pl.lit(anchor).alias("anchor"))
    if not predictions:
        raise ValueError("Screen adapter returned no predictions")
    columns: list[pl.Series] = []
    for name, values in predictions.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (base.height,) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid individual prediction column: {name}")
        columns.append(pl.Series(name, values))
    return base.with_columns(columns)


def _individual_metrics(bank: pl.DataFrame, result: ModelResult) -> dict[str, Any]:
    active = bank["was_active"].to_numpy().astype(np.int8)
    buy = bank["will_buy"].to_numpy().astype(np.int8)
    z = bank["z_target"].to_numpy().astype(np.float64)
    values = result.predictions
    report: dict[str, Any] = {"model_id": result.model_id, "rows": bank.height}
    if "cb_direct_z" in values:
        prediction = values["cb_direct_z"]
        report["direct"] = {"mse_logspace": float(np.square(prediction - z).mean()), "rmsle": float(np.sqrt(np.square(prediction - z).mean()))}
        return report
    react_name = next((name for name in values if name.endswith("_react_logit")), None)
    churn_name = next((name for name in values if name.endswith("_churn_logit")), None)
    amount_name = next((name for name in values if name.endswith("_amount_z")), None)
    if react_name:
        report["react"] = _binary_report(buy[active == 0], expit(values[react_name][active == 0]))
    if churn_name:
        report["churn"] = _binary_report(1 - buy[active == 1], expit(values[churn_name][active == 1]))
    if amount_name:
        positive = buy == 1
        error = np.square(values[amount_name][positive] - z[positive])
        report["amount"] = {"rows": int(positive.sum()), "mse_z": float(error.mean()) if positive.any() else None, "rmse_z": float(np.sqrt(error.mean())) if positive.any() else None}
    if react_name and churn_name and amount_name:
        p_buy = np.where(active == 0, expit(values[react_name]), 1.0 - expit(values[churn_name]))
        prediction = np.power(p_buy, 1.1) * np.clip(values[amount_name], 0.0, None)
        error = np.square(prediction - z)
        report["standalone_hurdle"] = {"mse_logspace": float(error.mean()), "rmsle": float(np.sqrt(error.mean()))}
    return report


def _target_config(config: ExperimentConfig, target_model: str) -> tuple[Any, ModelConfig]:
    adapter = build_adapters((target_model,))[0]
    values = dict(config.raw["models"][target_model])
    if target_model not in {"catboost", "catboost_direct"}:
        values["loss_weights"] = config.raw.get("loss_weights", {})
    return adapter, adapter.validate_config(values)


def run_individual_screen(config: ExperimentConfig, *, output_root: Path, pre_run_sha: str, job_id: str | None = None) -> dict[str, Any]:
    """Train only screening.target_model in independent RUN A and RUN B."""
    if config.stage not in {"screen", "full"} or not config.raw.get("screening"):
        raise ValueError("Individual screen requires a screen/full config with screening.target_model")
    _refuse(output_root, INDIVIDUAL_FILES)
    target_model = str(config.raw["screening"]["target_model"])
    device = require_cuda()
    raw, users, inputs = _load_inputs(config)
    adapter, model_config = _target_config(config, target_model)
    anchors = tuple(sorted(set((*config.profile.run_a_anchors, config.profile.meta_anchor, *config.profile.run_b_anchors, config.profile.validation_anchor))))
    stores = build_store_registry_for_anchors(raw, list(users), store_anchors=anchors, training_anchors=config.profile.run_b_anchors, root=output_root / "_work" / "stores", cohort_sha256=inputs["cohort_sha256"], required_stores=collect_required_stores([adapter]))
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config(config), sort_keys=False), encoding="utf-8")
    manifest: dict[str, Any] = {"status": "EXECUTING", "mode": "individual", "experiment_id": config.experiment_id, "target_model": target_model, "config_sha256": config.sha256, "pre_run_commit_sha": pre_run_sha, "job_id": job_id, "inputs": inputs, "gpu": gpu_info()}
    write_json(output_root / "screen_manifest.json", manifest)
    started = time.perf_counter()
    reports: dict[str, Any] = {}
    try:
        for run, train_anchors, holdout, file_name in (("RUN_A", config.profile.run_a_anchors, config.profile.meta_anchor, "run_a_target_prediction_bank.parquet"), ("RUN_B", config.profile.run_b_anchors, config.profile.validation_anchor, "run_b_target_prediction_bank.parquet")):
            context = RunContext(run, train_anchors, holdout, users, stores, device, config.root_seed, output_root, None, raw)
            resolved = ModelConfig(target_model, resolve_epoch_recipe(context, target_model, model_config.values))
            result = adapter.fit_predict(context, resolved)
            bank = _target_bank(stores.frames.get(holdout), holdout, result.predictions)
            bank.write_parquet(output_root / file_name)
            reports[run] = {"training": result.training_report, "metrics": _individual_metrics(bank, result), "prediction_columns": sorted(result.predictions)}
            del bank, result
            gc.collect(); torch.cuda.empty_cache()
        write_json(output_root / "model_training_report.json", reports)
        write_json(output_root / "individual_report.json", {"run_a_meta": reports["RUN_A"]["metrics"], "run_b_validation": reports["RUN_B"]["metrics"]})
        manifest.update({"status": "COMPLETED", "elapsed_seconds": time.perf_counter() - started})
        write_json(output_root / "screen_manifest.json", manifest)
        write_json(output_root / "artifact_sha256.json", {name: sha256_file(output_root / name) for name in INDIVIDUAL_FILES if name != "artifact_sha256.json"})
        return reports["RUN_B"]["metrics"]
    except Exception as error:
        manifest.update({"status": "FAILED", "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.perf_counter() - started})
        write_json(output_root / "screen_manifest.json", manifest)
        raise
    finally:
        stores.close()


def _replace(base: pl.DataFrame, candidate: pl.DataFrame, columns: tuple[str, ...], anchor: str) -> pl.DataFrame:
    """Replace an existing model channel, or append a new independent one.

    The latter is intentionally limited to a complete new channel.  It is how
    ``catboost_direct`` is evaluated: the immutable hurdle bank remains
    untouched and the fresh direct column is added only for a newly fit
    late-blend meta package.
    """
    required = ["user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "z_target"]
    if base.height != candidate.height or any(not base[name].equals(candidate[name]) for name in required):
        raise ValueError(f"Baseline and candidate banks are not aligned at {anchor}")
    if set(columns) != (set(candidate.columns) - set(required)):
        raise ValueError("Candidate bank has unexpected prediction columns")
    present = tuple(column for column in columns if column in base.columns)
    if present and len(present) != len(columns):
        raise ValueError("Cannot partially replace an existing prediction channel")
    if present:
        return base.drop(list(columns)).with_columns([candidate[name] for name in columns])
    return base.with_columns([candidate[name] for name in columns])


def run_incremental_gate(config: ExperimentConfig, *, baseline_root: Path, candidate_root: Path, output_root: Path, pre_run_sha: str, job_id: str | None = None) -> dict[str, Any]:
    """Refit meta after replacing one screened model's columns in immutable banks."""
    _refuse(output_root, GATE_FILES)
    target_model = str(config.raw.get("screening", {}).get("target_model", ""))
    if not target_model:
        raise ValueError("Incremental gate requires screening.target_model")
    adapter = build_adapters((target_model,))[0]
    baseline_manifest_path = baseline_root / "run_manifest.json"
    if not baseline_manifest_path.is_file():
        raise FileNotFoundError(f"Missing immutable baseline manifest: {baseline_manifest_path}")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    raw_schema = baseline_manifest.get("prediction_schema")
    if not isinstance(raw_schema, dict):
        raise ValueError("Baseline manifest has no prediction schema")
    try:
        schema = PredictionSchema(
            react_columns=tuple(raw_schema["react"]),
            churn_columns=tuple(raw_schema["churn"]),
            amount_columns=tuple(raw_schema["amount"]),
            direct_columns=tuple(raw_schema.get("direct", ())),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid baseline prediction schema") from error
    columns = tuple(column for column in (adapter.prediction_spec.react_column, adapter.prediction_spec.churn_column, adapter.prediction_spec.amount_column, adapter.prediction_spec.direct_column) if column)
    if not columns:
        raise ValueError(f"Screening adapter {target_model} declares no prediction columns")
    missing_from_baseline = tuple(column for column in columns if column not in schema.all_columns)
    if missing_from_baseline:
        if target_model != "catboost_direct" or missing_from_baseline != columns:
            raise ValueError("Only the complete independent direct channel may be appended to an immutable baseline bank")
        schema = PredictionSchema(
            react_columns=schema.react_columns,
            churn_columns=schema.churn_columns,
            amount_columns=schema.amount_columns,
            direct_columns=(*schema.direct_columns, *columns),
        )
    paths = {
        "base_a": baseline_root / "run_a_meta_prediction_bank.parquet", "base_b": baseline_root / "run_b_validation_prediction_bank.parquet",
        "candidate_a": candidate_root / "run_a_target_prediction_bank.parquet", "candidate_b": candidate_root / "run_b_target_prediction_bank.parquet",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError(f"Missing immutable prediction banks: {missing}")
    base_a, base_b = pl.read_parquet(paths["base_a"]), pl.read_parquet(paths["base_b"])
    candidate_a, candidate_b = pl.read_parquet(paths["candidate_a"]), pl.read_parquet(paths["candidate_b"])
    a_bank = _replace(base_a, candidate_a, columns, config.profile.meta_anchor)
    b_bank = _replace(base_b, candidate_b, columns, config.profile.validation_anchor)
    if set(schema.all_columns) != (set(a_bank.columns) - {"user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "z_target"}):
        raise ValueError("Composite bank does not match full enabled-model schema")
    a_path, b_path = output_root / "run_a_incremental_prediction_bank.parquet", output_root / "run_b_incremental_prediction_bank.parquet"
    a_bank.write_parquet(a_path); b_bank.write_parquet(b_path)
    package = fit_meta(bank_arrays(a_bank, schema), schema, root_seed=config.root_seed, prediction_bank_sha256=sha256_file(a_path), commit_sha=pre_run_sha, config_sha256=config.sha256)
    meta_path = output_root / "frozen_incremental_meta_package.json"
    write_json(meta_path, package)
    components = apply_meta_components(package, bank_arrays(b_bank, schema), schema)
    report = build_validation_report(target_z=b_bank["z_target"].to_numpy(), was_active=b_bank["was_active"].to_numpy(), will_buy=b_bank["will_buy"].to_numpy(), components=components, validation_anchor=config.profile.validation_anchor, job_id=job_id, commit_sha=pre_run_sha, config_sha256=config.sha256, bank_sha256=sha256_file(b_path), meta_sha256=sha256_file(meta_path))
    write_json(output_root / "incremental_validation_report.json", report)
    manifest = {"status": "COMPLETED", "mode": "incremental_gate", "experiment_id": config.experiment_id, "target_model": target_model, "baseline_root": str(baseline_root), "candidate_root": str(candidate_root), "baseline_hashes": {key: sha256_file(path) for key, path in paths.items()}, "config_sha256": config.sha256, "pre_run_commit_sha": pre_run_sha, "job_id": job_id, "rmsle": report["rmsle"]}
    write_json(output_root / "gate_manifest.json", manifest)
    write_json(output_root / "artifact_sha256.json", {name: sha256_file(output_root / name) for name in GATE_FILES if name != "artifact_sha256.json"})
    return report
