"""Independent config-driven RUN A -> RUN B orchestration."""

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

from src.ssl_temporal_stack_v1.diagnostics import build_validation_report
from src.ssl_temporal_stack_v1.runtime import gpu_info, require_cuda, sha256_file, write_json
from src.ssl_temporal_stack_v1.stores import build_store_registry_for_anchors

from .base import ModelResult, RunContext
from .config import ExperimentConfig, resolved_config
from .epochs import resolve_epoch_recipe
from .meta import apply_meta, apply_meta_components, fit_meta
from .predictions import PredictionSchema, bank_arrays, make_prediction_bank, schema_from_specs, validate_prediction_mapping
from .registry import build_adapters, collect_required_stores


RESULT_FILES = ("run_manifest.json", "resolved_config.yaml", "model_training_report.json", "run_a_meta_prediction_bank.parquet", "frozen_meta_package.json", "run_b_validation_prediction_bank.parquet", "validation_report.json", "artifact_sha256.json")


def _load_inputs(config: ExperimentConfig) -> tuple[pl.DataFrame, tuple[int, ...], dict[str, Any]]:
    cohort = pl.read_csv(config.cohort_path) if config.cohort_path.suffix.lower() == ".csv" else pl.read_parquet(config.cohort_path)
    cohort = cohort.select("user_id")
    if cohort.columns != ["user_id"] or cohort.height not in {25_000, 100_000, 250_000} or cohort["user_id"].n_unique() != cohort.height:
        raise ValueError("Selected cohort must contain unique user_id only")
    raw = pl.read_parquet(config.train_path)
    if raw["event_date"].dtype == pl.Utf8:
        raw = raw.with_columns(pl.col("event_date").str.to_date())
    return raw, tuple(cohort["user_id"].to_list()), {"train_sha256": sha256_file(config.train_path), "cohort_sha256": sha256_file(config.cohort_path), "cohort_rows": cohort.height}


def _merge(results: list[ModelResult], schema: PredictionSchema, rows: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for result in results:
        if result.model_id in report or set(values) & set(result.predictions):
            raise ValueError("Duplicate adapter result")
        values.update(result.predictions)
        report[result.model_id] = result.training_report
    validate_prediction_mapping(values, schema, rows)
    return values, report


def _fit(context: RunContext, adapters, model_configs, schema: PredictionSchema) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    context.stores.anchor_tickets = context.anchor_tickets
    resolved = {
        adapter.model_id: type(model_configs[adapter.model_id])(
            adapter.model_id,
            resolve_epoch_recipe(context, adapter.model_id, model_configs[adapter.model_id].values),
        )
        for adapter in adapters
    }
    results = [adapter.fit_predict(context, resolved[adapter.model_id]) for adapter in adapters]
    values, report = _merge(results, schema, len(context.users))
    gc.collect()
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return values, report


def _refuse_existing(root: Path) -> None:
    existing = [path for path in (root / name for name in RESULT_FILES) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite result artifacts: {existing}")
    root.mkdir(parents=True, exist_ok=True)


def run_validation(config: ExperimentConfig, *, pre_run_sha: str, job_id: str | None = None) -> dict[str, Any]:
    if config.stage not in {"screen", "full"}:
        raise ValueError("run_validation supports screen/full only")
    if len(pre_run_sha) < 7:
        raise ValueError("PRE-RUN SHA is required")
    _refuse_existing(config.output_root)
    device = require_cuda()
    (config.output_root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config(config), sort_keys=False), encoding="utf-8")
    raw, users, inputs = _load_inputs(config)
    adapters = build_adapters(config.enabled_models)
    model_configs = {}
    for adapter in adapters:
        values = dict(config.raw["models"][adapter.model_id])
        if adapter.model_id not in {"catboost", "catboost_direct"}:
            values["loss_weights"] = config.raw.get("loss_weights", {})
        model_configs[adapter.model_id] = adapter.validate_config(values)
    schema = schema_from_specs([adapter.prediction_spec for adapter in adapters])
    anchors = tuple(sorted(set((*config.profile.run_a_anchors, config.profile.meta_anchor, *config.profile.run_b_anchors, config.profile.validation_anchor))))
    stores = build_store_registry_for_anchors(raw, list(users), store_anchors=anchors, training_anchors=config.profile.run_b_anchors, root=config.output_root / "_work" / "stores", cohort_sha256=inputs["cohort_sha256"], required_stores=collect_required_stores(adapters))
    started = time.perf_counter()
    manifest: dict[str, Any] = {"experiment_id": config.experiment_id, "profile": config.profile.name, "stage": config.stage, "config_sha256": config.sha256, "pre_run_commit_sha": pre_run_sha, "job_id": job_id, "inputs": inputs, "gpu": gpu_info(), "enabled_models": list(config.enabled_models), "prediction_schema": {"react": list(schema.react_columns), "churn": list(schema.churn_columns), "amount": list(schema.amount_columns), "direct": list(schema.direct_columns)}, "required_stores": sorted(collect_required_stores(adapters))}
    write_json(config.output_root / "run_manifest.json", manifest)
    try:
        sampling = config.raw.get("anchor_sampling", {"mode": "uniform"})
        def tickets(anchors: tuple[str, ...]) -> dict[str, int] | None:
            return None if sampling["mode"] == "uniform" else {anchor: int(sampling["tickets"].get(anchor, 1)) for anchor in anchors}
        a_context = RunContext("RUN_A", config.profile.run_a_anchors, config.profile.meta_anchor, users, stores, device, config.root_seed, config.output_root, tickets(config.profile.run_a_anchors), raw)
        a_values, a_report = _fit(a_context, adapters, model_configs, schema)
        a_frame = stores.frames.get(config.profile.meta_anchor)
        a_bank = make_prediction_bank(a_frame, config.profile.meta_anchor, a_values, schema)
        a_path = config.output_root / "run_a_meta_prediction_bank.parquet"
        a_bank.write_parquet(a_path)
        package = fit_meta(bank_arrays(a_bank, schema), schema, root_seed=config.root_seed, prediction_bank_sha256=sha256_file(a_path), commit_sha=pre_run_sha, config_sha256=config.sha256)
        meta_path = config.output_root / "frozen_meta_package.json"
        write_json(meta_path, package)
        del a_values, a_bank, a_frame
        gc.collect(); torch.cuda.empty_cache()
        b_context = RunContext("RUN_B", config.profile.run_b_anchors, config.profile.validation_anchor, users, stores, device, config.root_seed, config.output_root, tickets(config.profile.run_b_anchors), raw)
        b_values, b_report = _fit(b_context, adapters, model_configs, schema)
        b_frame = stores.frames.get(config.profile.validation_anchor)
        b_bank = make_prediction_bank(b_frame, config.profile.validation_anchor, b_values, schema)
        b_path = config.output_root / "run_b_validation_prediction_bank.parquet"
        b_bank.write_parquet(b_path)
        components = apply_meta_components(package, bank_arrays(b_bank, schema), schema)
        report = build_validation_report(target_z=b_bank["z_target"].to_numpy(), was_active=b_bank["was_active"].to_numpy(), will_buy=b_bank["will_buy"].to_numpy(), components=components, validation_anchor=config.profile.validation_anchor, job_id=job_id, commit_sha=pre_run_sha, config_sha256=config.sha256, bank_sha256=sha256_file(b_path), meta_sha256=sha256_file(meta_path))
        write_json(config.output_root / "validation_report.json", report)
        write_json(config.output_root / "model_training_report.json", {"RUN_A": a_report, "RUN_B": b_report})
        manifest.update({"status": "COMPLETED", "elapsed_seconds": time.perf_counter() - started, "run_a_meta_rmsle": package["rmsle"], "run_b_validation_rmsle": report["rmsle"]})
        write_json(config.output_root / "run_manifest.json", manifest)
        write_json(config.output_root / "artifact_sha256.json", {name: sha256_file(config.output_root / name) for name in RESULT_FILES if name != "artifact_sha256.json"})
        return report
    except Exception as error:
        manifest.update({"status": "FAILED", "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.perf_counter() - started})
        write_json(config.output_root / "run_manifest.json", manifest)
        raise
    finally:
        stores.close()


def run_final(config: ExperimentConfig, *, pre_run_sha: str, job_id: str | None = None) -> Path:
    """Train fresh 250k first-level models and emit a template-aligned submission."""
    if config.stage != "final":
        raise ValueError("run_final requires stage=final")
    _refuse_existing(config.output_root)
    final = config.raw["final"]
    meta_path = Path(final["frozen_meta_path"])
    if sha256_file(meta_path) != final["frozen_meta_sha256"]:
        raise RuntimeError("Pinned frozen meta hash mismatch")
    package = json.loads(meta_path.read_text(encoding="utf-8"))
    device = require_cuda()
    raw, users, _ = _load_inputs(config)
    adapters = build_adapters(config.enabled_models)
    model_configs = {}
    for adapter in adapters:
        values = dict(config.raw["models"][adapter.model_id])
        if adapter.model_id not in {"catboost", "catboost_direct"}:
            values["loss_weights"] = config.raw.get("loss_weights", {})
        model_configs[adapter.model_id] = adapter.validate_config(values)
    schema = schema_from_specs([adapter.prediction_spec for adapter in adapters])
    anchors = tuple(sorted(set((*config.profile.final_train_anchors, config.profile.final_inference_anchor))))
    stores = build_store_registry_for_anchors(raw, list(users), store_anchors=anchors, training_anchors=config.profile.final_train_anchors, root=config.output_root / "_work" / "stores", cohort_sha256=sha256_file(config.cohort_path), required_stores=collect_required_stores(adapters))
    try:
        context = RunContext("FINAL", config.profile.final_train_anchors, config.profile.final_inference_anchor, users, stores, device, config.root_seed, config.output_root, None, raw)
        values, report = _fit(context, adapters, model_configs, schema)
        frame = stores.frames.get(config.profile.final_inference_anchor)
        bank = make_prediction_bank(frame, config.profile.final_inference_anchor, values, schema)
        prediction = np.expm1(apply_meta(package, bank_arrays(bank, schema), schema))
        template = pl.read_csv(config.sample_submit_path).select("user_id")
        submission = template.with_columns(pl.Series("predict", prediction))
        if submission.height != 250_000 or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise RuntimeError("Invalid final submission")
        result = config.output_root / "submission.csv"
        submission.write_csv(result)
        write_json(config.output_root / "model_training_report.json", {"FINAL": report, "pre_run_commit_sha": pre_run_sha, "job_id": job_id})
        return result
    finally:
        stores.close()
