"""RUN A/RUN B orchestration for SSL_TEMPORAL_STACK_V1."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import polars as pl
import torch

from .adapters import AdapterResult, fit_predict_catboost, fit_predict_ett, fit_predict_gru
from .config import LoadedConfig, resolved_contract
from .contract import EXPERIMENT
from .diagnostics import build_validation_report
from .meta import apply_meta_components, fit_meta
from .predictions import PREDICTION_COLUMNS, bank_arrays, make_prediction_bank, validate_prediction_mapping
from .runtime import gpu_info, progress, require_cuda, sha256_file, write_json
from .stores import build_store_registry


RESULT_FILENAMES = (
    "run_manifest.json",
    "model_training_report.json",
    "run_a_meta_prediction_bank.parquet",
    "frozen_meta_package.json",
    "run_b_validation_prediction_bank.parquet",
    "validation_report.json",
    "artifact_sha256.json",
)


def merge_adapter_results(
    results: Iterable[AdapterResult],
    *,
    expected_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for result in results:
        if result.model_id in reports:
            raise ValueError(f"Duplicate adapter result: {result.model_id}")
        overlap = set(predictions) & set(result.predictions)
        if overlap:
            raise ValueError(f"Duplicate prediction columns: {sorted(overlap)}")
        predictions.update(result.predictions)
        reports[result.model_id] = result.training_report
    validate_prediction_mapping(predictions, expected_rows=expected_rows)
    return predictions, reports


def _ensure_new_result_scope(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [name for name in RESULT_FILENAMES if (output_root / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing SSL result artifacts: {existing}"
        )


def _load_and_validate_inputs(train_path: Path, cohort_path: Path) -> tuple[pl.DataFrame, list[int], dict[str, Any]]:
    if not train_path.is_file() or not cohort_path.is_file():
        raise FileNotFoundError(f"Missing train/cohort input: {train_path} / {cohort_path}")
    cohort_sha = sha256_file(cohort_path)
    if cohort_sha != EXPERIMENT.cohort_sha256:
        raise RuntimeError(f"Cohort SHA256 mismatch: {cohort_sha}")
    cohort = pl.read_parquet(cohort_path)
    if cohort.columns != ["user_id"] or cohort.height != 100_000 or cohort["user_id"].n_unique() != 100_000:
        raise RuntimeError("Cohort must contain exactly 100000 unique ordered user_id values")
    users = cohort["user_id"].to_list()
    train_sha = sha256_file(train_path)
    raw = pl.read_parquet(train_path)
    if "event_date" not in raw.columns or "user_id" not in raw.columns or "gmv" not in raw.columns:
        raise RuntimeError("Raw train schema is missing user_id/event_date/gmv")
    if raw["event_date"].dtype == pl.Utf8:
        raw = raw.with_columns(pl.col("event_date").str.to_date())
    return raw, users, {
        "train_path": train_path.as_posix(), "train_sha256": train_sha,
        "cohort_path": cohort_path.as_posix(), "cohort_sha256": cohort_sha,
        "cohort_rows": len(users),
    }


def _fit_first_level(
    stores,
    anchors: tuple[str, ...],
    holdout_anchor: str,
    *,
    run: str,
    config: LoadedConfig,
    device: torch.device,
    checkpoint_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    progress("FIRST_LEVEL_START", run=run, anchors=list(anchors), holdout_anchor=holdout_anchor)
    results: list[AdapterResult] = []
    results.append(fit_predict_catboost(stores, anchors, holdout_anchor, run=run, config=config))
    for model_id in ("s1", "s2"):
        results.append(
            fit_predict_gru(
                stores, anchors, holdout_anchor, run=run, model_id=model_id,
                device=device, checkpoint_root=checkpoint_root,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    results.append(
        fit_predict_ett(
            stores, anchors, holdout_anchor, run=run, device=device,
            checkpoint_root=checkpoint_root,
        )
    )
    predictions, reports = merge_adapter_results(
        results, expected_rows=stores.frames.users
    )
    progress("FIRST_LEVEL_DONE", run=run, columns=list(predictions))
    return predictions, reports


def run_pipeline(
    config: LoadedConfig,
    *,
    train_path: Path,
    cohort_path: Path,
    pre_run_sha: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full sealed two-run validation. CUDA is mandatory."""

    if len(pre_run_sha) < 7:
        raise ValueError("A PRE-RUN commit SHA is mandatory")
    device = require_cuda()
    output_root = config.output_root
    _ensure_new_result_scope(output_root)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        **resolved_contract(config),
        "status": "STARTED",
        "pre_run_commit_sha": pre_run_sha,
        "job_id": job_id,
        "gpu": gpu_info(),
        "result_files": list(RESULT_FILENAMES),
    }
    write_json(output_root / "run_manifest.json", manifest)
    stores = None
    try:
        raw, users, inputs = _load_and_validate_inputs(train_path, cohort_path)
        manifest["inputs"] = inputs
        write_json(output_root / "run_manifest.json", manifest)
        progress("INPUTS_READY", **inputs)
        stores = build_store_registry(raw, users, output_root / "_work" / "stores")
        del raw
        gc.collect()
        progress("RAW_LOG_RELEASED")

        checkpoint_root = output_root / "_work" / "checkpoints"
        training_report: dict[str, Any] = {
            "experiment_id": EXPERIMENT.experiment_id,
            "pre_run_commit_sha": pre_run_sha,
            "config_sha256": config.sha256,
            "runs": {},
        }

        a_predictions, a_report = _fit_first_level(
            stores, EXPERIMENT.run_a_anchors, EXPERIMENT.meta_anchor,
            run="RUN_A", config=config, device=device, checkpoint_root=checkpoint_root,
        )
        training_report["runs"]["RUN_A"] = a_report
        write_json(output_root / "model_training_report.json", training_report)
        m_frame = stores.frames.get(EXPERIMENT.meta_anchor)
        a_bank = make_prediction_bank(m_frame, EXPERIMENT.meta_anchor, a_predictions)
        a_path = output_root / "run_a_meta_prediction_bank.parquet"
        a_bank.write_parquet(a_path)
        a_bank_sha = sha256_file(a_path)
        meta_package = fit_meta(
            bank_arrays(a_bank), prediction_bank_sha256=a_bank_sha,
            code_commit_sha=pre_run_sha, config_sha256=config.sha256,
        )
        meta_path = output_root / "frozen_meta_package.json"
        write_json(meta_path, meta_package)
        meta_sha = sha256_file(meta_path)
        progress("RUN_A_DONE", meta_rmsle=meta_package["rmsle"], bank_sha256=a_bank_sha, meta_sha256=meta_sha)
        del a_predictions, a_bank, m_frame
        gc.collect()
        torch.cuda.empty_cache()

        b_predictions, b_report = _fit_first_level(
            stores, EXPERIMENT.run_b_anchors, EXPERIMENT.validation_anchor,
            run="RUN_B", config=config, device=device, checkpoint_root=checkpoint_root,
        )
        training_report["runs"]["RUN_B"] = b_report
        write_json(output_root / "model_training_report.json", training_report)
        v_frame = stores.frames.get(EXPERIMENT.validation_anchor)
        b_bank = make_prediction_bank(v_frame, EXPERIMENT.validation_anchor, b_predictions)
        b_path = output_root / "run_b_validation_prediction_bank.parquet"
        b_bank.write_parquet(b_path)
        b_bank_sha = sha256_file(b_path)
        b_arrays = bank_arrays(b_bank)
        components = apply_meta_components(meta_package, b_arrays)
        validation = build_validation_report(
            target_z=b_arrays["target"], was_active=b_arrays["active"],
            will_buy=b_bank["will_buy"].to_numpy(), components=components,
            validation_anchor=EXPERIMENT.validation_anchor, job_id=job_id,
            commit_sha=pre_run_sha, config_sha256=config.sha256,
            bank_sha256=b_bank_sha, meta_sha256=meta_sha,
        )
        validation_path = output_root / "validation_report.json"
        write_json(validation_path, validation)

        manifest.update({
            "status": "COMPLETED",
            "elapsed_seconds": time.perf_counter() - started,
            "run_a_meta_rmsle": meta_package["rmsle"],
            "run_b_validation_rmsle": validation["rmsle"],
        })
        write_json(output_root / "run_manifest.json", manifest)
        artifact_hashes = {
            name: sha256_file(output_root / name)
            for name in RESULT_FILENAMES if name != "artifact_sha256.json"
        }
        write_json(output_root / "artifact_sha256.json", artifact_hashes)
        progress("PIPELINE_DONE", validation_rmsle=validation["rmsle"], elapsed_seconds=manifest["elapsed_seconds"])
        return validation
    except Exception as error:
        manifest.update({
            "status": "FAILED",
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        write_json(output_root / "run_manifest.json", manifest)
        progress("PIPELINE_FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if stores is not None:
            stores.close()


def contract_check(config: LoadedConfig) -> dict[str, Any]:
    """Read-only local check; not a training smoke and never reported as one."""

    result = resolved_contract(config)
    result["check_kind"] = "CONTRACT_ONLY_NO_TRAINING"
    result["prediction_columns"] = list(PREDICTION_COLUMNS)
    return result
