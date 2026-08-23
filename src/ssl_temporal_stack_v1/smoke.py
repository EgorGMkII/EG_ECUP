"""Real 100-user, tiny-budget local smoke for every SSL V1 model path."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl
import torch

from .adapters import fit_predict_catboost, fit_predict_ett, fit_predict_gru
from .config import LoadedConfig
from .contract import EXPERIMENT, NeuralBudget
from .diagnostics import build_validation_report
from .meta import apply_meta_components, fit_meta
from .pipeline import merge_adapter_results
from .predictions import bank_arrays, make_prediction_bank
from .runtime import progress, require_cuda, sha256_file, write_json
from .stores import build_state_labels, build_store_registry_for_anchors


SMOKE_TRAIN_ANCHOR = "2025-11-10"
SMOKE_HOLDOUT_ANCHOR = "2025-12-15"
SMOKE_BUDGET = NeuralBudget(
    ssl_steps=1,
    base_steps=1,
    specialist_head_steps=1,
    specialist_finetune_steps=1,
    batch_size=16,
)


def default_smoke_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts/ssl_temporal_stack_v1") / f"local_smoke_{stamp}"


def _smoke_cohort(raw: pl.DataFrame, full_users: list[int]) -> tuple[list[int], dict[str, int]]:
    candidates = full_users[:5000]
    labels = build_state_labels(raw, candidates, SMOKE_TRAIN_ANCHOR)
    selected: list[int] = []
    counts: dict[str, int] = {}
    for previous in (0, 1):
        for future in (0, 1):
            name = f"{previous}{future}"
            values = labels.filter(
                (pl.col("was_active") == previous) & (pl.col("will_buy") == future)
            )["user_id"].head(25).to_list()
            if len(values) != 25:
                raise RuntimeError(f"Could not select 25 smoke users for transition {name}")
            selected.extend(values)
            counts[name] = len(values)
    positions = {user_id: index for index, user_id in enumerate(full_users)}
    selected.sort(key=positions.__getitem__)
    if len(selected) != 100 or len(set(selected)) != 100:
        raise RuntimeError("Smoke cohort is not 100 unique users")
    return selected, counts


def _user_order_sha256(users: list[int]) -> str:
    encoded = json.dumps(users, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_smoke(
    config: LoadedConfig,
    *,
    train_path: Path,
    cohort_path: Path,
    output_root: Path,
    device: torch.device,
    run_kind: str,
    catboost_override: dict[str, Any],
    pre_run_sha: str,
) -> dict[str, Any]:
    """Run all real model paths with one optimizer step per neural phase."""

    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT.experiment_id,
        "run_kind": run_kind,
        "status": "STARTED",
        "train_anchor": SMOKE_TRAIN_ANCHOR,
        "holdout_anchor": SMOKE_HOLDOUT_ANCHOR,
        "neural_budget_override": asdict(SMOKE_BUDGET),
        "catboost_override": catboost_override,
        "config_sha256": config.sha256,
        "pre_run_commit_sha": pre_run_sha,
    }
    write_json(output_root / "smoke_manifest.json", manifest)
    stores = None
    try:
        raw = pl.read_parquet(train_path)
        if raw["event_date"].dtype == pl.Utf8:
            raw = raw.with_columns(pl.col("event_date").str.to_date())
        cohort = pl.read_parquet(cohort_path)
        full_users = cohort["user_id"].to_list()
        users, transition_counts = _smoke_cohort(raw, full_users)
        smoke_cohort_sha = _user_order_sha256(users)
        manifest.update({
            "smoke_users": len(users),
            "smoke_cohort_order_sha256": smoke_cohort_sha,
            "selection_transition_counts": transition_counts,
            "device": str(device),
        })
        write_json(output_root / "smoke_manifest.json", manifest)
        stores = build_store_registry_for_anchors(
            raw,
            users,
            store_anchors=(SMOKE_TRAIN_ANCHOR, SMOKE_HOLDOUT_ANCHOR),
            training_anchors=(SMOKE_TRAIN_ANCHOR,),
            root=output_root / "_work" / "stores",
            cohort_sha256=smoke_cohort_sha,
        )
        del raw
        gc.collect()
        checkpoint_root = output_root / "_work" / "checkpoints"
        results = [
            fit_predict_catboost(
                stores,
                (SMOKE_TRAIN_ANCHOR,),
                SMOKE_HOLDOUT_ANCHOR,
                run="SMOKE",
                config=config,
                parameter_overrides=catboost_override,
            ),
            fit_predict_gru(
                stores,
                (SMOKE_TRAIN_ANCHOR,),
                SMOKE_HOLDOUT_ANCHOR,
                run="SMOKE",
                model_id="s1",
                device=device,
                checkpoint_root=checkpoint_root,
                budget=SMOKE_BUDGET,
            ),
            fit_predict_gru(
                stores,
                (SMOKE_TRAIN_ANCHOR,),
                SMOKE_HOLDOUT_ANCHOR,
                run="SMOKE",
                model_id="s2",
                device=device,
                checkpoint_root=checkpoint_root,
                budget=SMOKE_BUDGET,
            ),
            fit_predict_ett(
                stores,
                (SMOKE_TRAIN_ANCHOR,),
                SMOKE_HOLDOUT_ANCHOR,
                run="SMOKE",
                device=device,
                checkpoint_root=checkpoint_root,
                budget=SMOKE_BUDGET,
                micro_batch_size=8,
                accumulation_steps=1,
            ),
        ]
        predictions, reports = merge_adapter_results(results, expected_rows=100)
        write_json(output_root / "smoke_training_report.json", reports)
        holdout = stores.frames.get(SMOKE_HOLDOUT_ANCHOR)
        bank = make_prediction_bank(holdout, SMOKE_HOLDOUT_ANCHOR, predictions)
        bank_path = output_root / "smoke_prediction_bank.parquet"
        bank.write_parquet(bank_path)
        bank_sha = sha256_file(bank_path)
        arrays = bank_arrays(bank)
        meta = fit_meta(
            arrays,
            prediction_bank_sha256=bank_sha,
            code_commit_sha=pre_run_sha,
            config_sha256=config.sha256,
        )
        meta_path = output_root / "smoke_meta_package.json"
        write_json(meta_path, meta)
        meta_sha = sha256_file(meta_path)
        components = apply_meta_components(meta, arrays)
        report = build_validation_report(
            target_z=arrays["target"],
            was_active=arrays["active"],
            will_buy=bank["will_buy"].to_numpy(),
            components=components,
            validation_anchor=SMOKE_HOLDOUT_ANCHOR,
            job_id=None,
            commit_sha=pre_run_sha,
            config_sha256=config.sha256,
            bank_sha256=bank_sha,
            meta_sha256=meta_sha,
        )
        report["warning"] = "IN_SAMPLE META FIT SMOKE METRIC; NOT A VALIDATION RESULT"
        write_json(output_root / "smoke_report.json", report)
        manifest.update({
            "status": "COMPLETED",
            "elapsed_seconds": time.perf_counter() - started,
            "prediction_columns": list(predictions),
            "prediction_bank_sha256": bank_sha,
            "meta_sha256": meta_sha,
        })
        write_json(output_root / "smoke_manifest.json", manifest)
        progress("LOCAL_SMOKE_DONE", output_root=str(output_root), elapsed_seconds=manifest["elapsed_seconds"])
        return manifest
    except Exception as error:
        manifest.update({
            "status": "FAILED",
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        write_json(output_root / "smoke_manifest.json", manifest)
        progress("LOCAL_SMOKE_FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if stores is not None:
            stores.close()


def run_local_smoke(
    config: LoadedConfig,
    *,
    train_path: Path,
    cohort_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    return _run_smoke(
        config,
        train_path=train_path,
        cohort_path=cohort_path,
        output_root=output_root,
        device=torch.device("cpu"),
        run_kind="LOCAL_100_USER_TINY_STEP_SMOKE_NOT_A_BASELINE",
        catboost_override={"task_type": "CPU", "iterations": 2, "verbose": False},
        pre_run_sha="UNCOMMITTED_LOCAL_SMOKE",
    )


def run_gpu_smoke(
    config: LoadedConfig,
    *,
    train_path: Path,
    cohort_path: Path,
    output_root: Path,
    pre_run_sha: str,
) -> dict[str, Any]:
    return _run_smoke(
        config,
        train_path=train_path,
        cohort_path=cohort_path,
        output_root=output_root,
        device=require_cuda(),
        run_kind="DATASPHERE_GPU_100_USER_TINY_STEP_SMOKE_NOT_A_BASELINE",
        catboost_override={"iterations": 2, "verbose": False},
        pre_run_sha=pre_run_sha,
    )
