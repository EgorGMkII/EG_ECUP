"""Cross-validation orchestration skeleton.

The filling agent must build/release one fold at a time and never reuse fitted
models, optimizer state, or validation labels between folds.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import polars as pl

from .artifacts import create_run_root, sha256_file, write_json, write_yaml
from .config import ExperimentConfig, resolved_config
from .datasets import build_target_z
from .features import SparseAggregateFeatureProvider
from .btyd import DirectBTYDFeatureProvider
from .metrics import evaluate_z, metrics_dict
from .registry import build_adapters
from src.ssl_temporal_stack_v1.stores import build_daily_tensor_store, build_event_memmap_store


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _template_users(path: Path) -> np.ndarray:
    table = pl.read_csv(path)
    if "user_id" not in table.columns:
        raise ValueError(f"sample_submit lacks user_id: {path}")
    values = table["user_id"].to_numpy()
    if values.ndim != 1 or len(values) != 250_000 or len(np.unique(values)) != len(values):
        raise ValueError("sample_submit must contain exactly 250,000 unique user IDs")
    return values.astype(np.int64, copy=False)


def _write_sha_manifest(root: Path) -> None:
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256sums.json":
            entries[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    write_json(root / "sha256sums.json", entries)


def run_cross_validation(config: ExperimentConfig, *, pre_run_sha: str, job_id: str | None = None) -> None:
    """Execute the four folds sequentially, releasing fold state after each."""
    output_root = _resolve(config.output_root)
    root = create_run_root(output_root)
    train_path = _resolve(config.train_path)
    submit_path = _resolve(config.sample_submit_path)
    users = _template_users(submit_path)
    raw = pl.read_parquet(train_path)
    if "event_date" not in raw.columns:
        raise ValueError("train parquet lacks event_date")
    adapters = build_adapters(config.enabled_models)
    for adapter in adapters:
        adapter.validate_config(config.raw["models"][adapter.model_id])
    use_coles = bool(config.raw.get("features", {}).get("coles_v1", False))
    needs_tabular = any(adapter.requirements.tabular_features for adapter in adapters)
    needs_daily = any(adapter.requirements.daily_tensor for adapter in adapters)
    needs_events = any(adapter.requirements.event_sequences for adapter in adapters) or use_coles
    if needs_tabular and not config.raw.get("features", {}).get("base_sparse_v1", False):
        raise ValueError("tabular direct models require features.base_sparse_v1=true")
    use_btyd = bool(config.raw.get("features", {}).get("btyd_v1", False))
    provider = DirectBTYDFeatureProvider() if use_btyd else SparseAggregateFeatureProvider()
    write_yaml(root / "resolved_config.yaml", resolved_config(config))
    write_json(root / "protocol_manifest.json", {
        "protocol": "FOUR_FOLD_250K_V1",
        "folds": [fold.as_dict() for fold in config.folds],
        "template_users": int(len(users)),
        "template_user_order_sha256": hashlib.sha256(users.tobytes()).hexdigest(),
        "pre_run_sha": pre_run_sha,
        "job_id": job_id,
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "sample_submit_path": str(submit_path),
        "sample_submit_sha256": sha256_file(submit_path),
        "causal_contract": "features <= anchor; train target (T-30, T]; validation target (T, T+30]",
    })
    all_metrics: list[dict[str, Any]] = []
    for fold in config.folds:
        print(f"[DIRECT_CV] fold={fold.fold_id} feature_build_start anchor={fold.inference_anchor}", flush=True)
        snapshots = provider.build_pair(raw, users.tolist(), fold.train_anchor, fold.inference_anchor) if needs_tabular else None
        train_z = build_target_z(raw, users.tolist(), fold.train_target_start, fold.train_target_end)
        validation_z = build_target_z(raw, users.tolist(), fold.validation_target_start, fold.validation_target_end)
        fold_dir = root / f"fold_{fold.fold_id}"
        fold_dir.mkdir()
        if snapshots is not None:
            write_json(fold_dir / "feature_manifest.json", snapshots.manifest)
        use_coles = bool(config.raw.get("features", {}).get("coles_v1", False))
        if use_coles:
            from .coles import train_coles_embeddings
            print(f"[DIRECT_CV] fold={fold.fold_id} CoLES training start...", flush=True)
            coles_train, coles_valid = train_coles_embeddings(
                event_store.get(fold.train_anchor.isoformat()),
                event_store.get(fold.inference_anchor.isoformat()),
                users.tolist(),
                __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"),
                epochs=2,
                out_dim=32
            )
            snapshots_train = snapshots.train.join(coles_train, on="user_id")
            snapshots_valid = snapshots.validation.join(coles_valid, on="user_id")
            print(f"[DIRECT_CV] fold={fold.fold_id} CoLES joined: total features={len(snapshots_train.columns)-1}", flush=True)
        else:
            snapshots_train = snapshots.train if snapshots is not None else None
            snapshots_valid = snapshots.validation if snapshots is not None else None

        fold_reports: dict[str, Any] = {}
        context_kwargs = {
            "fold": fold,
            "users": users,
            "train_target_z": train_z,
            "validation_target_z": validation_z,
            "train_tabular": snapshots_train,
            "validation_tabular": snapshots_valid,
            "train_daily": daily_store.get(fold.train_anchor.isoformat()) if daily_store is not None else None,
            "validation_daily": daily_store.get(fold.inference_anchor.isoformat()) if daily_store is not None else None,
            "train_events": event_store.get(fold.train_anchor.isoformat()) if event_store is not None else None,
            "validation_events": event_store.get(fold.inference_anchor.isoformat()) if event_store is not None else None,
            "device": __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"),
            "output_dir": fold_dir,
            "root_seed": config.root_seed,
        }
        from .base import FoldContext
        context = FoldContext(**context_kwargs)
        fold_summary: dict[str, Any] = {"fold_id": fold.fold_id, "models": {}}
        for adapter in adapters:
            print(f"[DIRECT_CV] fold={fold.fold_id} model={adapter.model_id} train_start", flush=True)
            model_config = adapter.validate_config(config.raw["models"][adapter.model_id])
            result = adapter.fit_predict_fold(context, model_config)
            if not np.array_equal(result.user_ids, users):
                raise ValueError(f"{adapter.model_id} prediction user order mismatch")
            fold_metrics = evaluate_z(validation_z, result.prediction_z)
            bank = pl.DataFrame({"user_id": users, "prediction_z": result.prediction_z, "prediction": np.maximum(np.expm1(result.prediction_z), 0.0), "target_z": validation_z})
            bank.write_parquet(fold_dir / f"{adapter.model_id}_predictions.parquet")
            # Keep the guide's compact root-level name for the one-model gate;
            # the fold directory remains the canonical location once multiple
            # independent model banks are enabled.
            if len(adapters) == 1:
                bank.write_parquet(root / f"fold_{fold.fold_id}_predictions.parquet")
            write_json(fold_dir / f"{adapter.model_id}_metrics.json", metrics_dict(fold_metrics))
            fold_summary["models"][adapter.model_id] = metrics_dict(fold_metrics)
            fold_reports[adapter.model_id] = result.training_report
            print(f"[DIRECT_CV] fold={fold.fold_id} model={adapter.model_id} rmsle={fold_metrics.rmsle:.9f} done", flush=True)
        write_json(fold_dir / "training_report.json", fold_reports)
        write_json(fold_dir / "fold_metrics.json", fold_summary)
        all_metrics.append(fold_summary)
        if event_store is not None:
            event_store.close()
        del snapshots, context, train_z, validation_z, daily_store, event_store
        shutil.rmtree(store_root, ignore_errors=True)
        gc.collect()
    model_summary: dict[str, Any] = {}
    for model_id in config.enabled_models:
        scores = [item["models"][model_id]["rmsle"] for item in all_metrics]
        model_summary[model_id] = {"fold_rmsle": scores, "mean_rmsle": float(np.mean(scores))}
    write_json(root / "cv_summary.json", {"experiment_id": config.experiment_id, "models": model_summary, "folds": all_metrics})
    _write_sha_manifest(root)
    print(f"[DIRECT_CV] completed output_root={root}", flush=True)


def run_final_submission(config: ExperimentConfig, *, pre_run_sha: str, job_id: str | None = None) -> Path:
    """TODO: fresh final training after a separately approved CV result."""
    raise NotImplementedError("Final submission is intentionally not implemented in the skeleton")
