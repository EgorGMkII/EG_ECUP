"""Isolated training-level reproduction attempt for the 100k joint recipe.

The historical builder is intentionally imported unchanged.  This entrypoint
adds the operational contract around it: immutable-input hashes, CUDA and
CatBoost GPU gates, a run-scoped output layout, live heartbeats and a small
DataSphere smoke mode.  It does not fit new meta-weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl


EXPECTED_INPUT_HASHES = {
    "sample_template": "06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1",
    "joint_meta": "e9077605f9b438311c46fa7151a099b617ff457eb5f87d972f465502c873961b",
    "reference_prediction_bank": "ddb0e882d80f002752f95d10388df40f09a7bebb3d3e61f92153a1a99fdab0d0",
    "record_submission": "3300512c94579fc6692efb3a6d51a160f0ae5f2375c1476c3aaa54ff775aedcd",
}

INPUT_CANDIDATES = {
    "sample_template": (Path("sample_submit.csv"),),
    # Depending on DataSphere CLI packaging, a one-file local-path may be
    # mounted at its project-relative path or at /job/<basename>.
    "joint_meta": (
        Path("artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json"),
        Path("joint_weights_all_oof_candidate.json"),
    ),
    "reference_prediction_bank": (Path("test_specialists_raw_predictions_250k.parquet"),),
    "record_submission": (Path("submission_specialized_hurdle_joint_rmsle.csv"),),
}


class ContractError(RuntimeError):
    """Raised when a run could not be safely identified or verified."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, candidates in INPUT_CANDIDATES.items():
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ContractError(f"missing immutable input {name}; tried: {[str(candidate) for candidate in candidates]}")
        paths[name] = path
    return paths


def immutable_input_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in resolve_input_paths().items():
        actual = sha256(path)
        expected = EXPECTED_INPUT_HASHES[name]
        if actual != expected:
            raise ContractError(f"immutable input hash mismatch for {name}: {actual} != {expected}")
        hashes[name] = actual
    return hashes


def required_raw_inputs() -> tuple[Path, ...]:
    raw = next((path for path in (Path("data/train.parquet"), Path("train.parquet")) if path.is_file()), None)
    snapshots = next((path for path in (Path("data/snapshots"), Path("snapshots")) if path.is_dir()), None)
    if raw is None or snapshots is None:
        return tuple(path for path in (raw, snapshots) if path is not None)
    return (raw, snapshots, *resolve_input_paths().values())


def validate_local_inputs() -> dict[str, str]:
    raw_candidates = (Path("data/train.parquet"), Path("train.parquet"))
    snapshot_candidates = (Path("data/snapshots"), Path("snapshots"))
    if not any(path.is_file() for path in raw_candidates) or not any(path.is_dir() for path in snapshot_candidates):
        raise ContractError(
            "missing DataSphere local-paths inputs: "
            f"raw_candidates={[str(path) for path in raw_candidates]}, "
            f"snapshot_candidates={[str(path) for path in snapshot_candidates]}"
        )
    resolve_input_paths()
    return immutable_input_hashes()


def run_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "output_dir": output_dir,
        "work_dir": output_dir / "work",
        "feature_store": output_dir / "work" / "feature_store",
        "mmap_dir": output_dir / "work" / "mmap",
        "submission": output_dir / "candidate_submission.csv",
        "raw_predictions": output_dir / "raw_specialist_predictions.parquet",
        "diagnostics": output_dir / "diagnostics.parquet",
        "manifest": output_dir / "run_manifest.json",
        "report": output_dir / "result_report.json",
    }


def routed_path_factory(output_dir: Path) -> Callable[..., Path]:
    """Redirect only legacy writable paths; all historical inputs stay untouched."""
    routes = run_paths(output_dir)
    redirects = {
        "artifacts/specialized_hurdle/feature_store": routes["feature_store"],
        "artifacts/specialized_hurdle/test_specialists_raw_predictions_250k.parquet": routes["raw_predictions"],
        "test_specialists_raw_predictions_250k.parquet": routes["raw_predictions"],
        "submission_specialized_hurdle_joint_rmsle_diagnostics.parquet": routes["diagnostics"],
        "submission_specialized_hurdle_joint_rmsle.csv": routes["submission"],
        "scratch/mmap_joint_run": routes["mmap_dir"],
    }

    def routed_path(*parts: object) -> Path:
        path = Path(*parts)
        return redirects.get(path.as_posix(), path)

    return routed_path


def require_cuda() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ContractError("CUDA is required for record-recipe attempt; torch.cuda.is_available() is False")
    info = {
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
    }
    print(f"[GPU] {info}", flush=True)
    return info


def catboost_gpu_smoke() -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool

    features = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
    target = np.array([0, 1, 1, 0], dtype=np.int32)
    model = CatBoostClassifier(
        iterations=1,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
        verbose=False,
        random_seed=42,
    )
    model.fit(Pool(features, target))
    prediction = model.predict_proba(Pool(features))[:, 1]
    if not np.isfinite(prediction).all():
        raise ContractError("CatBoost GPU smoke produced non-finite predictions")
    print("[GPU] CatBoost task_type=GPU smoke passed", flush=True)
    return {"catboost_task_type": "GPU", "rows": int(features.shape[0])}


def memmap_smoke(work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "fp16_memmap_smoke.dat"
    values = np.memmap(path, dtype=np.float16, mode="w+", shape=(8, 16, 12))
    values[:] = 1.25
    values.flush()
    ok = bool(np.isclose(float(values[0, 0, 0]), 1.25))
    del values
    path.unlink(missing_ok=True)
    if not ok:
        raise ContractError("float16 memmap smoke failed")
    print("[GPU] float16 memmap smoke passed", flush=True)
    return {"dtype": "float16", "shape": [8, 16, 12]}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def local_dry_run(output_dir: Path) -> None:
    hashes = validate_local_inputs()
    template = pl.read_csv(resolve_input_paths()["sample_template"], n_rows=500)
    if template.columns != ["user_id", "predict"] or template.height == 0:
        raise ContractError("sample template must expose non-empty user_id,predict schema")
    snapshot_dir = next(path for path in (Path("data/snapshots"), Path("snapshots")) if path.is_dir())
    snapshot = pl.scan_parquet(snapshot_dir / "snapshot_2026-01-14.parquet").head(1).collect()
    if "user_id" not in snapshot.columns:
        raise ContractError("snapshot_2026-01-14.parquet lacks user_id")
    report = {
        "status": "LOCAL_DRY_RUN_OK",
        "input_sha256": hashes,
        "template_columns": template.columns,
        "template_rows_checked": template.height,
        "snapshot_columns": snapshot.columns,
    }
    write_json(run_paths(output_dir)["report"], report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def validate_outputs(paths: dict[str, Path], template_path: Path) -> dict[str, Any]:
    for key in ("submission", "raw_predictions", "diagnostics"):
        if not paths[key].is_file():
            raise ContractError(f"legacy builder did not create {key}: {paths[key]}")
    template = pl.read_csv(template_path)
    submission = pl.read_csv(paths["submission"])
    raw = pl.read_parquet(paths["raw_predictions"])
    diagnostics = pl.read_parquet(paths["diagnostics"])
    if submission.columns != ["user_id", "predict"] or submission.height != template.height:
        raise ContractError("candidate submission schema or row count differs from template")
    if not np.array_equal(submission["user_id"].to_numpy(), template["user_id"].to_numpy()):
        raise ContractError("candidate submission user_id order differs from template")
    prediction = submission["predict"].to_numpy()
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ContractError("candidate submission contains invalid predictions")
    if raw.height != template.height or diagnostics.height != template.height:
        raise ContractError("prediction bank or diagnostics row count differs from template")
    return {
        "status": "OUTPUT_VALIDATION_OK",
        "row_count": int(submission.height),
        "prediction_min": float(np.min(prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_max": float(np.max(prediction)),
        "artifact_sha256": {key: sha256(paths[key]) for key in ("submission", "raw_predictions", "diagnostics")},
    }


def heartbeat(stop: threading.Event, started: float) -> None:
    while not stop.wait(60):
        print(f"[HEARTBEAT] record-recipe attempt still running; elapsed_seconds={int(time.monotonic() - started)}", flush=True)


def run_full(output_dir: Path, pre_run_sha: str) -> None:
    if output_dir.exists():
        raise ContractError(f"refusing to reuse run directory: {output_dir}")
    hashes_before = validate_local_inputs()
    paths = run_paths(output_dir)
    paths["work_dir"].mkdir(parents=True)
    gpu = require_cuda()
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    manifest = {
        "status": "RUNNING",
        "started_at": started_wall.isoformat(),
        "pre_run_commit": pre_run_sha,
        "entrypoint": "scripts/record_recipe_attempt.py",
        "legacy_builder": "scripts/build_joint_rmsle_submission.py",
        "input_sha256": hashes_before,
        "meta_strategy": "fixed immutable joint meta; no RUN A/meta fitting",
        "output_paths": {key: str(value) for key, value in paths.items() if key not in {"work_dir", "feature_store", "mmap_dir"}},
        "gpu": gpu,
    }
    write_json(paths["manifest"], manifest)

    from scripts import build_joint_rmsle_submission as legacy

    original_path = legacy.Path
    original_classifier = legacy.CatBoostClassifier
    original_regressor = legacy.CatBoostRegressor

    def gpu_classifier(*args: Any, **kwargs: Any) -> Any:
        kwargs.update(task_type="GPU", devices="0", allow_writing_files=False, verbose=100)
        return original_classifier(*args, **kwargs)

    def gpu_regressor(*args: Any, **kwargs: Any) -> Any:
        kwargs.update(task_type="GPU", devices="0", allow_writing_files=False, verbose=100)
        return original_regressor(*args, **kwargs)

    legacy.Path = routed_path_factory(output_dir)
    legacy.CatBoostClassifier = gpu_classifier
    legacy.CatBoostRegressor = gpu_regressor
    stop = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat, args=(stop, started), daemon=True)
    heartbeat_thread.start()
    try:
        print("[STAGE] FULL_TRAINING_STARTED: fixed joint meta; no RUN A", flush=True)
        legacy.main()
    finally:
        stop.set()
        heartbeat_thread.join(timeout=5)
        legacy.Path = original_path
        legacy.CatBoostClassifier = original_classifier
        legacy.CatBoostRegressor = original_regressor

    hashes_after = immutable_input_hashes()
    if hashes_before != hashes_after:
        raise ContractError("an immutable record input changed during the run")
    report = validate_outputs(paths, resolve_input_paths()["sample_template"])
    report.update({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "pre_run_commit": pre_run_sha,
        "input_sha256": hashes_after,
        "public_lb_status": "NOT_SUBMITTED",
    })
    write_json(paths["report"], report)
    manifest["status"] = "COMPLETED"
    manifest["result_report_sha256"] = sha256(paths["report"])
    write_json(paths["manifest"], manifest)
    print("[STAGE] FULL_TRAINING_COMPLETED", flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def run_smoke(output_dir: Path) -> None:
    if output_dir.exists():
        raise ContractError(f"refusing to reuse smoke output directory: {output_dir}")
    hashes = validate_local_inputs()
    paths = run_paths(output_dir)
    paths["work_dir"].mkdir(parents=True)
    report = {
        "status": "DATASPHERE_SMOKE_OK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "input_sha256": hashes,
        "gpu": require_cuda(),
        "catboost": catboost_gpu_smoke(),
        "memmap": memmap_smoke(paths["work_dir"]),
    }
    write_json(paths["report"], report)
    print("[STAGE] SMOKE_COMPLETED", flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-run-sha", default=os.environ.get("PRE_RUN_SHA", "UNSET"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local-dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.local_dry_run:
        local_dry_run(args.output_dir)
    elif args.smoke:
        run_smoke(args.output_dir)
    else:
        run_full(args.output_dir, args.pre_run_sha)


if __name__ == "__main__":
    main()
