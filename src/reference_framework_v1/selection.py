"""Result collection and paired validation comparison for candidate promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from .config import ExperimentConfig
from .meta import apply_meta
from .predictions import bank_arrays, schema_from_specs
from .registry import build_adapters


def collect_results(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if (root / "individual_report.json").exists():
            report = json.loads((root / "individual_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "screen_manifest.json").read_text(encoding="utf-8"))
            metrics = report["run_b_validation"]
            row = {"root": str(root), "mode": "individual", "experiment_id": manifest["experiment_id"], "target_model": manifest["target_model"], "status": manifest["status"], "config_sha256": manifest["config_sha256"]}
            for section, values in metrics.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if not isinstance(value, (dict, list)):
                            row[f"{section}.{key}"] = value
            rows.append(row)
        elif (root / "incremental_validation_report.json").exists():
            report = json.loads((root / "incremental_validation_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "gate_manifest.json").read_text(encoding="utf-8"))
            rows.append({"root": str(root), "mode": "incremental_gate", "experiment_id": manifest["experiment_id"], "target_model": manifest["target_model"], "status": manifest["status"], "config_sha256": manifest["config_sha256"], "rmsle": report["rmsle"], "mse_logspace": report["mse_logspace"], "rmsle_01": report["transitions"]["01"]["rmsle"], "rmsle_10": report["transitions"]["10"]["rmsle"]})
        elif (root / "validation_report.json").exists():
            report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            rows.append({"root": str(root), "mode": "full_validation", "experiment_id": manifest["experiment_id"], "target_model": None, "status": manifest["status"], "config_sha256": manifest["config_sha256"], "rmsle": report["rmsle"], "mse_logspace": report["mse_logspace"], "rmsle_01": report["transitions"]["01"]["rmsle"], "rmsle_10": report["transitions"]["10"]["rmsle"]})
        else:
            raise FileNotFoundError(f"No known result report below {root}")
    return rows


def _scored(root: Path, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    schema = schema_from_specs([adapter.prediction_spec for adapter in build_adapters(config.enabled_models)])
    options = (
        ("run_b_incremental_prediction_bank.parquet", "frozen_incremental_meta_package.json"),
        ("run_b_validation_prediction_bank.parquet", "frozen_meta_package.json"),
    )
    for bank_name, package_name in options:
        bank_path, package_path = root / bank_name, root / package_name
        if bank_path.exists() and package_path.exists():
            bank = pl.read_parquet(bank_path)
            package = json.loads(package_path.read_text(encoding="utf-8"))
            return bank["z_target"].to_numpy().astype(np.float64), apply_meta(package, bank_arrays(bank, schema), schema)
    raise FileNotFoundError(f"No scored prediction bank below {root}")


def paired_bootstrap(config: ExperimentConfig, *, baseline_root: Path, candidate_root: Path, repeats: int = 5000, seed: int = 42) -> dict[str, Any]:
    target_a, prediction_a = _scored(baseline_root, config)
    target_b, prediction_b = _scored(candidate_root, config)
    if target_a.shape != target_b.shape or not np.array_equal(target_a, target_b):
        raise ValueError("Paired comparison requires identical ordered validation targets")
    delta = np.square(prediction_b - target_a) - np.square(prediction_a - target_a)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        samples[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    observed = float(delta.mean())
    return {"baseline_root": str(baseline_root), "candidate_root": str(candidate_root), "rows": len(delta), "delta_mse_candidate_minus_baseline": observed, "delta_rmsle_approx": float(np.sqrt(np.mean(np.square(prediction_b - target_a))) - np.sqrt(np.mean(np.square(prediction_a - target_a)))), "bootstrap_ci95_mse": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "probability_candidate_better": float(np.mean(samples < 0.0)), "repeats": repeats, "seed": seed}
