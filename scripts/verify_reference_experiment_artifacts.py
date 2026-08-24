"""Fail-fast integrity verification for one completed reference-framework run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import polars as pl


REQUIRED = (
    "run_manifest.json",
    "resolved_config.yaml",
    "model_training_report.json",
    "run_a_meta_prediction_bank.parquet",
    "frozen_meta_package.json",
    "run_b_validation_prediction_bank.parquet",
    "validation_report.json",
    "artifact_sha256.json",
)
BASE_COLUMNS = ("user_id", "anchor", "was_active", "will_buy", "future_gmv_30d", "z_target")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _validate_bank(path: Path, *, rows: int, schema: dict[str, list[str]]) -> pl.DataFrame:
    bank = pl.read_parquet(path)
    expected_predictions = [*schema["react"], *schema["churn"], *schema["amount"], *schema.get("direct", [])]
    expected = [*BASE_COLUMNS, *expected_predictions]
    if bank.columns != expected:
        raise ValueError(f"Unexpected bank schema in {path.name}: {bank.columns}")
    if bank.height != rows or bank["user_id"].n_unique() != rows:
        raise ValueError(f"Invalid user cardinality in {path.name}")
    if bank["anchor"].n_unique() != 1:
        raise ValueError(f"Mixed holdout anchors in {path.name}")
    numeric = bank.select(["future_gmv_30d", "z_target", *expected_predictions]).to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite values in {path.name}")
    if (bank["future_gmv_30d"] < 0).any() or (bank["z_target"] < 0).any():
        raise ValueError(f"Negative target in {path.name}")
    return bank


def verify_experiment(root: Path, *, expected_rows: int | None = None) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result artifacts: {missing}")
    manifest = _read_json(root / "run_manifest.json")
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Run is not completed: {manifest.get('status')}")
    rows = expected_rows or int(manifest["inputs"]["cohort_rows"])
    if rows != int(manifest["inputs"]["cohort_rows"]):
        raise ValueError("Expected rows differs from manifest cohort")
    hashes = _read_json(root / "artifact_sha256.json")
    expected_hash_names = set(REQUIRED) - {"artifact_sha256.json"}
    if set(hashes) != expected_hash_names:
        raise ValueError("artifact_sha256.json does not cover exactly the result set")
    for name, expected_hash in hashes.items():
        if sha256(root / name) != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {name}")
    schema = manifest.get("prediction_schema")
    if not isinstance(schema, dict):
        raise ValueError("Invalid prediction schema in manifest")
    schema = {key: list(schema.get(key, [])) for key in ("react", "churn", "amount", "direct")}
    if not all(schema[key] for key in ("react", "churn", "amount")):
        raise ValueError("Invalid prediction schema in manifest")
    a_path, b_path = root / "run_a_meta_prediction_bank.parquet", root / "run_b_validation_prediction_bank.parquet"
    a_bank, b_bank = _validate_bank(a_path, rows=rows, schema=schema), _validate_bank(b_path, rows=rows, schema=schema)
    if a_bank["user_id"].to_list() != b_bank["user_id"].to_list():
        raise ValueError("RUN A/RUN B prediction bank user order differs")
    meta = _read_json(root / "frozen_meta_package.json")
    if meta.get("config_sha256") != manifest.get("config_sha256") or meta.get("prediction_bank_sha256") != sha256(a_path):
        raise ValueError("Frozen meta provenance mismatch")
    expected_feature_order = {key: values for key, values in schema.items() if key != "direct" or values}
    if meta.get("feature_order") != expected_feature_order:
        raise ValueError("Frozen meta feature order mismatch")
    report = _read_json(root / "validation_report.json")
    provenance = report.get("provenance", {})
    if provenance.get("config_sha256") != manifest.get("config_sha256") or provenance.get("prediction_bank_sha256") != sha256(b_path) or provenance.get("frozen_meta_sha256") != sha256(root / "frozen_meta_package.json"):
        raise ValueError("Validation provenance mismatch")
    rmsle = float(report["rmsle"])
    if not np.isfinite(rmsle) or rmsle < 0:
        raise ValueError("Invalid validation RMSLE")
    return {"status": "OK", "experiment_id": manifest["experiment_id"], "rows": rows, "run_a_anchor": a_bank["anchor"][0], "run_b_anchor": b_bank["anchor"][0], "rmsle": rmsle, "artifact_sha256": sha256(root / "artifact_sha256.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    print(json.dumps(verify_experiment(args.root, expected_rows=args.expected_rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
