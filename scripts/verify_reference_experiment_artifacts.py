"""Fail-fast integrity verification for completed reference-framework outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import polars as pl
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
SCREEN_REQUIRED = (
    "screen_manifest.json",
    "resolved_config.yaml",
    "model_training_report.json",
    "run_a_target_prediction_bank.parquet",
    "run_b_target_prediction_bank.parquet",
    "individual_report.json",
    "artifact_sha256.json",
)
FINAL_REQUIRED = (
    "final_manifest.json",
    "resolved_config.yaml",
    "model_training_report.json",
    "submission.csv",
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


def _validate_bank(
    path: Path,
    *,
    rows: int,
    schema: dict[str, list[str]],
    strict_column_order: bool = True,
) -> pl.DataFrame:
    bank = pl.read_parquet(path)
    expected_predictions = [*schema["react"], *schema["churn"], *schema["amount"], *schema.get("direct", [])]
    expected = [*BASE_COLUMNS, *expected_predictions]
    if strict_column_order and bank.columns != expected:
        raise ValueError(f"Unexpected bank schema in {path.name}: {bank.columns}")
    if not strict_column_order and set(bank.columns) != set(expected):
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


def verify_screen(root: Path, *, expected_rows: int | None = None) -> dict[str, Any]:
    """Verify one individual-model screen before using it in an ensemble gate."""
    missing = [name for name in SCREEN_REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing screen artifacts: {missing}")
    manifest = _read_json(root / "screen_manifest.json")
    if manifest.get("status") != "COMPLETED" or manifest.get("mode") != "individual":
        raise ValueError("Screen manifest is not a completed individual run")
    rows = expected_rows or int(manifest["inputs"]["cohort_rows"])
    if rows != int(manifest["inputs"]["cohort_rows"]):
        raise ValueError("Expected rows differs from screen manifest cohort")
    hashes = _read_json(root / "artifact_sha256.json")
    expected_hash_names = set(SCREEN_REQUIRED) - {"artifact_sha256.json"}
    if set(hashes) != expected_hash_names:
        raise ValueError("artifact_sha256.json does not cover exactly the screen result set")
    for name, expected_hash in hashes.items():
        if sha256(root / name) != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {name}")
    target_model = manifest.get("target_model")
    if not isinstance(target_model, str):
        raise ValueError("Screen manifest has no target model")
    from src.reference_framework_v1.registry import build_adapters

    spec = build_adapters((target_model,))[0].prediction_spec
    schema = {
        "react": [spec.react_column] if spec.react_column else [],
        "churn": [spec.churn_column] if spec.churn_column else [],
        "amount": [spec.amount_column] if spec.amount_column else [],
        "direct": [spec.direct_column] if spec.direct_column else [],
    }


def verify_final(root: Path, *, expected_rows: int | None = None) -> dict[str, Any]:
    """Verify one template-aligned, pinned 250k final submission."""
    missing = [name for name in FINAL_REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final artifacts: {missing}")
    manifest = _read_json(root / "final_manifest.json")
    if manifest.get("status") != "COMPLETED" or manifest.get("stage") != "final":
        raise ValueError("Final manifest is not completed")
    rows = expected_rows or 250_000
    if rows != 250_000 or int(manifest.get("submission_rows", -1)) != rows:
        raise ValueError("Invalid final submission row count")
    hashes = _read_json(root / "artifact_sha256.json")
    expected_hash_names = set(FINAL_REQUIRED) - {"artifact_sha256.json"}
    if set(hashes) != expected_hash_names:
        raise ValueError("Final artifact hash set is incomplete")
    for name, expected_hash in hashes.items():
        if sha256(root / name) != expected_hash:
            raise ValueError(f"Final artifact hash mismatch: {name}")
    submission = pl.read_csv(root / "submission.csv")
    if submission.columns != ["user_id", "predict"] or submission.height != rows or submission["user_id"].n_unique() != rows:
        raise ValueError("Invalid final submission schema")
    prediction = submission["predict"].to_numpy()
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid final predictions")
    resolved = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    template_path = Path(resolved["inputs"]["sample_submit"])
    if not template_path.is_file() or sha256(template_path) != manifest["inputs"]["cohort_sha256"]:
        raise ValueError("Final template provenance mismatch")
    template = pl.read_csv(template_path).select("user_id")
    if not submission["user_id"].equals(template["user_id"]):
        raise ValueError("Final user_id order differs from sample template")
    if manifest.get("submission_sha256") != sha256(root / "submission.csv"):
        raise ValueError("Final manifest submission hash mismatch")
    if manifest.get("submission_schema") != ["user_id", "predict"]:
        raise ValueError("Final manifest submission schema mismatch")
    return {"status": "OK", "mode": "final", "experiment_id": manifest["experiment_id"], "rows": rows, "submission_sha256": manifest["submission_sha256"], "artifact_sha256": sha256(root / "artifact_sha256.json")}
    a_path, b_path = root / "run_a_target_prediction_bank.parquet", root / "run_b_target_prediction_bank.parquet"
    a_bank = _validate_bank(a_path, rows=rows, schema=schema, strict_column_order=False)
    b_bank = _validate_bank(b_path, rows=rows, schema=schema, strict_column_order=False)
    if a_bank["user_id"].to_list() != b_bank["user_id"].to_list():
        raise ValueError("Screen RUN A/RUN B prediction-bank user order differs")
    report = _read_json(root / "individual_report.json")
    for section in ("run_a_meta", "run_b_validation"):
        if int(report.get(section, {}).get("rows", -1)) != rows:
            raise ValueError(f"Invalid screen report row count: {section}")
    return {
        "status": "OK",
        "mode": "individual_screen",
        "experiment_id": manifest["experiment_id"],
        "target_model": target_model,
        "rows": rows,
        "run_a_anchor": a_bank["anchor"][0],
        "run_b_anchor": b_bank["anchor"][0],
        "artifact_sha256": sha256(root / "artifact_sha256.json"),
    }


def verify(root: Path, *, expected_rows: int | None = None) -> dict[str, Any]:
    if (root / "final_manifest.json").is_file():
        return verify_final(root, expected_rows=expected_rows)
    if (root / "run_manifest.json").is_file():
        return verify_experiment(root, expected_rows=expected_rows)
    if (root / "screen_manifest.json").is_file():
        return verify_screen(root, expected_rows=expected_rows)
    raise FileNotFoundError(f"No supported reference-framework manifest under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, expected_rows=args.expected_rows), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
