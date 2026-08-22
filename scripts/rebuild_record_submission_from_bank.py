"""Rebuild the immutable 100k record submission from its saved prediction bank.

This module is deliberately an artifact-level assembler.  It never imports model
code, creates features, trains models, or performs inference.  The raw specialist
bank is a contract: classifier columns are logits and amount columns are values in
``z = log1p(GMV)`` space, in the fixed model order below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_ORDER = ("CatBoost", "S1_GRU", "S2_GRU", "ETT")
REACT_COLUMNS = ("cb_react_logit", "s1_react_logit", "s2_react_logit", "ett_react_logit")
CHURN_COLUMNS = ("cb_churn_logit", "s1_churn_logit", "s2_churn_logit", "ett_churn_logit")
AMOUNT_COLUMNS = ("cb_amount_z", "s1_amount_z", "s2_amount_z", "ett_amount_z")
REQUIRED_BANK_COLUMNS = ("user_id", "was_active", *REACT_COLUMNS, *CHURN_COLUMNS, *AMOUNT_COLUMNS)
SUPPORTED_META_VERSIONS = {None, "record_100k_joint_meta_v1"}


class ContractError(ValueError):
    """Raised when an input cannot be the canonical record-reproduction input."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (4,) or not np.isfinite(vector).all():
        raise ContractError(f"{name} must contain exactly four finite values")
    return vector


def validate_meta(meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    version = meta.get("schema_version")
    if version not in SUPPORTED_META_VERSIONS:
        raise ContractError(f"unsupported meta JSON version: {version!r}")
    if tuple(meta.get("model_order", ())) != MODEL_ORDER:
        raise ContractError(f"invalid model_order; expected {list(MODEL_ORDER)}")
    if "amount_scaler" in meta:
        raise ContractError("record 100k meta JSON must not contain an amount_scaler")

    react = _as_float_vector(meta.get("react_stack_weights", ()), "react_stack_weights")
    churn = _as_float_vector(meta.get("churn_stack_weights", ()), "churn_stack_weights")
    amount = _as_float_vector(meta.get("amount_ridge_coefficients", ()), "amount_ridge_coefficients")
    intercept = float(meta.get("amount_ridge_intercept", float("nan")))
    alpha = float(meta.get("ALPHA", float("nan")))
    if not math.isfinite(intercept) or not math.isfinite(alpha):
        raise ContractError("amount_ridge_intercept and ALPHA must be finite")
    if not np.isclose(react.sum(), 1.0, atol=1e-8) or not np.isclose(churn.sum(), 1.0, atol=1e-8):
        raise ContractError("React and Churn weights must each sum to 1")
    if alpha != 1.1:
        raise ContractError(f"record contract requires ALPHA=1.1, got {alpha}")
    return react, churn, amount, intercept, alpha


def validate_bank_columns(columns: Sequence[str]) -> None:
    missing = [column for column in REQUIRED_BANK_COLUMNS if column not in columns]
    if missing:
        raise ContractError(f"prediction bank is missing required columns: {missing}")
    # The explicit *_logit names are the contract.  Values are never classified as
    # logits/probabilities by their numeric range.
    if any(column.endswith("_prob") for column in columns):
        raise ContractError("probability columns are unsupported; record contract requires raw logits")


def _expit(values: np.ndarray) -> np.ndarray:
    # Stable enough for the stored float32/float64 logits and avoids SciPy runtime.
    return 1.0 / (1.0 + np.exp(-values))


def compute_predictions(
    was_active: Sequence[int | bool],
    react_logits: np.ndarray,
    churn_logits: np.ndarray,
    amount_z: np.ndarray,
    meta: Mapping[str, Any],
) -> np.ndarray:
    """Apply the canonical record formula in float64, returning GMV predictions."""
    react_w, churn_w, amount_w, intercept, alpha = validate_meta(meta)
    active = np.asarray(was_active)
    react = np.asarray(react_logits, dtype=np.float64)
    churn = np.asarray(churn_logits, dtype=np.float64)
    amount = np.asarray(amount_z, dtype=np.float64)
    if react.ndim != 2 or churn.shape != react.shape or amount.shape != react.shape or react.shape[1] != 4:
        raise ContractError("React, Churn, and Amount matrices must have shape (n_users, 4)")
    if active.shape != (react.shape[0],):
        raise ContractError("was_active must have one value per prediction row")
    if not np.isin(active, [0, 1, False, True]).all():
        raise ContractError("was_active must contain only 0/1 values")
    if not (np.isfinite(react).all() and np.isfinite(churn).all() and np.isfinite(amount).all()):
        raise ContractError("prediction bank contains NaN or Inf")
    if np.all(np.var(np.column_stack((react, churn, amount)), axis=0) == 0.0):
        raise ContractError("prediction bank has zero variance in every prediction column")

    p_react = _expit(react @ react_w)
    p_churn = _expit(churn @ churn_w)
    p_buy = np.where(active.astype(bool), 1.0 - p_churn, p_react)
    conditional_z = np.maximum(0.0, amount @ amount_w + intercept)
    z_prediction = np.clip(np.power(p_buy, alpha) * conditional_z, 0.0, None)
    return np.expm1(z_prediction)


def _require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on execution environment
        raise RuntimeError("polars is required to read the Parquet prediction bank") from exc
    return pl


def _read_template_user_ids(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["user_id", "predict"]:
            raise ContractError("sample template schema must be exactly user_id,predict")
        user_ids = [row["user_id"] for row in reader]
    if len(user_ids) != len(set(user_ids)):
        raise ContractError("sample template contains duplicate user_id values")
    return np.asarray(user_ids, dtype=str)


def validate_user_ids(bank_ids: Sequence[str], template_ids: Sequence[str]) -> None:
    """Validate uniqueness and exact membership before the template-order join."""
    bank_values = list(map(str, bank_ids))
    template_values = list(map(str, template_ids))
    if len(bank_values) != len(set(bank_values)):
        raise ContractError("prediction bank contains duplicate user_id values")
    if len(template_values) != len(set(template_values)):
        raise ContractError("sample template contains duplicate user_id values")
    if len(bank_values) != len(template_values) or set(bank_values) != set(template_values):
        raise ContractError("prediction-bank users do not exactly match sample template users")


def rebuild(bank_path: Path, meta_path: Path, template_path: Path, output_path: Path) -> dict[str, Any]:
    """Read immutable inputs, rebuild CSV in template order, and return verification data."""
    if output_path.exists():
        raise ContractError(f"refusing to overwrite output: {output_path}")
    pl = _require_polars()
    with meta_path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    validate_meta(meta)

    bank = pl.read_parquet(bank_path)
    validate_bank_columns(bank.columns)
    template_ids = _read_template_user_ids(template_path)
    bank_ids = bank["user_id"].cast(pl.Utf8).to_numpy()
    validate_user_ids(bank_ids, template_ids)

    # The join—not source row order—defines the output order.
    template = pl.DataFrame({"user_id": template_ids, "_template_order": np.arange(len(template_ids))})
    bank = bank.with_columns(pl.col("user_id").cast(pl.Utf8))
    ordered = template.join(bank, on="user_id", how="left", validate="1:1").sort("_template_order")
    if ordered.height != len(template_ids) or ordered.null_count().sum_horizontal().sum() != 0:
        raise ContractError("join to template produced missing prediction rows")

    prediction = compute_predictions(
        ordered["was_active"].to_numpy(),
        ordered.select(REACT_COLUMNS).to_numpy(),
        ordered.select(CHURN_COLUMNS).to_numpy(),
        ordered.select(AMOUNT_COLUMNS).to_numpy(),
        meta,
    )
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ContractError("canonical formula produced invalid predictions")

    output_path.parent.mkdir(parents=True, exist_ok=False)
    # Polars CSV is intentionally used because it is also the writer used by the
    # record builder; float64 formula inputs preserve the builder's NumPy semantics.
    result = pl.DataFrame({"user_id": template_ids, "predict": prediction})
    result.write_csv(output_path)
    return {
        "row_count": int(result.height),
        "schema": ["user_id", "predict"],
        "prediction_stats": prediction_stats(prediction),
        "output_sha256": sha256(output_path),
        "meta": meta,
    }


def prediction_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def compare_csvs(rebuilt_path: Path, record_path: Path) -> dict[str, Any]:
    def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return np.asarray([row["user_id"] for row in rows], dtype=str), np.asarray([float(row["predict"]) for row in rows])

    rebuilt_ids, rebuilt = load(rebuilt_path)
    record_ids, record = load(record_path)
    same_users = bool(np.array_equal(rebuilt_ids, record_ids))
    if rebuilt.shape != record.shape or not same_users:
        return {"status": "REPRODUCTION_FAILED", "row_count_match": rebuilt.shape == record.shape, "user_id_order_match": same_users}
    difference = np.abs(rebuilt - record)
    byte_exact = sha256(rebuilt_path) == sha256(record_path)
    return {
        "status": "BYTE_EXACT_REPRODUCTION" if byte_exact else "NUMERICALLY_EQUIVALENT_REPRODUCTION",
        "row_count_match": True,
        "user_id_order_match": True,
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "non_matching_values": int(np.count_nonzero(difference != 0.0)),
        "record_stats": prediction_stats(record),
        "rebuilt_stats": prediction_stats(rebuilt),
        "record_sha256": sha256(record_path),
        "rebuilt_sha256": sha256(rebuilt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("test_specialists_raw_predictions_250k.parquet"))
    parser.add_argument("--meta", type=Path, default=Path("artifacts/specialized_hurdle/joint_meta_optimization/joint_weights_all_oof_candidate.json"))
    parser.add_argument("--template", type=Path, default=Path("sample_submit.csv"))
    parser.add_argument("--record", type=Path, default=Path("submission_specialized_hurdle_joint_rmsle.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/record_100k_reproduction"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ContractError(f"refusing to reuse existing output directory: {args.output_dir}")
    inputs = (args.bank, args.meta, args.template, args.record)
    for input_path in inputs:
        if not input_path.is_file():
            raise ContractError(f"required input does not exist: {input_path}")

    output_path = args.output_dir / "rebuilt_submission.csv"
    rebuilt = rebuild(args.bank, args.meta, args.template, output_path)
    verification = compare_csvs(output_path, args.record)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {"prediction_bank": str(args.bank), "meta_json": str(args.meta), "sample_template": str(args.template), "record_submission": str(args.record)},
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "output_path": str(output_path),
        "output_sha256": rebuilt["output_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "meta_json_sha256": sha256(args.meta),
        "row_count": rebuilt["row_count"],
        "schema": rebuilt["schema"],
        "reproduction_status": verification["status"],
    }
    with (args.output_dir / "verification.json").open("w", encoding="utf-8") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (args.output_dir / "artifact_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
