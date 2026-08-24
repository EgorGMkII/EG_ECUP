"""Create one DataSphere manifest for one immutable individual-screen config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    screening = config.raw.get("screening")
    if not screening:
        raise ValueError("Individual manifest requires screening.target_model")
    cohort = config.cohort_path.as_posix()
    root = config.output_root.as_posix()
    payload = {
        "name": f"ozon-reference-v1-screen-{config.experiment_id}",
        "desc": f"Individual {screening['target_model']} screen: {config.experiment_id}",
        "cmd": f"export PYTHONPATH=. ; export CUDA_VISIBLE_DEVICES=0 ; python3 -u scripts/run_reference_individual_screen.py --experiment-config {config.path.as_posix()} --output-root {root} --pre-run-sha __PRE_RUN_SHA__",
        "env": {"python": {"type": "manual", "version": "3.10.13", "requirements-file": "requirements-datasphere.txt", "local-paths": ["src/", "scripts/", "configs/", "requirements-datasphere.txt", config.train_path.as_posix(), config.sample_submit_path.as_posix(), cohort]}},
        "inputs": [config.train_path.as_posix(), config.sample_submit_path.as_posix(), cohort],
        "outputs": [f"{root}/{name}" for name in ("screen_manifest.json", "resolved_config.yaml", "run_a_target_prediction_bank.parquet", "run_b_target_prediction_bank.parquet", "individual_report.json", "model_training_report.json", "artifact_sha256.json")],
        "cloud-instance-types": ["g1.1"],
        "working-storage": {"type": "SSD", "size": "100Gb"},
        "graceful-shutdown": {"signal": "SIGTERM", "timeout": "30s"},
    }
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_manifest}")
    args.output_manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(args.output_manifest.as_posix(), flush=True)


if __name__ == "__main__":
    main()
