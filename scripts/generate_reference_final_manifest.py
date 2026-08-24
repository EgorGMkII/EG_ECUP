"""Create one immutable DataSphere manifest for a pinned 250k final run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.config import load_experiment_config
from src.reference_framework_v1.pipeline import FINAL_RESULT_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    if config.stage != "final":
        raise ValueError("Final manifest requires stage=final")
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_manifest}")
    meta_path = Path(config.raw["final"]["frozen_meta_path"])
    if not meta_path.is_file():
        raise FileNotFoundError(f"Pinned frozen meta package is missing: {meta_path}")
    root = config.output_root.as_posix()
    paths = ["src/", "scripts/", "configs/", "requirements-datasphere.txt", config.train_path.as_posix(), config.sample_submit_path.as_posix(), meta_path.as_posix()]
    payload = {
        "name": f"ozon-reference-v1-final-{config.experiment_id}",
        "desc": f"Pinned 250k final submission: {config.experiment_id}",
        "cmd": f"export PYTHONPATH=. ; export CUDA_VISIBLE_DEVICES=0 ; python3 -u scripts/run_reference_experiment.py --experiment-config {config.path.as_posix()} --pre-run-sha __PRE_RUN_SHA__",
        "env": {"python": {"type": "manual", "version": "3.10.13", "requirements-file": "requirements-datasphere.txt", "local-paths": paths}},
        "inputs": [config.train_path.as_posix(), config.sample_submit_path.as_posix(), meta_path.as_posix()],
        "outputs": [f"{root}/{name}" for name in FINAL_RESULT_FILES],
        "cloud-instance-types": ["g1.1"],
        "working-storage": {"type": "SSD", "size": "100Gb"},
        "graceful-shutdown": {"signal": "SIGTERM", "timeout": "30s"},
    }
    args.output_manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(args.output_manifest.as_posix(), flush=True)


if __name__ == "__main__":
    main()
