"""Run one immutable config-driven reference experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.config import load_experiment_config
from src.reference_framework_v1.pipeline import run_final, run_validation
from src.reference_framework_v1.registry import build_adapters, collect_required_stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--pre-run-sha")
    parser.add_argument("--job-id")
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    if args.contract_check:
        adapters = build_adapters(config.enabled_models)
        print(json.dumps({"experiment_id": config.experiment_id, "profile": config.profile.name, "stage": config.stage, "enabled_models": list(config.enabled_models), "required_stores": sorted(collect_required_stores(adapters)), "config_sha256": config.sha256}, sort_keys=True), flush=True)
        return
    if not args.pre_run_sha:
        raise SystemExit("--pre-run-sha is required for training or final inference")
    if config.stage == "final":
        run_final(config, pre_run_sha=args.pre_run_sha, job_id=args.job_id)
    else:
        run_validation(config, pre_run_sha=args.pre_run_sha, job_id=args.job_id)


if __name__ == "__main__":
    main()
