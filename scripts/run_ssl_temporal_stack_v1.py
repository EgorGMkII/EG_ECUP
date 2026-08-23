"""CLI entrypoint for the isolated SSL temporal validation experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ssl_temporal_stack_v1.config import DEFAULT_CONFIG_PATH, load_config
from src.ssl_temporal_stack_v1.pipeline import contract_check, run_pipeline
from src.ssl_temporal_stack_v1.smoke import default_smoke_output, run_gpu_smoke, run_local_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--pre-run-sha")
    parser.add_argument("--job-id")
    parser.add_argument("--contract-check", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.experiment_config)
    if args.contract_check:
        print(json.dumps(contract_check(config), indent=2, sort_keys=True), flush=True)
        return
    if args.local_smoke:
        result = run_local_smoke(
            config,
            train_path=args.train or config.train_path,
            cohort_path=args.cohort or config.cohort_path,
            output_root=args.smoke_output or default_smoke_output(),
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    if args.gpu_smoke:
        if not args.pre_run_sha:
            raise SystemExit("--pre-run-sha is required for a DataSphere GPU smoke")
        result = run_gpu_smoke(
            config,
            train_path=args.train or config.train_path,
            cohort_path=args.cohort or config.cohort_path,
            output_root=args.smoke_output or default_smoke_output(),
            pre_run_sha=args.pre_run_sha,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    pre_run_sha = args.pre_run_sha or os.environ.get("PRE_RUN_SHA")
    if not pre_run_sha:
        raise SystemExit("--pre-run-sha or PRE_RUN_SHA is required for a full run")
    run_pipeline(
        config,
        train_path=args.train or config.train_path,
        cohort_path=args.cohort or config.cohort_path,
        pre_run_sha=pre_run_sha,
        job_id=args.job_id or os.environ.get("DATASPHERE_JOB_ID"),
    )


if __name__ == "__main__":
    main()
