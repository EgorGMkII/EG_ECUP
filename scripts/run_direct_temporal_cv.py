"""Run one immutable direct four-fold temporal-CV experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.direct_temporal_cv_v1.config import load_experiment_config
from src.direct_temporal_cv_v1.pipeline import run_cross_validation
from src.direct_temporal_cv_v1.registry import build_adapters, collect_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--pre-run-sha")
    parser.add_argument("--job-id")
    parser.add_argument("--contract-check", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    adapters = build_adapters(config.enabled_models)
    if args.contract_check:
        requirements = collect_requirements(adapters)
        print(json.dumps({"experiment_id": config.experiment_id, "config_sha256": config.sha256, "folds": [fold.as_dict() for fold in config.folds], "models": list(config.enabled_models), "requirements": requirements.__dict__}, sort_keys=True), flush=True)
        return
    if not args.pre_run_sha:
        raise SystemExit("--pre-run-sha is required for a CV run")
    run_cross_validation(config, pre_run_sha=args.pre_run_sha, job_id=args.job_id)


if __name__ == "__main__":
    main()
