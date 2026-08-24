"""Run one first-level model independently for 25k/100k hyperparameter screening."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.config import load_experiment_config
from src.reference_framework_v1.screening import run_individual_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pre-run-sha", required=True)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    report = run_individual_screen(load_experiment_config(args.experiment_config), output_root=args.output_root, pre_run_sha=args.pre_run_sha, job_id=args.job_id)
    print(report, flush=True)


if __name__ == "__main__":
    main()
