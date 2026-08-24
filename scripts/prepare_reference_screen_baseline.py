"""Materialize an immutable full-stack 25k baseline config for later incremental gates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-config", required=True, type=Path)
    args = parser.parse_args()
    base = load_experiment_config(args.base_config)
    raw = copy.deepcopy(base.raw)
    raw.pop("screening", None)
    raw["stage"] = "screen"
    raw["experiment_id"] = args.experiment_id
    raw["output_root"] = f"artifacts/reference_v1/experiments/{args.experiment_id}"
    if args.output_config.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_config}")
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(args.output_config)
    print(json.dumps({"config": args.output_config.as_posix(), "config_sha256": config.sha256, "stage": config.stage}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
