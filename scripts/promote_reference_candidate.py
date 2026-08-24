"""Create, but never launch, the next immutable config after a completed gate."""

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
from src.ssl_temporal_stack_v1.runtime import sha256_file


def _completed_manifest(root: Path) -> dict:
    for name in ("gate_manifest.json", "run_manifest.json"):
        path = root / name
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "COMPLETED":
                raise RuntimeError(f"Source result is not completed: {path}")
            return manifest
    raise FileNotFoundError("Promotion requires a completed incremental gate or full validation result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--to", required=True, choices=("full_100k", "validation_250k", "final_250k"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-config", required=True, type=Path)
    args = parser.parse_args()
    source = load_experiment_config(args.source_config)
    manifest = _completed_manifest(args.source_result)
    if manifest.get("config_sha256") != source.sha256:
        raise RuntimeError("Source result config SHA does not match source config")
    raw = copy.deepcopy(source.raw)
    raw.pop("screening", None)
    raw["experiment_id"] = args.experiment_id
    raw["output_root"] = f"artifacts/reference_v1/experiments/{args.experiment_id}"
    if args.to == "full_100k":
        raw["stage"] = "full"
    elif args.to == "validation_250k":
        raw["stage"] = "full"
        raw["inputs"]["cohort"]["full"] = raw["inputs"]["sample_submit"]
    else:
        package = args.source_result / "frozen_meta_package.json"
        if not package.exists():
            raise FileNotFoundError("final_250k requires a completed full validation frozen_meta_package.json")
        raw["stage"] = "final"
        raw["final"] = {
            "frozen_meta_path": package.as_posix(),
            "frozen_meta_sha256": sha256_file(package),
            "source_full_config_sha256": source.sha256,
        }
    if args.output_config.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_config}")
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    promoted = load_experiment_config(args.output_config)
    print(json.dumps({"config": args.output_config.as_posix(), "config_sha256": promoted.sha256, "stage": promoted.stage, "source_result": str(args.source_result)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
