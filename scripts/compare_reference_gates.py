"""Paired bootstrap comparison of two composite/full validation banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.config import load_experiment_config
from src.reference_framework_v1.selection import paired_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    result = paired_bootstrap(load_experiment_config(args.experiment_config), baseline_root=args.baseline_root, candidate_root=args.candidate_root, repeats=args.repeats)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
