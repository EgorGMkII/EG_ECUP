"""Expand a declarative sweep into immutable candidate configs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.sweep import expand_sweep
from src.reference_framework_v1.search import generate_individual_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.sweep.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "search_id" in raw:
        paths = generate_individual_search(args.sweep, args.output_dir)
    else:
        paths = expand_sweep(args.sweep, args.output_dir)
    for path in paths:
        print(path.as_posix(), flush=True)


if __name__ == "__main__":
    main()
