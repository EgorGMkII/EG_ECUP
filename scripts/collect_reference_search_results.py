"""Collect immutable individual, gate, and full-validation reports into JSON/CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.selection import collect_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True, nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    rows = collect_results(args.roots)
    args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    pl.DataFrame(rows).write_csv(args.output_csv)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
