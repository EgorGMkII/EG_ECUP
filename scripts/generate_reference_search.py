"""Generate immutable individual-screen configs from a one-model search plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.search import generate_individual_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in generate_individual_search(args.plan, args.output_dir):
        print(path.as_posix(), flush=True)


if __name__ == "__main__":
    main()
