"""Build deterministic nested cohorts for a named reference profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_framework_v1.cohorts import build_nested_cohorts
from src.reference_framework_v1.profiles import get_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="POST_NY_PUBLIC_PROXY")
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--sample-submit", type=Path, default=Path("sample_submit.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/reference_v1/cohorts/POST_NY_PUBLIC_PROXY"))
    parser.add_argument("--root-seed", type=int, default=42)
    args = parser.parse_args()
    build_nested_cohorts(profile=get_profile(args.profile), train_path=args.train, sample_submit_path=args.sample_submit, output_root=args.output_root, root_seed=args.root_seed)


if __name__ == "__main__":
    main()
