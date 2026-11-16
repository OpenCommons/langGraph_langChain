#!/usr/bin/env python3
"""Run the golden dataset through an evaluation policy; exit non-zero if the gate fails.

python3 scripts/run_evals.py [--policy oee] [--cases data/evals/golden.json]
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "services" / "api"))

from evals.harness import evaluate, load_cases  # noqa: E402
from evals.policy import load_policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="oee")
    ap.add_argument("--cases", type=pathlib.Path, default=None)
    args = ap.parse_args()
    report = evaluate(load_policy(args.policy), load_cases(args.cases))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
