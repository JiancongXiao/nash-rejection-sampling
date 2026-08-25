"""Aggregate Nash-RS fixed-test results across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nashrs.seed_aggregation import aggregate_seed_runs, load_seed_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--candidate", default="step_24")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [load_seed_run(path, args.baseline, args.candidate) for path in args.runs]
    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        **aggregate_seed_runs(runs, args.bootstrap_samples, args.bootstrap_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **result}, indent=2))


if __name__ == "__main__":
    main()
