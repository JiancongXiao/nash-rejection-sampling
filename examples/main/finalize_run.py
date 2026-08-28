"""Validate and finalize a non-smoke method-native Main run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.main.finalize_smoke import normalize_record, read_jsonl
from nashrs.main_metrics import write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-kind", choices=("pilot", "full"), required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--parameter-update", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = write_run_manifest(
        args.output,
        method=args.method,
        run_kind=args.run_kind,
        records=[
            normalize_record(record, method=args.method)
            for record in read_jsonl(args.metrics)
        ],
        expected_steps=args.steps,
        parameter_update=json.loads(args.parameter_update.read_text()),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
