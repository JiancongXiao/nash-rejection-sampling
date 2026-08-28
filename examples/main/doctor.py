"""Inspect Main-suite source pins and print launch metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from nashrs.main_suite import (
    MAIN_METHOD_SPECS,
    required_sources,
    validate_source_checkout,
)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    source_status = []
    failures = []
    for source in required_sources():
        checkout = args.source_root / source.subdirectory
        problems = validate_source_checkout(args.source_root, source)
        revision = git_revision(checkout) if not problems else None
        if revision is not None and revision != source.revision:
            problems.append(
                f"revision mismatch: expected {source.revision}, found {revision}"
            )
        failures.extend(problems)
        source_status.append(
            {
                "name": source.name,
                "path": str(checkout),
                "expected_revision": source.revision,
                "actual_revision": revision,
                "ok": not problems,
                "problems": problems,
            }
        )

    payload = {
        "suite": "main",
        "methods": [
            {
                "name": spec.name,
                "experiment_id": spec.experiment_id,
                "backend": spec.backend,
                "entrypoint": spec.entrypoint,
                "smoke_steps": spec.smoke_steps,
                "min_gpus": spec.min_gpus,
            }
            for spec in MAIN_METHOD_SPECS
        ],
        "sources": source_status,
        "ok": not failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and failures:
        raise SystemExit("Main source doctor failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
