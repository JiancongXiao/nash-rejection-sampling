"""Validate native metrics and parameter changes after a Main smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nashrs.main_metrics import write_smoke_manifest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_record(record: dict) -> dict:
    """Map trainer-specific log names onto the Main smoke metric contract."""

    value = dict(record)
    aliases = {
        "loss": ("policy_loss", "train_loss"),
        "learning_rate": ("actor_lr", "lr"),
        "preference_model_calls": (
            "comparison/preference_model_calls",
            "nashrs/preference_model_calls",
        ),
        "generated_tokens": (
            "comparison/generated_tokens",
            "nashrs/generated_tokens",
        ),
        "gpu_hours": ("comparison/gpu_hours", "nashrs/gpu_hours"),
    }
    for destination, sources in aliases.items():
        if destination in value:
            continue
        for source in sources:
            if source in value:
                value[destination] = value[source]
                break
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--parameter-update", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = write_smoke_manifest(
        args.output,
        method=args.method,
        records=[normalize_record(record) for record in read_jsonl(args.metrics)],
        expected_steps=args.steps,
        parameter_update=json.loads(args.parameter_update.read_text()),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
