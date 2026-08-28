"""Validate native metrics and parameter changes after a Main smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nashrs.main_metrics import write_smoke_manifest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_record(record: dict, method: str | None = None) -> dict:
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
    # OpenRLHF agent metrics count method-specific auxiliary generations.  The
    # actor rollout itself is logged separately as the mean response length;
    # include it here so Main's generated-token cost is comparable with native
    # trainers, which already count every generated completion.
    if method in {"reward_ppo", "nash_rs"} and "response_length" in value:
        samples = float(value.get("nashrs/samples_in_step", 1.0))
        value["generated_tokens"] = float(value.get("generated_tokens", 0.0)) + (
            float(value["response_length"]) * samples
        )
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
