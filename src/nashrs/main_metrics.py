"""Common artifact contract for heterogeneous method-native trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .main_suite import MAIN_SUITE, get_main_method, validate_smoke_steps


REQUIRED_STEP_FIELDS = (
    "step",
    "loss",
    "learning_rate",
    "preference_model_calls",
    "generated_tokens",
    "gpu_hours",
)


def validate_step_records(records: Iterable[dict], *, expected_steps: int) -> list[dict]:
    expected_steps = validate_smoke_steps(expected_steps)
    values = list(records)
    if len(values) != expected_steps:
        raise ValueError(f"expected {expected_steps} step records, found {len(values)}")
    for expected, record in enumerate(values, start=1):
        missing = [field for field in REQUIRED_STEP_FIELDS if field not in record]
        if missing:
            raise ValueError(f"step {expected} is missing fields: {', '.join(missing)}")
        if int(record["step"]) != expected:
            raise ValueError(f"non-contiguous step sequence at record {expected}")
    return values


def write_smoke_manifest(
    output_path: Path,
    *,
    method: str,
    records: Iterable[dict],
    expected_steps: int,
    parameter_update: dict,
) -> dict:
    """Write one machine-checkable success manifest for a Main smoke run."""

    spec = get_main_method(method)
    values = validate_step_records(records, expected_steps=expected_steps)
    changed = int(parameter_update.get("changed_elements", 0))
    update_norm = float(parameter_update.get("l2_update_norm", 0.0))
    if changed <= 0 or update_norm <= 0:
        raise ValueError("smoke checkpoint did not change from the base model")
    manifest = {
        "suite": MAIN_SUITE,
        "experiment_id": spec.experiment_id,
        "method": method,
        "backend": spec.backend,
        "fidelity": spec.fidelity,
        "optimizer_steps": expected_steps,
        "parameter_update_verified": True,
        "parameter_update": parameter_update,
        "metrics_fields": list(REQUIRED_STEP_FIELDS),
        "final_step": values[-1],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
