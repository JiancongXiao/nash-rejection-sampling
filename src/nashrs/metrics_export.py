"""Export structured step metrics from OpenRLHF's textual training log."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable


STEP_MARKER = "✨ Global step "
CUMULATIVE_FIELDS = (
    "nashrs/preference_model_calls",
    "nashrs/generated_tokens",
    "nashrs/proposals",
    "nashrs/accepted",
    "nashrs/gpu_hours",
)


def parse_step_lines(lines: Iterable[str]) -> list[dict]:
    """Parse OpenRLHF global-step dictionaries and add cumulative costs."""

    records = []
    cumulative = {field: 0.0 for field in CUMULATIVE_FIELDS}
    for line in lines:
        if STEP_MARKER not in line:
            continue
        suffix = line.split(STEP_MARKER, maxsplit=1)[1]
        step_text, payload = suffix.split(":", maxsplit=1)
        metrics = ast.literal_eval(payload.strip())
        if not isinstance(metrics, dict):
            raise ValueError("OpenRLHF step payload is not a dictionary")
        record = {"step": int(step_text.strip()), **metrics}
        for field in CUMULATIVE_FIELDS:
            cumulative[field] += float(metrics.get(field, 0.0))
            suffix_name = field.split("/", maxsplit=1)[-1]
            record[f"nashrs/cumulative_{suffix_name}"] = cumulative[field]
        records.append(record)
    return records


def export_step_metrics(log_path: Path, output_path: Path) -> list[dict]:
    records = parse_step_lines(log_path.read_text(errors="replace").splitlines())
    if not records:
        raise ValueError(f"No OpenRLHF global-step metrics found in {log_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records
