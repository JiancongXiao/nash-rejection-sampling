"""Export structured step metrics from OpenRLHF's textual training log."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable


STEP_MARKER = "✨ Global step "
COST_FIELDS = (
    "nashrs/preference_model_calls",
    "nashrs/generated_tokens",
    "nashrs/proposals",
    "nashrs/acceptance_trials",
    "nashrs/accepted",
)
COMMON_COST_SUFFIXES = (
    "preference_model_calls",
    "generated_tokens",
    "opponent_generations",
)


def parse_step_lines(lines: Iterable[str], samples_per_step: int = 1) -> list[dict]:
    """Parse global-step dictionaries and recover totals from batch means."""

    if samples_per_step <= 0:
        raise ValueError("samples_per_step must be positive")
    records = []
    cumulative = {field: 0.0 for field in COST_FIELDS}
    cumulative_sample_gpu_hours = 0.0
    experiment_cumulative = {suffix: 0.0 for suffix in COMMON_COST_SUFFIXES}
    experiment_cumulative_gpu_hours = 0.0
    for line in lines:
        if STEP_MARKER not in line:
            continue
        suffix = line.split(STEP_MARKER, maxsplit=1)[1]
        step_text, payload = suffix.split(":", maxsplit=1)
        metrics = ast.literal_eval(payload.strip())
        if not isinstance(metrics, dict):
            raise ValueError("OpenRLHF step payload is not a dictionary")
        record = {"step": int(step_text.strip()), **metrics}
        record["nashrs/samples_in_step"] = samples_per_step
        for field in COST_FIELDS:
            step_total = float(metrics.get(field, 0.0)) * samples_per_step
            cumulative[field] += step_total
            suffix_name = field.split("/", maxsplit=1)[-1]
            record[f"nashrs/step_total_{suffix_name}"] = step_total
            record[f"nashrs/cumulative_total_{suffix_name}"] = cumulative[field]
        step_accepted = record.get("nashrs/step_total_accepted", 0.0)
        step_trials = record.get("nashrs/step_total_acceptance_trials", 0.0)
        step_proposals = record.get("nashrs/step_total_proposals", 0.0)
        record["nashrs/aggregate_acceptance_rate"] = (
            step_accepted / step_trials if step_trials else 0.0
        )
        record["nashrs/aggregate_proposal_efficiency"] = (
            step_accepted / step_proposals if step_proposals else 0.0
        )
        cumulative_accepted = cumulative["nashrs/accepted"]
        cumulative_trials = cumulative["nashrs/acceptance_trials"]
        cumulative_proposals = cumulative["nashrs/proposals"]
        record["nashrs/cumulative_acceptance_rate"] = (
            cumulative_accepted / cumulative_trials if cumulative_trials else 0.0
        )
        record["nashrs/cumulative_proposal_efficiency"] = (
            cumulative_accepted / cumulative_proposals if cumulative_proposals else 0.0
        )
        # Agent GPU-hours are also batch-averaged, but executions may overlap.
        # Their sum measures sample work, not PBS allocated wall-clock GPU-hours.
        step_sample_gpu_hours = (
            float(metrics.get("nashrs/gpu_hours", 0.0)) * samples_per_step
        )
        cumulative_sample_gpu_hours += step_sample_gpu_hours
        record["nashrs/step_sum_sample_gpu_hours"] = step_sample_gpu_hours
        record["nashrs/cumulative_sum_sample_gpu_hours"] = (
            cumulative_sample_gpu_hours
        )
        for suffix_name in COMMON_COST_SUFFIXES:
            source_key = f"comparison/{suffix_name}"
            if source_key not in metrics:
                source_key = f"nashrs/{suffix_name}"
            step_total = float(metrics.get(source_key, 0.0)) * samples_per_step
            experiment_cumulative[suffix_name] += step_total
            record[f"experiment/step_total_{suffix_name}"] = step_total
            record[f"experiment/cumulative_total_{suffix_name}"] = (
                experiment_cumulative[suffix_name]
            )
        gpu_source = (
            "comparison/gpu_hours"
            if "comparison/gpu_hours" in metrics
            else "nashrs/gpu_hours"
        )
        experiment_step_gpu_hours = (
            float(metrics.get(gpu_source, 0.0)) * samples_per_step
        )
        experiment_cumulative_gpu_hours += experiment_step_gpu_hours
        record["experiment/step_sum_sample_gpu_hours"] = experiment_step_gpu_hours
        record["experiment/cumulative_sum_sample_gpu_hours"] = (
            experiment_cumulative_gpu_hours
        )
        records.append(record)
    return records


def export_step_metrics(
    log_path: Path, output_path: Path, samples_per_step: int = 1
) -> list[dict]:
    records = parse_step_lines(
        log_path.read_text(errors="replace").splitlines(), samples_per_step
    )
    if not records:
        raise ValueError(f"No OpenRLHF global-step metrics found in {log_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records
