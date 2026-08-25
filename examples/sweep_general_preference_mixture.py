"""Sweep mixture weights and temperatures using cached validation scores."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from nashrs import aggregate_diagnostics, diagnose_score_mixture


def comma_floats(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",")]
    if not parsed or any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive floats")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--first-component-weights",
        type=comma_floats,
        default=comma_floats("0.2,0.3,0.4,0.5,0.6,0.7,0.8"),
    )
    parser.add_argument(
        "--temperature-multipliers",
        type=comma_floats,
        default=comma_floats("0.25,0.5,1,2,4"),
    )
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def evaluate_candidate(
    records: list[dict],
    weights: list[float],
    temperatures: list[float],
) -> dict:
    per_prompt = [
        diagnose_score_mixture(
            record["component_scores"], weights, temperatures
        )
        for record in records
    ]
    return {
        "weights": weights,
        "temperatures": temperatures,
        **aggregate_diagnostics(per_prompt),
    }


def rank_key(candidate: dict) -> tuple[float, float, float]:
    # Prefer observed intransitivity, then departure from the best scalar BTL
    # fit. The final term favors balanced weights when diagnostics tie.
    return (
        candidate["cycle_rate"],
        candidate["bt_probability_rmse"],
        -abs(candidate["weights"][0] - 0.5),
    )


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text())
    records = source["per_prompt"]
    base_temperatures = [float(value) for value in source["effective_temperatures"]]
    if len(base_temperatures) != 2:
        raise ValueError("the initial sweep supports exactly two components")

    candidates = []
    for first_weight, first_multiplier, second_multiplier in itertools.product(
        args.first_component_weights,
        args.temperature_multipliers,
        args.temperature_multipliers,
    ):
        if first_weight >= 1.0:
            raise ValueError("first-component weights must be below one")
        candidates.append(
            evaluate_candidate(
                records,
                [first_weight, 1.0 - first_weight],
                [
                    base_temperatures[0] * first_multiplier,
                    base_temperatures[1] * second_multiplier,
                ],
            )
        )
    candidates.sort(key=rank_key, reverse=True)
    result = {
        "source": str(args.input),
        "candidates_evaluated": len(candidates),
        "selection_data_only": True,
        "requires_independent_validation": True,
        "recommended": candidates[0],
        "top_candidates": candidates[: args.top_k],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
