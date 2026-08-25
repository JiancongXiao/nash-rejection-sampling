"""Diagnostics for determining whether a reward mixture is meaningfully non-BT."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Sequence

from .preference_oracles import _sigmoid


def _logit(probability: float, epsilon: float = 1e-6) -> float:
    probability = min(1.0 - epsilon, max(epsilon, probability))
    return math.log(probability / (1.0 - probability))


def mixture_probability(
    component_scores: Sequence[Sequence[float]],
    left: int,
    right: int,
    weights: Sequence[float],
    temperatures: Sequence[float],
) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0 or len(weights) != len(component_scores):
        raise ValueError("component weights must match scores and have positive sum")
    if len(temperatures) != len(component_scores) or any(t <= 0 for t in temperatures):
        raise ValueError("component temperatures must match scores and be positive")
    return sum(
        weight * _sigmoid((scores[left] - scores[right]) / temperature)
        for scores, weight, temperature in zip(
            component_scores, weights, temperatures
        )
    ) / total_weight


def diagnose_score_mixture(
    component_scores: Sequence[Sequence[float]],
    weights: Sequence[float],
    temperatures: Sequence[float],
) -> dict[str, float | int]:
    """Diagnose one prompt's response pool using every pair and triple."""

    if len(component_scores) < 2:
        raise ValueError("general-preference diagnostics require >=2 components")
    response_count = len(component_scores[0])
    if response_count < 3 or any(len(scores) != response_count for scores in component_scores):
        raise ValueError("each component needs scores for at least three responses")

    disagreement_sum = 0.0
    disagreement_denominator = 0
    pair_logits = [[0.0] * response_count for _ in range(response_count)]
    pair_probabilities = [[0.5] * response_count for _ in range(response_count)]
    for left, right in combinations(range(response_count), 2):
        signs = [
            1 if scores[left] > scores[right] else -1 if scores[left] < scores[right] else 0
            for scores in component_scores
        ]
        for first, second in combinations(signs, 2):
            disagreement_sum += float(first * second < 0)
            disagreement_denominator += 1
        probability = mixture_probability(
            component_scores, left, right, weights, temperatures
        )
        pair_probabilities[left][right] = probability
        pair_probabilities[right][left] = 1.0 - probability
        pair_logits[left][right] = _logit(probability)
        pair_logits[right][left] = -pair_logits[left][right]

    # Closed-form least-squares scores for a complete comparison graph.
    fitted_scores = [sum(row) / response_count for row in pair_logits]
    squared_logit_error = 0.0
    squared_probability_error = 0.0
    pair_count = 0
    for left, right in combinations(range(response_count), 2):
        fitted_logit = fitted_scores[left] - fitted_scores[right]
        squared_logit_error += (pair_logits[left][right] - fitted_logit) ** 2
        squared_probability_error += (
            pair_probabilities[left][right] - _sigmoid(fitted_logit)
        ) ** 2
        pair_count += 1

    cycles = 0
    triple_count = 0
    for first, second, third in combinations(range(response_count), 3):
        first_second = pair_probabilities[first][second] > 0.5
        second_third = pair_probabilities[second][third] > 0.5
        third_first = pair_probabilities[third][first] > 0.5
        cycles += int(
            (first_second and second_third and third_first)
            or (not first_second and not second_third and not third_first)
        )
        triple_count += 1

    return {
        "responses": response_count,
        "pairs": pair_count,
        "triples": triple_count,
        "component_disagreement_rate": disagreement_sum
        / disagreement_denominator,
        "cycle_rate": cycles / triple_count,
        "bt_logit_rmse": math.sqrt(squared_logit_error / pair_count),
        "bt_probability_rmse": math.sqrt(squared_probability_error / pair_count),
    }


def aggregate_diagnostics(records: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("at least one diagnostic record is required")
    keys = (
        "component_disagreement_rate",
        "cycle_rate",
        "bt_logit_rmse",
        "bt_probability_rmse",
    )
    return {
        "prompts": len(records),
        **{
            key: sum(float(record[key]) for record in records) / len(records)
            for key in keys
        },
    }
