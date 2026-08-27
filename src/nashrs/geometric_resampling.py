"""Finite-pool importance resampling for a geometric policy mixture."""

from __future__ import annotations

import math
import random
from typing import Sequence


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def geometric_importance_weights(
    current_log_probs: Sequence[float],
    reference_log_probs: Sequence[float],
    reference_weight: float,
) -> list[float]:
    """Return normalized weights for q ∝ pi^(1-a) mu^a.

    Candidates are assumed to be drawn equally from ``pi`` and ``mu``.  The
    denominator is therefore the arithmetic proposal mixture
    ``0.5 * pi + 0.5 * mu``; only the target distribution is geometric.
    """

    if len(current_log_probs) != len(reference_log_probs):
        raise ValueError("current and reference log-probabilities must align")
    if not current_log_probs:
        raise ValueError("at least one candidate is required")
    if not 0.0 <= reference_weight <= 1.0:
        raise ValueError("reference_weight must lie in [0, 1]")

    log_weights = []
    for current, reference in zip(current_log_probs, reference_log_probs):
        proposal = _logaddexp(current, reference) - math.log(2.0)
        target = (1.0 - reference_weight) * current + reference_weight * reference
        log_weights.append(target - proposal)
    maximum = max(log_weights)
    unnormalized = [math.exp(value - maximum) for value in log_weights]
    total = sum(unnormalized)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("geometric importance weights are not finite")
    return [value / total for value in unnormalized]


def effective_sample_size(weights: Sequence[float]) -> float:
    if not weights or any(value < 0.0 for value in weights):
        raise ValueError("weights must be non-negative and non-empty")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    normalized = [value / total for value in weights]
    return 1.0 / sum(value * value for value in normalized)


def resample_indices(
    weights: Sequence[float], count: int, rng: random.Random
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    return list(rng.choices(range(len(weights)), weights=weights, k=count))
