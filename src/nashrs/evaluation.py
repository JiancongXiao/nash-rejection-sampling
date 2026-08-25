"""Game-theoretic evaluation shared by every method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .interfaces import PreferenceOracle


@dataclass(frozen=True)
class PairwiseEvaluation:
    methods: Sequence[str]
    matrix: Sequence[Sequence[float]]
    preference_model_calls: int
    per_prompt: Sequence[Sequence[Sequence[float]]] = ()

    def average_win_rate(self, method: str) -> float:
        index = self.methods.index(method)
        opponents = [value for j, value in enumerate(self.matrix[index]) if j != index]
        return sum(opponents) / len(opponents) if opponents else 0.5

    def worst_case_win_rate(self, method: str) -> float:
        index = self.methods.index(method)
        opponents = [value for j, value in enumerate(self.matrix[index]) if j != index]
        return min(opponents) if opponents else 0.5

    def empirical_exploitability(self, method: str) -> float:
        """Best observed opponent advantage over the evaluated policy."""
        index = self.methods.index(method)
        best_opponent_win_rate = max(
            self.matrix[j][index] for j in range(len(self.methods)) if j != index
        ) if len(self.methods) > 1 else 0.5
        return max(0.0, best_opponent_win_rate - 0.5)


def evaluate_pairwise(
    prompts: Sequence[str],
    responses: Mapping[str, Sequence[str]],
    oracle: PreferenceOracle,
) -> PairwiseEvaluation:
    """Build a pairwise matrix using one query per unordered response pair.

    The reverse entry is `1-p`, consistent with the constant-sum preference
    model assumed by NLHF. Every method must provide one response per prompt.
    """
    methods = list(responses)
    for method, values in responses.items():
        if len(values) != len(prompts):
            raise ValueError(f"{method} has {len(values)} responses for {len(prompts)} prompts")
    size = len(methods)
    matrix = [[0.5 for _ in methods] for _ in methods]
    per_prompt = [
        [[0.5 for _ in prompts] for _ in methods]
        for _ in methods
    ]
    calls = 0
    for i in range(size):
        for j in range(i + 1, size):
            values = [
                float(value)
                for value in oracle.compare(prompts, responses[methods[i]], responses[methods[j]])
            ]
            if len(values) != len(prompts):
                raise ValueError("preference oracle returned the wrong evaluation batch size")
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError("preference oracle returned an invalid probability")
            win_rate = sum(values) / len(values) if values else 0.5
            matrix[i][j] = win_rate
            matrix[j][i] = 1.0 - win_rate
            per_prompt[i][j] = values
            per_prompt[j][i] = [1.0 - value for value in values]
            calls += len(values)
    return PairwiseEvaluation(methods, matrix, calls, per_prompt)
