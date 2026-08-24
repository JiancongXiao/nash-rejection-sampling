"""Data passed between method-specific reward construction and PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass
class Accounting:
    preference_model_calls: int = 0
    policy_generations: int = 0
    reference_generations: int = 0
    opponent_generations: int = 0
    generated_tokens: int = 0
    proposals: int = 0
    acceptance_trials: int = 0
    accepted: int = 0
    wall_time_seconds: float = 0.0
    gpu_hours: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.acceptance_trials if self.acceptance_trials else 0.0

    @property
    def proposal_efficiency(self) -> float:
        """Accepted opponents per generated reference proposal."""

        return self.accepted / self.proposals if self.proposals else 0.0

    def to_logs(self, prefix: str = "nashrs") -> dict[str, float]:
        values = asdict(self)
        values["acceptance_rate"] = self.acceptance_rate
        values["proposal_efficiency"] = self.proposal_efficiency
        return {f"{prefix}/{key}": float(value) for key, value in values.items()}


@dataclass(frozen=True)
class OpponentSample:
    prompt: str
    response: str
    g_hat: Optional[float] = None
    acceptance_probability: Optional[float] = None


@dataclass
class RewardBatch:
    rewards: Sequence[float]
    scores: Sequence[float]
    opponents: Sequence[Sequence[OpponentSample]]
    accounting: Accounting
    diagnostics: Mapping[str, float] = field(default_factory=dict)
