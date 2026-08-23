"""COMAL conceptual-prox outer loop with a shared PPO inner solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..interfaces import PreferenceOracle, TextSampler
from ..types import RewardBatch
from .base import NoOpHooks, PhaseSpec, SnapshotFn, preference_reward_batch


@dataclass(frozen=True)
class COMALPPOConfig:
    inner_iterations: int = 8
    opponents_per_prompt: int = 1
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.inner_iterations <= 0:
            raise ValueError("inner_iterations must be positive")


class COMALPPOController(NoOpHooks):
    """Self-play inner game regularized to a fixed outer-loop anchor."""

    name = "comal_ppo"

    def __init__(self, config, oracle, initial_policy, snapshot) -> None:
        self.config: COMALPPOConfig = config
        self.oracle: PreferenceOracle = oracle
        self.snapshot: SnapshotFn = snapshot
        self.anchor_policy: TextSampler = snapshot(initial_policy, "comal-anchor-0")
        self.outer_round = 0

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [
            PhaseSpec(
                "main",
                kl_reference=self.anchor_policy,
                metadata={"outer_round": float(self.outer_round)},
            )
        ]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        if phase != "main":
            raise ValueError("COMAL PPO only supports the main phase")
        return preference_reward_batch(
            prompts,
            responses,
            current_policy,
            self.oracle,
            self.config.opponents_per_prompt,
            self.config.reward_scale,
            method_name=self.name,
        )

    def after_iteration(self, iteration: int, updated_policy: TextSampler) -> None:
        if (iteration + 1) % self.config.inner_iterations == 0:
            self.outer_round += 1
            self.anchor_policy = self.snapshot(
                updated_policy, f"comal-anchor-{self.outer_round}"
            )

