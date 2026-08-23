"""Conventional scalar-reward PPO baseline."""

from __future__ import annotations

from typing import Sequence

from ..interfaces import ScalarRewardOracle, TextSampler
from ..types import Accounting, RewardBatch
from .base import NoOpHooks, PhaseSpec


class RewardPPOController(NoOpHooks):
    name = "reward_ppo"

    def __init__(self, oracle: ScalarRewardOracle, reference_policy: TextSampler) -> None:
        self.oracle = oracle
        self.reference_policy = reference_policy

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [PhaseSpec("main", kl_reference=self.reference_policy)]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        del current_policy
        if phase != "main":
            raise ValueError("reward PPO only supports the main phase")
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")
        rewards = [float(value) for value in self.oracle.score(prompts, responses)]
        if len(rewards) != len(responses):
            raise ValueError("reward oracle returned the wrong batch size")
        return RewardBatch(
            rewards=rewards,
            scores=[min(1.0, max(0.0, value)) for value in rewards],
            opponents=[[] for _ in responses],
            accounting=Accounting(),
            diagnostics={"method/reward_mean": sum(rewards) / max(1, len(rewards))},
        )
