"""Nash-RS controller for the shared PPO method interface."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..interfaces import PreferenceOracle, TextSampler
from ..nash_rs import NashRSConfig, NashRSRewardConstructor
from ..types import RewardBatch
from .base import NoOpHooks, PhaseSpec


class NashRSPPOController(NoOpHooks):
    name = "nash_rs"

    def __init__(self, config, oracle, reference_policy) -> None:
        self.config: NashRSConfig = config
        self.oracle: PreferenceOracle = oracle
        self.reference_policy: TextSampler = reference_policy
        self.calls = 0

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [PhaseSpec("main", kl_reference=self.reference_policy)]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        if phase != "main":
            raise ValueError("Nash-RS only supports the main phase")
        # Advance the seed per rollout while retaining full reproducibility.
        config = replace(self.config, seed=self.config.seed + self.calls)
        self.calls += 1
        return NashRSRewardConstructor(
            config,
            policy_sampler=current_policy,
            reference_sampler=self.reference_policy,
            preference_oracle=self.oracle,
        ).construct(prompts, responses)

