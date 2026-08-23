"""Magnetic Preference Optimization adapted to a shared PPO inner step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..interfaces import PreferenceOracle, TextSampler
from ..types import RewardBatch
from .base import NoOpHooks, PhaseSpec, SnapshotFn, preference_reward_batch


@dataclass(frozen=True)
class MPOPPOConfig:
    update_interval: int = 8
    opponents_per_prompt: int = 1
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")


class MPOPPOController(NoOpHooks):
    """Fixed magnet/opponent within an interval; synchronize both periodically."""

    name = "mpo_ppo"

    def __init__(self, config, oracle, initial_policy, snapshot) -> None:
        self.config: MPOPPOConfig = config
        self.oracle: PreferenceOracle = oracle
        self.snapshot: SnapshotFn = snapshot
        self.magnet_policy: TextSampler = snapshot(initial_policy, "mpo-magnet-0")
        self.opponent_policy: TextSampler = self.magnet_policy

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [PhaseSpec("main", kl_reference=self.magnet_policy)]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        del current_policy
        if phase != "main":
            raise ValueError("MPO PPO only supports the main phase")
        return preference_reward_batch(
            prompts,
            responses,
            self.opponent_policy,
            self.oracle,
            self.config.opponents_per_prompt,
            self.config.reward_scale,
            method_name=self.name,
        )

    def after_iteration(self, iteration: int, updated_policy: TextSampler) -> None:
        if (iteration + 1) % self.config.update_interval == 0:
            snapshot = self.snapshot(updated_policy, f"mpo-magnet-{iteration + 1}")
            self.magnet_policy = snapshot
            self.opponent_policy = snapshot

