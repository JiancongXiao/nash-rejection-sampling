"""Self-play and Nash-MD PPO reward/opponent construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..interfaces import GeometricMixtureSampler, PreferenceOracle, TextSampler
from ..types import RewardBatch
from .base import NoOpHooks, PhaseSpec, preference_reward_batch


@dataclass(frozen=True)
class SelfPlayPPOConfig:
    opponents_per_prompt: int = 1
    reward_scale: float = 1.0


class SelfPlayPPOController(NoOpHooks):
    name = "self_play_ppo"

    def __init__(self, config: SelfPlayPPOConfig, oracle: PreferenceOracle, reference_policy: TextSampler) -> None:
        self.config = config
        self.oracle = oracle
        self.reference_policy = reference_policy

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [PhaseSpec("main", kl_reference=self.reference_policy)]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        if phase != "main":
            raise ValueError("self-play only supports the main phase")
        return preference_reward_batch(
            prompts,
            responses,
            current_policy,
            self.oracle,
            self.config.opponents_per_prompt,
            self.config.reward_scale,
            method_name=self.name,
        )


@dataclass(frozen=True)
class NashMDPPOConfig:
    reference_weight: float = 0.25
    opponents_per_prompt: int = 1
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.reference_weight <= 1.0:
            raise ValueError("reference_weight must lie in [0, 1]")


class _NashMDMixtureOpponent:
    def __init__(self, mixture, current, reference, weight) -> None:
        self.mixture = mixture
        self.current = current
        self.reference = reference
        self.weight = weight

    def sample(self, prompt: str, n: int):
        return self.mixture.sample_geometric(
            prompt, n, self.current, self.reference, self.weight
        )


class NashMDPPOController(NoOpHooks):
    """Practical Nash-MD-PG/PPO adaptation with a geometric opponent."""

    name = "nash_md_ppo"

    def __init__(self, config, oracle, reference_policy, mixture_sampler) -> None:
        self.config: NashMDPPOConfig = config
        self.oracle: PreferenceOracle = oracle
        self.reference_policy: TextSampler = reference_policy
        self.mixture_sampler: GeometricMixtureSampler = mixture_sampler

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [
            PhaseSpec(
                "main",
                kl_reference=self.reference_policy,
                metadata={"reference_weight": self.config.reference_weight},
            )
        ]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        if phase != "main":
            raise ValueError("Nash-MD PPO only supports the main phase")
        opponent = _NashMDMixtureOpponent(
            self.mixture_sampler,
            current_policy,
            self.reference_policy,
            self.config.reference_weight,
        )
        return preference_reward_batch(
            prompts,
            responses,
            opponent,
            self.oracle,
            self.config.opponents_per_prompt,
            self.config.reward_scale,
            method_name=self.name,
        )

