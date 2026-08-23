"""Shared interfaces; alternative methods should implement the same boundary."""

from __future__ import annotations

from typing import Protocol, Sequence

from .types import RewardBatch


class TextSampler(Protocol):
    """Generate `n` responses for one prompt."""

    def sample(self, prompt: str, n: int) -> Sequence[str]: ...


class PreferenceOracle(Protocol):
    """Return P(left beats right | prompt), one value per comparison."""

    def compare(
        self,
        prompts: Sequence[str],
        left: Sequence[str],
        right: Sequence[str],
    ) -> Sequence[float]: ...


class RewardConstructor(Protocol):
    """The only method-specific boundary used by the shared PPO backbone."""

    def construct(self, prompts: Sequence[str], responses: Sequence[str]) -> RewardBatch: ...


class ScalarRewardOracle(Protocol):
    """Return one scalar reward for each prompt/response pair."""

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> Sequence[float]: ...


class GeometricMixtureSampler(Protocol):
    """Sample from a token- or sequence-level geometric policy mixture."""

    def sample_geometric(
        self,
        prompt: str,
        n: int,
        current: TextSampler,
        reference: TextSampler,
        reference_weight: float,
    ) -> Sequence[str]: ...
