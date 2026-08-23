"""Shared controller boundary between method logic and OpenRLHF PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from ..interfaces import PreferenceOracle, TextSampler
from ..types import Accounting, OpponentSample, RewardBatch


SnapshotFn = Callable[[TextSampler, str], TextSampler]


@dataclass(frozen=True)
class PhaseSpec:
    """Instructions the shared trainer must honor for one PPO phase."""

    name: str
    start_from: str = "current"
    kl_reference: Optional[TextSampler] = None
    metadata: Optional[dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.start_from not in {"current", "iteration_start"}:
            raise ValueError("start_from must be current or iteration_start")


class MethodController(Protocol):
    name: str

    def phase_specs(self) -> Sequence[PhaseSpec]: ...

    def construct(
        self,
        phase: str,
        prompts: Sequence[str],
        responses: Sequence[str],
        current_policy: TextSampler,
    ) -> RewardBatch: ...

    def after_phase(
        self, phase: str, iteration: int, updated_policy: TextSampler
    ) -> None: ...

    def after_iteration(self, iteration: int, updated_policy: TextSampler) -> None: ...


def preference_reward_batch(
    prompts: Sequence[str],
    responses: Sequence[str],
    opponent_sampler: TextSampler,
    oracle: PreferenceOracle,
    opponents_per_prompt: int,
    reward_scale: float = 1.0,
    baseline: float = 0.5,
    method_name: str = "preference_ppo",
) -> RewardBatch:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal length")
    if opponents_per_prompt <= 0:
        raise ValueError("opponents_per_prompt must be positive")

    unique_opponents: dict[str, list[str]] = {}
    accounting = Accounting()
    for prompt in dict.fromkeys(prompts):
        values = list(opponent_sampler.sample(prompt, opponents_per_prompt))
        if len(values) != opponents_per_prompt:
            raise ValueError("opponent sampler returned the wrong number of samples")
        unique_opponents[prompt] = values
        accounting.opponent_generations += len(values)
        accounting.generated_tokens += sum(len(value.split()) for value in values)

    rewards: list[float] = []
    opponents_out: list[list[OpponentSample]] = []
    for prompt, response in zip(prompts, responses):
        opponents = unique_opponents[prompt]
        probabilities = [
            float(value)
            for value in oracle.compare(
                [prompt] * len(opponents),
                [response] * len(opponents),
                opponents,
            )
        ]
        if len(probabilities) != len(opponents):
            raise ValueError("preference oracle returned the wrong batch size")
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("preference oracle returned an invalid probability")
        accounting.preference_model_calls += len(probabilities)
        mean_probability = sum(probabilities) / len(probabilities)
        rewards.append(reward_scale * (mean_probability - baseline))
        opponents_out.append(
            [OpponentSample(prompt=prompt, response=value) for value in opponents]
        )

    return RewardBatch(
        rewards=rewards,
        scores=[min(1.0, max(0.0, reward / reward_scale + baseline)) for reward in rewards],
        opponents=opponents_out,
        accounting=accounting,
        diagnostics={"method/reward_mean": sum(rewards) / max(1, len(rewards))},
    )


class NoOpHooks:
    def after_phase(self, phase: str, iteration: int, updated_policy: TextSampler) -> None:
        del phase, iteration, updated_policy

    def after_iteration(self, iteration: int, updated_policy: TextSampler) -> None:
        del iteration, updated_policy
