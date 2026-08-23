"""Nash-RS Gibbs opponent construction and implicit rewards."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Sequence

from .interfaces import PreferenceOracle, TextSampler
from .rejection import accept_gibbs_proposal, acceptance_probability
from .types import Accounting, OpponentSample, RewardBatch


@dataclass(frozen=True)
class NashRSConfig:
    tau: float
    b1: int
    b2: int
    seed: int = 0
    proposal_batch_size: int = 16
    max_proposals_per_prompt: int = 100_000

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.b1 <= 0 or self.b2 <= 0:
            raise ValueError("b1 and b2 must be positive")
        if self.proposal_batch_size <= 0 or self.max_proposals_per_prompt <= 0:
            raise ValueError("proposal limits must be positive")


def _validate_probabilities(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    for value in result:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"preference oracle returned invalid probability {value}")
    return result


def _token_count(texts: Sequence[str]) -> int:
    # Backend-independent accounting estimate. Production integrations may
    # replace this with tokenizer-exact counts.
    return sum(len(text.split()) for text in texts)


class NashRSRewardConstructor:
    """Construct scalar rewards for a PPO rollout batch.

    A single B1 current-policy sample set is reused for every reference
    proposal associated with the same prompt. Accepted opponents are shared by
    all PPO responses with that prompt, matching the conditional Gibbs target.
    """

    def __init__(
        self,
        config: NashRSConfig,
        policy_sampler: TextSampler,
        reference_sampler: TextSampler,
        preference_oracle: PreferenceOracle,
    ) -> None:
        self.config = config
        self.policy_sampler = policy_sampler
        self.reference_sampler = reference_sampler
        self.preference_oracle = preference_oracle
        self.rng = random.Random(config.seed)

    def construct(self, prompts: Sequence[str], responses: Sequence[str]) -> RewardBatch:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")
        started = time.perf_counter()
        accounting = Accounting()
        opponents_by_prompt: dict[str, list[OpponentSample]] = {}

        # Preserve first-seen order and avoid regenerating opponents when a PPO
        # rollout contains multiple responses for the same prompt.
        for prompt in dict.fromkeys(prompts):
            prompt_proposals = 0
            current = list(self.policy_sampler.sample(prompt, self.config.b1))
            if len(current) != self.config.b1:
                raise ValueError("policy sampler returned the wrong number of samples")
            accounting.policy_generations += len(current)
            accounting.generated_tokens += _token_count(current)
            accepted: list[OpponentSample] = []

            while len(accepted) < self.config.b2:
                remaining_budget = self.config.max_proposals_per_prompt - prompt_proposals
                if remaining_budget <= 0:
                    raise RuntimeError(
                        "Nash-RS proposal budget exhausted; increase tau or "
                        "max_proposals_per_prompt"
                    )
                count = min(self.config.proposal_batch_size, remaining_budget)
                proposals = list(self.reference_sampler.sample(prompt, count))
                if len(proposals) != count:
                    raise ValueError("reference sampler returned the wrong number of samples")
                accounting.reference_generations += count
                accounting.generated_tokens += _token_count(proposals)
                accounting.proposals += count
                prompt_proposals += count

                compare_prompts = [prompt] * (count * self.config.b1)
                left = current * count
                right = [proposal for proposal in proposals for _ in range(self.config.b1)]
                probabilities = _validate_probabilities(
                    self.preference_oracle.compare(compare_prompts, left, right)
                )
                if len(probabilities) != count * self.config.b1:
                    raise ValueError("preference oracle returned the wrong batch size")
                accounting.preference_model_calls += len(probabilities)

                for index, proposal in enumerate(proposals):
                    start = index * self.config.b1
                    g_hat = sum(probabilities[start : start + self.config.b1]) / self.config.b1
                    if accept_gibbs_proposal(g_hat, self.config.tau, self.rng):
                        accepted.append(
                            OpponentSample(
                                prompt=prompt,
                                response=proposal,
                                g_hat=g_hat,
                                acceptance_probability=acceptance_probability(
                                    g_hat, self.config.tau
                                ),
                            )
                        )
                        accounting.accepted += 1
                        if len(accepted) == self.config.b2:
                            break
            opponents_by_prompt[prompt] = accepted

        rewards: list[float] = []
        all_opponents: list[Sequence[OpponentSample]] = []
        for prompt, response in zip(prompts, responses):
            opponents = opponents_by_prompt[prompt]
            probabilities = _validate_probabilities(
                self.preference_oracle.compare(
                    [prompt] * len(opponents),
                    [response] * len(opponents),
                    [opponent.response for opponent in opponents],
                )
            )
            if len(probabilities) != len(opponents):
                raise ValueError("preference oracle returned the wrong reward batch size")
            accounting.preference_model_calls += len(probabilities)
            rewards.append(sum(probabilities) / (self.config.tau * len(probabilities)))
            all_opponents.append(opponents)

        accounting.wall_time_seconds = time.perf_counter() - started
        mean_g_hat = (
            sum(item.g_hat for values in opponents_by_prompt.values() for item in values)
            / max(1, accounting.accepted)
        )
        return RewardBatch(
            rewards=rewards,
            scores=[min(1.0, max(0.0, self.config.tau * reward)) for reward in rewards],
            opponents=all_opponents,
            accounting=accounting,
            diagnostics={"mean_accepted_g_hat": mean_g_hat},
        )
