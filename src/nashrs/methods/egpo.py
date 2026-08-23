"""Predictive-update EGPO adapted to two shared PPO phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..interfaces import PreferenceOracle, TextSampler
from ..types import RewardBatch
from .base import PhaseSpec, SnapshotFn, preference_reward_batch


@dataclass(frozen=True)
class EGPPOConfig:
    opponents_per_prompt: int = 1
    reward_scale: float = 1.0


class EGPPOController:
    """Prediction uses pi_t; correction uses pi_{t+1/2} from the same base."""

    name = "egpo_ppo"

    def __init__(self, config, oracle, reference_policy, snapshot) -> None:
        self.config: EGPPOConfig = config
        self.oracle: PreferenceOracle = oracle
        self.reference_policy: TextSampler = reference_policy
        self.snapshot: SnapshotFn = snapshot
        self.predicted_policy: Optional[TextSampler] = None

    def phase_specs(self) -> Sequence[PhaseSpec]:
        return [
            PhaseSpec("prediction", start_from="current", kl_reference=self.reference_policy),
            PhaseSpec(
                "correction",
                start_from="iteration_start",
                kl_reference=self.reference_policy,
            ),
        ]

    def construct(self, phase, prompts, responses, current_policy) -> RewardBatch:
        if phase == "prediction":
            opponent = current_policy
        elif phase == "correction":
            if self.predicted_policy is None:
                raise RuntimeError("correction requires a completed prediction phase")
            opponent = self.predicted_policy
        else:
            raise ValueError(f"unknown EGPO phase {phase}")
        return preference_reward_batch(
            prompts,
            responses,
            opponent,
            self.oracle,
            self.config.opponents_per_prompt,
            self.config.reward_scale,
            method_name=f"{self.name}/{phase}",
        )

    def after_phase(self, phase: str, iteration: int, updated_policy: TextSampler) -> None:
        if phase == "prediction":
            self.predicted_policy = self.snapshot(
                updated_policy, f"egpo-prediction-{iteration}"
            )
        elif phase != "correction":
            raise ValueError(f"unknown EGPO phase {phase}")

    def after_iteration(self, iteration: int, updated_policy: TextSampler) -> None:
        del iteration, updated_policy
        self.predicted_policy = None

