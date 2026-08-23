"""Backend-agnostic execution of stateful PPO method controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .interfaces import TextSampler
from .methods.base import MethodController, PhaseSpec
from .types import Accounting, RewardBatch


class PPOBackend(Protocol):
    """Minimum operations required from an OpenRLHF integration."""

    def snapshot_current(self, label: str) -> TextSampler: ...

    def restore(self, policy: TextSampler) -> None: ...

    def current_policy(self) -> TextSampler: ...

    def rollout(self, prompts: Sequence[str]) -> Sequence[str]: ...

    def ppo_update(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        rewards: RewardBatch,
        phase: PhaseSpec,
    ) -> None: ...


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    rewards: RewardBatch


@dataclass(frozen=True)
class IterationResult:
    method: str
    iteration: int
    phases: Sequence[PhaseResult]
    accounting: Accounting


def _merge_accounting(target: Accounting, source: Accounting) -> None:
    for field in (
        "preference_model_calls",
        "policy_generations",
        "reference_generations",
        "opponent_generations",
        "generated_tokens",
        "proposals",
        "accepted",
        "wall_time_seconds",
        "gpu_hours",
    ):
        setattr(target, field, getattr(target, field) + getattr(source, field))


def run_iteration(
    controller: MethodController,
    backend: PPOBackend,
    prompts: Sequence[str],
    iteration: int,
) -> IterationResult:
    """Execute all phases while enforcing restore/snapshot semantics."""
    iteration_start = backend.snapshot_current(
        f"{controller.name}-iteration-{iteration}-start"
    )
    results: list[PhaseResult] = []
    accounting = Accounting()

    for phase in controller.phase_specs():
        if phase.start_from == "iteration_start":
            backend.restore(iteration_start)
        current = backend.current_policy()
        responses = list(backend.rollout(prompts))
        if len(responses) != len(prompts):
            raise ValueError("PPO backend returned the wrong rollout batch size")
        reward_batch = controller.construct(phase.name, prompts, responses, current)
        backend.ppo_update(prompts, responses, reward_batch, phase)
        updated = backend.current_policy()
        controller.after_phase(phase.name, iteration, updated)
        _merge_accounting(accounting, reward_batch.accounting)
        results.append(PhaseResult(phase.name, reward_batch))

    controller.after_iteration(iteration, backend.current_policy())
    return IterationResult(controller.name, iteration, results, accounting)

