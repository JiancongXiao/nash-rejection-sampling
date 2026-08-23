"""Thin conversion layer for OpenRLHF custom reward functions."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .interfaces import TextSampler
from .methods.base import MethodController
from .nash_rs import NashRSRewardConstructor


class OpenRLHFRewardAdapter:
    """Callable matching OpenRLHF's custom reward-function contract.

    OpenRLHF currently passes full queries and prompts. This adapter extracts
    responses conservatively and delegates all method-specific work to the
    shared constructor. A production deployment should pass decoded strings.
    """

    def __init__(self, constructor: NashRSRewardConstructor) -> None:
        self.constructor = constructor

    def __call__(
        self,
        queries: Sequence[str],
        prompts: Sequence[str],
        labels: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del labels, kwargs
        decoded_queries = [str(query) for query in queries]
        decoded_prompts = [str(prompt) for prompt in prompts]
        responses = [
            query[len(prompt) :] if query.startswith(prompt) else query
            for query, prompt in zip(decoded_queries, decoded_prompts)
        ]
        batch = self.constructor.construct(decoded_prompts, responses)
        try:
            import torch

            rewards: Any = torch.tensor(batch.rewards, dtype=torch.float32)
            scores: Any = torch.tensor(batch.scores, dtype=torch.float32)
        except ImportError:
            rewards = list(batch.rewards)
            scores = list(batch.scores)
        logs = batch.accounting.to_logs()
        logs.update({f"nashrs/{key}": value for key, value in batch.diagnostics.items()})
        return {"rewards": rewards, "scores": scores, "extra_logs": logs}


class OpenRLHFMethodRewardAdapter:
    """OpenRLHF adapter for any unified MethodController phase."""

    def __init__(
        self,
        controller: MethodController,
        current_policy: Callable[[], TextSampler],
        phase: str = "main",
    ) -> None:
        self.controller = controller
        self.current_policy = current_policy
        self.phase = phase

    def __call__(
        self,
        queries: Sequence[str],
        prompts: Sequence[str],
        labels: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del labels, kwargs
        decoded_queries = [str(query) for query in queries]
        decoded_prompts = [str(prompt) for prompt in prompts]
        responses = [
            query[len(prompt) :] if query.startswith(prompt) else query
            for query, prompt in zip(decoded_queries, decoded_prompts)
        ]
        batch = self.controller.construct(
            self.phase, decoded_prompts, responses, self.current_policy()
        )
        try:
            import torch

            rewards: Any = torch.tensor(batch.rewards, dtype=torch.float32)
            scores: Any = torch.tensor(batch.scores, dtype=torch.float32)
        except ImportError:
            rewards = list(batch.rewards)
            scores = list(batch.scores)
        logs = batch.accounting.to_logs(prefix=self.controller.name)
        logs.update(
            {f"{self.controller.name}/{key}": value for key, value in batch.diagnostics.items()}
        )
        return {"rewards": rewards, "scores": scores, "extra_logs": logs}
