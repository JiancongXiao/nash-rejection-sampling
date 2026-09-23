"""Reusable scalar-reward and pairwise-preference model adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .interfaces import ScalarRewardOracle


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class BTLComponent:
    oracle: ScalarRewardOracle
    weight: float = 1.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("component weight must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("component temperature must be positive")


class MixtureBTLPreferenceOracle:
    """Weighted mixture of BTL probabilities from scalar reward models.

    With one component this is an ordinary BTL oracle. Multiple independently
    trained reward models generally define a preference relation that cannot
    be represented by one scalar reward, which is useful for NLHF experiments.
    """

    def __init__(self, components: Sequence[BTLComponent]) -> None:
        self.components = tuple(components)
        if not self.components:
            raise ValueError("at least one BTL component is required")
        self.total_weight = sum(component.weight for component in self.components)
        if self.total_weight <= 0.0:
            raise ValueError("component weights must have a positive sum")

    def compare(
        self,
        prompts: Sequence[str],
        left: Sequence[str],
        right: Sequence[str],
    ) -> list[float]:
        if not (len(prompts) == len(left) == len(right)):
            raise ValueError("prompts, left, and right must have equal length")
        probabilities = [0.0] * len(prompts)
        for component in self.components:
            left_scores = list(component.oracle.score(prompts, left))
            right_scores = list(component.oracle.score(prompts, right))
            if len(left_scores) != len(prompts) or len(right_scores) != len(prompts):
                raise ValueError("scalar reward oracle returned the wrong batch size")
            for index, (left_score, right_score) in enumerate(
                zip(left_scores, right_scores)
            ):
                probabilities[index] += component.weight * _sigmoid(
                    (float(left_score) - float(right_score)) / component.temperature
                )
        return [probability / self.total_weight for probability in probabilities]


class TransformersScalarRewardOracle:
    """Batched Hugging Face sequence-classification reward model."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
        )
        self.model.to(self.device)
        self.model.eval()

    def score(
        self, prompts: Sequence[str], responses: Sequence[str]
    ) -> list[float]:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                encoded = self.tokenizer(
                    list(prompts[start : start + self.batch_size]),
                    list(responses[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError(
                        "expected a scalar sequence-classification reward model"
                    )
                scores.extend(float(value) for value in logits[:, 0].float().cpu())
        return scores


class TransformersChatRewardOracle:
    """Sequence-classification reward model using its native chat template.

    Models such as Skywork Reward V2 are trained on rendered conversations,
    rather than tokenizer ``text``/``text_pair`` inputs. Keeping this adapter
    separate prevents accidentally evaluating them with the DeBERTa encoding.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 1024,
        system_prompt: str | None = None,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.chat_template is None:
            raise ValueError(f"reward model has no chat template: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
        )
        self.model.to(self.device)
        self.model.eval()

    def _render(self, prompt: str, response: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def score(
        self, prompts: Sequence[str], responses: Sequence[str]
    ) -> list[float]:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have equal length")
        rendered = [
            self._render(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(rendered), self.batch_size):
                encoded = self.tokenizer(
                    rendered[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError(
                        "expected a scalar sequence-classification reward model"
                    )
                scores.extend(float(value) for value in logits[:, 0].float().cpu())
        return scores


def build_preference_oracle(
    component_configs: Sequence[Mapping[str, Any]],
    *,
    device: str = "cuda",
) -> MixtureBTLPreferenceOracle:
    """Build a pinned mixture from JSON-compatible component definitions."""

    if (
        len(component_configs) == 1
        and component_configs[0].get("kind") == "egpo_pair"
    ):
        from .egpo_pair_oracle import EGPODirectPreferenceOracle

        return EGPODirectPreferenceOracle(component_configs[0], device=device)

    components: list[BTLComponent] = []
    for config in component_configs:
        kind = str(config.get("kind", "pair"))
        common = {
            "model_name": str(config["model"]),
            "revision": config.get("revision"),
            "device": device,
            "batch_size": int(config.get("batch_size", 8)),
            "max_length": int(config.get("max_length", 512)),
        }
        if kind == "pair":
            oracle: ScalarRewardOracle = TransformersScalarRewardOracle(**common)
        elif kind == "chat":
            oracle = TransformersChatRewardOracle(
                **common,
                system_prompt=config.get("system_prompt"),
            )
        else:
            raise ValueError(f"unknown preference component kind: {kind}")
        components.append(
            BTLComponent(
                oracle=oracle,
                weight=float(config.get("weight", 1.0)),
                temperature=float(config.get("temperature", 1.0)),
            )
        )
    return MixtureBTLPreferenceOracle(components)
