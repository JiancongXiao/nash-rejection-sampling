"""Adapter for the pair classifier released with EGPO.

The checkpoint predicts which of two responses wins directly.  It is not a
scalar reward model and must therefore not be wrapped in a Bradley--Terry
conversion.  Response order is randomized exactly as in the author code.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence, Any


PAIR_TEMPLATE = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

'''


def oriented_probability(probabilities: Sequence[float], flipped: bool) -> float:
    """Return P(original-left wins) after an optional presentation swap."""

    if len(probabilities) != 2:
        raise ValueError("the EGPO preference checkpoint must return two classes")
    return float(probabilities[1 if flipped else 0])


class EGPODirectPreferenceOracle:
    """Batched direct pairwise oracle using the published EGPO prompt format."""

    def __init__(self, config: Mapping[str, Any], device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.components = (self,)
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 1024))
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.rng = random.Random(int(config.get("order_seed", 47)))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config["model"]),
            revision=config.get("revision"),
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(config["model"]),
            revision=config.get("revision"),
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(self.device).eval()
        if self.model.config.num_labels != 2:
            raise ValueError("EGPO preference checkpoint must have two logits")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def compare(
        self,
        prompts: Sequence[str],
        left: Sequence[str],
        right: Sequence[str],
    ) -> list[float]:
        if not (len(prompts) == len(left) == len(right)):
            raise ValueError("preference input lengths differ")
        flips = [bool(self.rng.getrandbits(1)) for _ in prompts]
        rendered = [
            PAIR_TEMPLATE.format(
                prompt=prompt,
                response0=right_value if flip else left_value,
                response1=left_value if flip else right_value,
            )
            for prompt, left_value, right_value, flip in zip(
                prompts, left, right, flips
            )
        ]
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(rendered), self.batch_size):
                end = start + self.batch_size
                inputs = self.tokenizer(
                    rendered[start:end],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.float()
                if logits.shape[1] != 2 or not self.torch.isfinite(logits).all():
                    raise ValueError("invalid EGPO preference logits")
                probabilities = logits.softmax(-1).cpu().tolist()
                scores.extend(
                    oriented_probability(row, flip)
                    for row, flip in zip(probabilities, flips[start:end])
                )
        return scores
