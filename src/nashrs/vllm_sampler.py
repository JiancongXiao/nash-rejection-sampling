"""Optional vLLM implementation of the shared TextSampler protocol."""

from __future__ import annotations

from typing import Any


class VLLMTextSampler:
    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.calls = 0

    def sample(self, prompt: str, n: int) -> list[str]:
        from vllm import SamplingParams

        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = SamplingParams(
            n=n,
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.max_new_tokens,
            seed=self.seed + self.calls,
        )
        self.calls += 1
        result = self.llm.generate([formatted], params, use_tqdm=False)[0]
        return [candidate.text.strip() for candidate in result.outputs]
