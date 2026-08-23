"""Exercise Nash-RS rejection sampling with real vLLM generations on one GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from huggingface_hub import model_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from nashrs import NashRSConfig, NashRSRewardConstructor


class VLLMTextSampler:
    def __init__(
        self,
        llm: LLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.calls = 0

    def sample(self, prompt: str, n: int) -> list[str]:
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


class LengthPreferenceOracle:
    """Bounded antisymmetric smoke oracle; replace with the real PM next."""

    def __init__(self, scale: float = 12.0) -> None:
        self.scale = scale

    def compare(
        self,
        prompts: Sequence[str],
        left: Sequence[str],
        right: Sequence[str],
    ) -> list[float]:
        del prompts
        return [
            1.0 / (1.0 + math.exp(-(len(a.split()) - len(b.split())) / self.scale))
            for a, b in zip(left, right)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--b1", type=int, default=2)
    parser.add_argument("--b2", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = "Explain briefly why a Nash equilibrium is useful."
    info = model_info(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=info.sha)
    llm = LLM(
        model=args.model,
        revision=info.sha,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=31,
    )

    policy = VLLMTextSampler(llm, tokenizer, args.max_new_tokens, seed=101)
    reference = VLLMTextSampler(llm, tokenizer, args.max_new_tokens, seed=701)
    rollout_response = policy.sample(prompt, 1)[0]
    constructor = NashRSRewardConstructor(
        NashRSConfig(
            tau=args.tau,
            b1=args.b1,
            b2=args.b2,
            seed=17,
            proposal_batch_size=4,
            max_proposals_per_prompt=64,
        ),
        policy_sampler=policy,
        reference_sampler=reference,
        preference_oracle=LengthPreferenceOracle(),
    )
    batch = constructor.construct([prompt], [rollout_response])

    record = {
        "model": args.model,
        "revision": info.sha,
        "prompt": prompt,
        "rollout_response": rollout_response,
        "tau": args.tau,
        "b1": args.b1,
        "b2": args.b2,
        "implicit_reward": batch.rewards[0],
        "preference_score": batch.scores[0],
        "opponents": [
            {
                "response": opponent.response,
                "g_hat": opponent.g_hat,
                "acceptance_probability": opponent.acceptance_probability,
            }
            for opponent in batch.opponents[0]
        ],
        "accounting": batch.accounting.to_logs(),
        "diagnostics": dict(batch.diagnostics),
        "acceptance_rule": "u <= exp(-g_hat/tau)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
