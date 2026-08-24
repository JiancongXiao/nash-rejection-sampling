"""Nash-RS real-generation smoke test with a real preference model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info
from transformers import AutoTokenizer
from vllm import LLM

from nashrs import (
    BTLComponent,
    MixtureBTLPreferenceOracle,
    NashRSConfig,
    NashRSRewardConstructor,
    TransformersScalarRewardOracle,
)
from nashrs.vllm_sampler import VLLMTextSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--preference-model",
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
    )
    parser.add_argument("--preference-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--b1", type=int, default=2)
    parser.add_argument("--b2", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = "Explain briefly why reproducible machine-learning experiments matter."
    policy_info = model_info(args.model)
    preference_revision = args.preference_revision or model_info(
        args.preference_model
    ).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=policy_info.sha)
    llm = LLM(
        model=args.model,
        revision=policy_info.sha,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=37,
    )
    scalar_reward = TransformersScalarRewardOracle(
        args.preference_model,
        revision=preference_revision,
        device="cuda",
        batch_size=8,
        max_length=512,
    )
    preference = MixtureBTLPreferenceOracle([BTLComponent(scalar_reward)])
    policy = VLLMTextSampler(llm, tokenizer, args.max_new_tokens, seed=109)
    reference = VLLMTextSampler(llm, tokenizer, args.max_new_tokens, seed=709)
    rollout_response = policy.sample(prompt, 1)[0]
    constructor = NashRSRewardConstructor(
        NashRSConfig(
            tau=args.tau,
            b1=args.b1,
            b2=args.b2,
            seed=19,
            proposal_batch_size=4,
            max_proposals_per_prompt=64,
        ),
        policy_sampler=policy,
        reference_sampler=reference,
        preference_oracle=preference,
    )
    batch = constructor.construct([prompt], [rollout_response])

    record = {
        "policy_model": args.model,
        "policy_revision": policy_info.sha,
        "preference_model": args.preference_model,
        "preference_revision": preference_revision,
        "preference_type": "single-component BTL smoke",
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
