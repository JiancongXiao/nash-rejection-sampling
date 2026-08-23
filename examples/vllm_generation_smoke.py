"""Small real-model vLLM smoke test for the Hopper environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import model_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [
        "Explain in two sentences why reproducible experiments matter.",
        "用两句话解释什么是纳什均衡。",
    ]

    info = model_info(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=info.sha)
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    llm = LLM(
        model=args.model,
        revision=info.sha,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=17,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    started = time.perf_counter()
    outputs = llm.generate(formatted, sampling)
    elapsed = time.perf_counter() - started
    generated_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for prompt, output in zip(prompts, outputs):
        records.append(
            {
                "model": args.model,
                "revision": info.sha,
                "prompt": prompt,
                "response": output.outputs[0].text.strip(),
                "generated_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }
        )
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "revision": info.sha,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for record in records:
        print(f"PROMPT: {record['prompt']}")
        print(f"RESPONSE: {record['response']}")


if __name__ == "__main__":
    main()
