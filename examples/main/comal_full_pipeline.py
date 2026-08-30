"""Stages for the full COMAL Main pipeline on the shared prompt/oracle setup.

The orchestration remains COMAL's published sampling -> pairwise scoring ->
log-probability -> INPO loop.  This compatibility layer only replaces the
paper's OffsetBias judge with the experiment's frozen shared preference oracle
and makes generation length/model paths configurable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time

from examples.main.smoke_common import CountingJudgeState, load_prompts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_splits(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    width = math.ceil(len(prompts) / args.splits)
    for index in range(args.splits):
        chunk = prompts[index * width : min((index + 1) * width, len(prompts))]
        if not chunk:
            raise ValueError(f"empty COMAL prompt split {index}")
        write_jsonl(
            args.output / f"train_{index}.jsonl",
            [{"prompt": prompt} for prompt in chunk],
        )
    manifest = {
        "prompts": len(prompts),
        "splits": args.splits,
        "split_sizes": [
            len(read_jsonl(args.output / f"train_{index}.jsonl"))
            for index in range(args.splits)
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_jsonl(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    rendered = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )[: args.prompt_max_length]
        rendered.append(tokenizer.decode(prompt_ids, skip_special_tokens=False))
    model = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.prompt_max_length + args.max_new_tokens,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    outputs = model.generate(
        rendered,
        SamplingParams(
            n=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        ),
        use_tqdm=True,
    )
    generated = []
    total_tokens = 0
    for source, request in zip(rows, outputs, strict=True):
        candidates = []
        for output in request.outputs:
            token_count = len(output.token_ids)
            total_tokens += token_count
            candidates.append({"text": output.text, "tokens": token_count})
        if len(candidates) != args.num_samples:
            raise ValueError(
                f"expected {args.num_samples} COMAL candidates, found {len(candidates)}"
            )
        generated.append({"prompt": source["prompt"], "candidates": candidates})
    write_jsonl(args.output, generated)
    summary = {
        "prompts": len(generated),
        "num_samples": args.num_samples,
        "generated_tokens": total_tokens,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def rank_candidates(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    config = json.loads(args.preference_config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    judge = CountingJudgeState(config["components"], tokenizer)
    ranked = []
    for row in read_jsonl(args.candidates):
        candidates = [item["text"] for item in row["candidates"]]
        if len(candidates) < 2:
            raise ValueError("COMAL needs at least two candidates per prompt")
        pairs = [(left, right) for left in range(len(candidates)) for right in range(left + 1, len(candidates))]
        probabilities = judge.compare(
            [row["prompt"]] * len(pairs),
            [(candidates[left], candidates[right]) for left, right in pairs],
        )
        scores = [0.0] * len(candidates)
        for (left, right), probability in zip(pairs, probabilities, strict=True):
            scores[left] += probability
            scores[right] += 1.0 - probability
        divisor = len(candidates) - 1
        scores = [score / divisor for score in scores]
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        chosen, rejected = order[:2]
        ranked.append(
            {
                "prompt": row["prompt"],
                "chosen": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": candidates[chosen]},
                ],
                "rejected": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": candidates[rejected]},
                ],
                "score_chosen": scores[chosen],
                "score_rejected": scores[rejected],
            }
        )
    write_jsonl(args.output, ranked)
    summary = {
        "prompts": len(ranked),
        "preference_model_calls": judge.preference_model_calls,
        "preference_components": judge.component_count,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    oracle = getattr(judge, "oracle", None)
    if hasattr(oracle, "close"):
        oracle.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


def split_pairs(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    validation_size = min(args.validation_size, max(1, len(rows) // 10))
    if len(rows) <= validation_size:
        raise ValueError("COMAL pair split leaves no training rows")
    write_jsonl(args.output / "train.jsonl", rows[:-validation_size])
    write_jsonl(args.output / "test.jsonl", rows[-validation_size:])
    print(
        json.dumps(
            {"train": len(rows) - validation_size, "validation": validation_size},
            indent=2,
            sort_keys=True,
        )
    )


def record_iteration(args: argparse.Namespace) -> None:
    log_text = args.training_log.read_text(errors="replace")
    losses = [float(value) for value in re.findall(r"loss:\s+([-+0-9.eE]+)", log_text)]
    if not losses:
        raise ValueError(f"no COMAL loss found in {args.training_log}")
    learning_rates = [
        float(value)
        for value in re.findall(r"learning rate:\s+([-+0-9.eE]+)", log_text)
    ]
    generation = json.loads(args.generation_summary.read_text())
    preference = json.loads(args.preference_summary.read_text())
    record = {
        "step": args.iteration + 1,
        "outer_iteration": args.iteration,
        "loss": losses[-1],
        "learning_rate": learning_rates[-1] if learning_rates else args.learning_rate,
        "preference_model_calls": preference["preference_model_calls"],
        "preference_components": preference["preference_components"],
        "generated_tokens": generation["generated_tokens"],
        "gpu_hours": (time.time() - args.started_at) * args.num_gpus / 3600.0,
        "num_candidates": generation["num_samples"],
        "reference_updated_after_iteration": (args.iteration + 1) % 12 == 0,
    }
    existing = read_jsonl(args.metrics) if args.metrics.exists() else []
    if len(existing) != args.iteration:
        raise ValueError(
            f"expected {args.iteration} prior COMAL records, found {len(existing)}"
        )
    write_jsonl(args.metrics, [*existing, record])
    print(json.dumps(record, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-splits")
    prepare.add_argument("--prompts", required=True, type=Path)
    prepare.add_argument("--count", type=int, default=2048)
    prepare.add_argument("--splits", type=int, default=6)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.set_defaults(function=prepare_splits)

    generation = commands.add_parser("generate")
    generation.add_argument("--model", required=True)
    generation.add_argument("--tokenizer", required=True)
    generation.add_argument("--prompts", required=True, type=Path)
    generation.add_argument("--output", required=True, type=Path)
    generation.add_argument("--summary", required=True, type=Path)
    generation.add_argument("--num-gpus", type=int, default=8)
    generation.add_argument("--num-samples", type=int, default=5)
    generation.add_argument("--prompt-max-length", type=int, default=128)
    generation.add_argument("--max-new-tokens", type=int, default=512)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-p", type=float, default=0.95)
    generation.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    generation.add_argument("--seed", type=int, required=True)
    generation.set_defaults(function=generate)

    ranking = commands.add_parser("rank")
    ranking.add_argument("--candidates", required=True, type=Path)
    ranking.add_argument("--preference-config", required=True, type=Path)
    ranking.add_argument("--tokenizer", required=True)
    ranking.add_argument("--output", required=True, type=Path)
    ranking.add_argument("--summary", required=True, type=Path)
    ranking.set_defaults(function=rank_candidates)

    split = commands.add_parser("split-pairs")
    split.add_argument("--input", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--validation-size", type=int, default=32)
    split.set_defaults(function=split_pairs)

    record = commands.add_parser("record")
    record.add_argument("--iteration", type=int, required=True)
    record.add_argument("--metrics", required=True, type=Path)
    record.add_argument("--training-log", required=True, type=Path)
    record.add_argument("--generation-summary", required=True, type=Path)
    record.add_argument("--preference-summary", required=True, type=Path)
    record.add_argument("--started-at", type=float, required=True)
    record.add_argument("--num-gpus", type=int, default=8)
    record.add_argument("--learning-rate", type=float, default=5e-7)
    record.set_defaults(function=record_iteration)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
