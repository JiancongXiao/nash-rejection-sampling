"""Evaluate PPO checkpoints on one fixed prompt set with one fixed oracle."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import random
import time

from nashrs import (
    BTLComponent,
    MixtureBTLPreferenceOracle,
    TransformersScalarRewardOracle,
    evaluate_pairwise,
)


STEP_PATTERN = re.compile(r"global_step(\d+)_hf$")


def discover_models(
    base_model: Path,
    checkpoint_root: Path,
    final_model: Path | None,
    steps: set[int] | None = None,
) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = [("base", base_model)]
    checkpoints = []
    if checkpoint_root.exists():
        for path in checkpoint_root.iterdir():
            match = STEP_PATTERN.fullmatch(path.name)
            if match and (path / "config.json").is_file():
                step = int(match.group(1))
                if steps is None or step in steps:
                    checkpoints.append((step, path))
    for step, path in sorted(checkpoints):
        models.append((f"step_{step}", path))
    if not checkpoints and final_model and (final_model / "config.json").is_file():
        models.append(("final", final_model))
    return models


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"invalid prompt at {path}:{line_number}")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("evaluation prompt set is empty")
    return prompts


def generate_responses(
    model_path: Path,
    tokenizer_path: Path,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[int], list[bool]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Every checkpoint comes from the same base model. Reusing the pinned base
    # tokenizer prevents checkpoint-local tokenizer serialization differences
    # from confounding the comparison.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    responses: list[str] = []
    lengths: list[int] = []
    truncated: list[bool] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]
            encoded = tokenizer(
                rendered,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            continuations = generated[:, prompt_width:]
            for tokens in continuations:
                token_ids = tokens.tolist()
                if tokenizer.pad_token_id in token_ids:
                    token_ids = token_ids[: token_ids.index(tokenizer.pad_token_id)]
                responses.append(tokenizer.decode(token_ids, skip_special_tokens=True))
                lengths.append(len(token_ids))
                truncated.append(len(token_ids) >= max_new_tokens)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return responses, lengths, truncated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--final-model", type=Path)
    parser.add_argument("--preference-model", required=True)
    parser.add_argument("--preference-revision")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_intervals(
    methods: list[str],
    scalar_scores: dict[str, list[float]],
    per_prompt: list[list[list[float]]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    prompt_count = len(next(iter(scalar_scores.values())))
    rng = random.Random(seed)
    draws = {
        metric: {method: [] for method in methods}
        for metric in (
            "mean_scalar_reward",
            "average_win_rate",
            "worst_case_win_rate",
            "empirical_exploitability",
        )
    }
    for _ in range(samples):
        indices = [rng.randrange(prompt_count) for _ in range(prompt_count)]
        matrix = [[0.5 for _ in methods] for _ in methods]
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i != j:
                    matrix[i][j] = sum(per_prompt[i][j][k] for k in indices) / prompt_count
        for i, method in enumerate(methods):
            opponents = [matrix[i][j] for j in range(len(methods)) if j != i]
            opponent_wins = [matrix[j][i] for j in range(len(methods)) if j != i]
            draws["mean_scalar_reward"][method].append(
                sum(scalar_scores[method][k] for k in indices) / prompt_count
            )
            draws["average_win_rate"][method].append(sum(opponents) / len(opponents))
            draws["worst_case_win_rate"][method].append(min(opponents))
            draws["empirical_exploitability"][method].append(
                max(0.0, max(opponent_wins) - 0.5)
            )
    return {
        metric: {
            method: [percentile(values, 0.025), percentile(values, 0.975)]
            for method, values in method_values.items()
        }
        for metric, method_values in draws.items()
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.batch_size <= 0:
        raise ValueError("generation lengths and batch sizes must be positive")
    started = time.perf_counter()
    prompts = load_prompts(args.prompts)
    requested_steps = set(args.steps) if args.steps else None
    models = discover_models(
        args.base_model, args.checkpoint_root, args.final_model, requested_steps
    )
    if requested_steps:
        found_steps = {int(name.split("_", 1)[1]) for name, _ in models if name.startswith("step_")}
        missing = requested_steps - found_steps
        if missing:
            raise ValueError(f"missing requested checkpoints: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    responses = {}
    generation_stats = {}
    response_path = args.output_dir / "responses.jsonl"
    with response_path.open("w") as handle:
        for name, path in models:
            values, lengths, truncated = generate_responses(
                path,
                args.base_model,
                prompts,
                args.max_new_tokens,
                args.batch_size,
            )
            responses[name] = values
            generation_stats[name] = {
                "mean_response_tokens": sum(lengths) / len(lengths),
                "truncation_rate": sum(truncated) / len(truncated),
            }
            for prompt, response, length, hit_limit in zip(
                prompts, values, lengths, truncated
            ):
                handle.write(
                    json.dumps(
                        {
                            "method": name,
                            "prompt": prompt,
                            "response": response,
                            "response_tokens": length,
                            "truncated": hit_limit,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    scalar_reward = TransformersScalarRewardOracle(
        args.preference_model,
        revision=args.preference_revision,
        device="cuda",
        batch_size=8,
        max_length=512,
    )
    scalar_scores = {
        name: list(scalar_reward.score(prompts, values))
        for name, values in responses.items()
    }
    preference = MixtureBTLPreferenceOracle([BTLComponent(scalar_reward)])
    pairwise = evaluate_pairwise(prompts, responses, preference)
    methods = list(responses)
    intervals = bootstrap_intervals(
        methods,
        scalar_scores,
        pairwise.per_prompt,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    detail_path = args.output_dir / "per_prompt_metrics.jsonl"
    with detail_path.open("w") as handle:
        for index, prompt in enumerate(prompts):
            handle.write(
                json.dumps(
                    {
                        "prompt_index": index,
                        "prompt": prompt,
                        "scalar_reward": {
                            method: scalar_scores[method][index] for method in methods
                        },
                        "pairwise_probability": {
                            left: {
                                right: pairwise.per_prompt[i][j][index]
                                for j, right in enumerate(methods)
                            }
                            for i, left in enumerate(methods)
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "methods": methods,
        "generation": generation_stats,
        "mean_scalar_reward": {
            name: sum(values) / len(values) for name, values in scalar_scores.items()
        },
        "pairwise_matrix": pairwise.matrix,
        "average_win_rate": {
            name: pairwise.average_win_rate(name) for name in methods
        },
        "worst_case_win_rate": {
            name: pairwise.worst_case_win_rate(name) for name in methods
        },
        "empirical_exploitability": {
            name: pairwise.empirical_exploitability(name) for name in methods
        },
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_intervals_95": intervals,
        "preference_model_calls": pairwise.preference_model_calls,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
