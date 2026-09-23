"""Eight-GPU evaluation of initial Tulu and four full-data PKU policies."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import time


METHODS = ("initial", "nash_rs", "nash_md", "egpo", "mpo")


def load_prompts(path: Path, count: int) -> list[str]:
    rows = [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line]
    if len(rows) < count:
        raise ValueError(f"requested {count} prompts but {path} has {len(rows)}")
    return rows[:count]


def generate(
    name: str,
    base_model: str,
    adapter: Path | None,
    prompts: list[str],
    output: Path,
    max_new_tokens: int,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 8):
            batch = prompts[start : start + 8]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]
            values = tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")
            generated = model.generate(
                **values,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            width = values["input_ids"].shape[1]
            for offset, tokens in enumerate(generated[:, width:]):
                token_ids = tokens.tolist()
                if tokenizer.pad_token_id in token_ids:
                    token_ids = token_ids[: token_ids.index(tokenizer.pad_token_id)]
                records.append(
                    {
                        "index": start + offset,
                        "prompt": batch[offset],
                        "response": tokenizer.decode(token_ids, skip_special_tokens=True),
                        "response_tokens": len(token_ids),
                        "truncated": len(token_ids) >= max_new_tokens,
                    }
                )
    with output.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"generated {len(records)} responses for {name}", flush=True)


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def score(
    responses: dict[str, list[dict]],
    preference_config: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    from nashrs import build_preference_oracle

    config = json.loads(preference_config.read_text())
    oracle = build_preference_oracle(config["components"], device="cuda")
    count = len(next(iter(responses.values())))
    matrix = [[0.5 for _ in METHODS] for _ in METHODS]
    per_prompt = [[[0.5] * count for _ in METHODS] for _ in METHODS]
    calls = 0
    for left_index, left_name in enumerate(METHODS):
        for right_index in range(left_index + 1, len(METHODS)):
            right_name = METHODS[right_index]
            values = oracle.compare(
                [row["prompt"] for row in responses[left_name]],
                [row["response"] for row in responses[left_name]],
                [row["response"] for row in responses[right_name]],
            )
            calls += count
            per_prompt[left_index][right_index] = values
            per_prompt[right_index][left_index] = [1.0 - value for value in values]
            matrix[left_index][right_index] = sum(values) / count
            matrix[right_index][left_index] = 1.0 - matrix[left_index][right_index]

    def metrics(active_matrix):
        average, worst, exploit = {}, {}, {}
        for index, name in enumerate(METHODS):
            opponents = [active_matrix[index][other] for other in range(len(METHODS)) if other != index]
            opponent_wins = [active_matrix[other][index] for other in range(len(METHODS)) if other != index]
            average[name] = sum(opponents) / len(opponents)
            worst[name] = min(opponents)
            exploit[name] = max(0.0, max(opponent_wins) - 0.5)
        return average, worst, exploit

    average, worst, exploit = metrics(matrix)
    rng = random.Random(bootstrap_seed)
    draws = {
        metric: {name: [] for name in METHODS}
        for metric in ("average_win_rate", "worst_case_win_rate", "empirical_exploitability")
    }
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(count) for _ in range(count)]
        boot = [[0.5 for _ in METHODS] for _ in METHODS]
        for left in range(len(METHODS)):
            for right in range(len(METHODS)):
                if left != right:
                    boot[left][right] = sum(per_prompt[left][right][i] for i in indices) / count
        values = metrics(boot)
        for metric, by_method in zip(draws, values):
            for name, value in by_method.items():
                draws[metric][name].append(value)
    confidence = {
        metric: {
            name: [percentile(values, 0.025), percentile(values, 0.975)]
            for name, values in by_method.items()
        }
        for metric, by_method in draws.items()
    }
    generation = {
        name: {
            "mean_response_tokens": sum(row["response_tokens"] for row in rows) / count,
            "truncation_rate": sum(row["truncated"] for row in rows) / count,
        }
        for name, rows in responses.items()
    }
    return {
        "methods": list(METHODS),
        "prompts": count,
        "generation": generation,
        "pairwise_matrix": matrix,
        "average_win_rate": average,
        "worst_case_win_rate": worst,
        "empirical_exploitability": exploit,
        "confidence_intervals_95": confidence,
        "preference_model_calls": calls,
        "preference_config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--preference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    started = time.perf_counter()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.get_world_size() != 8:
        raise RuntimeError("evaluation requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    response_root = args.output / "responses"
    response_root.mkdir(exist_ok=True)
    prompts = load_prompts(args.prompts, args.max_prompts)

    if rank < len(METHODS):
        name = METHODS[rank]
        adapter = None if name == "initial" else args.result_root / name / "actor_adapter"
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing completed adapter for {name}: {adapter}")
        destination = response_root / f"{name}.jsonl"
        if not destination.is_file():
            generate(
                name,
                args.base_model,
                adapter,
                prompts,
                destination,
                args.max_new_tokens,
            )
    dist.barrier()

    if rank == 0:
        responses = {
            name: [json.loads(line) for line in (response_root / f"{name}.jsonl").read_text().splitlines()]
            for name in METHODS
        }
        for name, rows in responses.items():
            if len(rows) != args.max_prompts:
                raise RuntimeError(f"incomplete response file for {name}: {len(rows)}")
        summary = score(
            responses,
            args.preference_config,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        summary["base_model"] = args.base_model
        summary["training_result_root"] = str(args.result_root)
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["evaluation_gpu_hours_upper_bound"] = summary["elapsed_seconds"] * 8 / 3600
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
