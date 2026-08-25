"""Generate a response pool and test whether a reward mixture is non-BT."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from statistics import pstdev

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from nashrs import aggregate_diagnostics, build_preference_oracle, diagnose_score_mixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--temperature-calibration",
        choices=("config", "score_std"),
        default="score_std",
    )
    return parser.parse_args()


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                prompts.append(str(json.loads(line)["prompt"]))
            if len(prompts) == limit:
                break
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def generate_pool(args: argparse.Namespace, prompts: list[str]) -> list[list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, local_files_only=True)
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    llm = LLM(
        model=args.policy_model,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        seed=args.seed,
    )
    params = SamplingParams(
        n=args.responses_per_prompt,
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(formatted, params)
    pool = [[candidate.text.strip() for candidate in output.outputs] for output in outputs]
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return pool


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    component_configs = config["components"]
    prompts = load_prompts(args.prompts, args.max_prompts)
    response_pool = generate_pool(args, prompts)
    mixture = build_preference_oracle(component_configs, device="cuda")

    flattened_prompts = [
        prompt
        for prompt, responses in zip(prompts, response_pool)
        for _ in responses
    ]
    flattened_responses = [response for responses in response_pool for response in responses]
    flat_component_scores = [
        list(component.oracle.score(flattened_prompts, flattened_responses))
        for component in mixture.components
    ]
    nested_scores = [
        [
            scores[index * args.responses_per_prompt : (index + 1) * args.responses_per_prompt]
            for index in range(len(prompts))
        ]
        for scores in flat_component_scores
    ]
    weights = [component.weight for component in mixture.components]
    configured_temperatures = [component.temperature for component in mixture.components]
    if args.temperature_calibration == "score_std":
        temperatures = [max(pstdev(scores), 1e-6) for scores in flat_component_scores]
    else:
        temperatures = configured_temperatures

    per_prompt = []
    for prompt_index in range(len(prompts)):
        scores = [component[prompt_index] for component in nested_scores]
        per_prompt.append(diagnose_score_mixture(scores, weights, temperatures))
    aggregate = aggregate_diagnostics(per_prompt)

    result = {
        "config_name": config.get("name"),
        "policy_model": args.policy_model,
        "prompts": len(prompts),
        "responses_per_prompt": args.responses_per_prompt,
        "temperature_calibration": args.temperature_calibration,
        "configured_temperatures": configured_temperatures,
        "effective_temperatures": temperatures,
        "component_score_mean": [sum(scores) / len(scores) for scores in flat_component_scores],
        "component_score_std": [pstdev(scores) for scores in flat_component_scores],
        "aggregate": aggregate,
        "interpretation": {
            "has_component_disagreement": aggregate["component_disagreement_rate"] > 0.0,
            "has_observed_cycles": aggregate["cycle_rate"] > 0.0,
            "not_exactly_single_bt": not math.isclose(
                aggregate["bt_logit_rmse"], 0.0, abs_tol=1e-6
            ),
        },
        "components": [
            {
                "name": component.get("name", component["model"]),
                "kind": component["kind"],
                "model": component["model"],
                "revision": component.get("revision"),
                "weight": weights[index],
                "effective_temperature": temperatures[index],
            }
            for index, component in enumerate(component_configs)
        ],
        "per_prompt": [
            {
                "prompt": prompt,
                "responses": responses,
                "component_scores": [component[index] for component in nested_scores],
                "diagnostics": per_prompt[index],
            }
            for index, (prompt, responses) in enumerate(zip(prompts, response_pool))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "config_name",
        "prompts",
        "responses_per_prompt",
        "effective_temperatures",
        "component_score_mean",
        "component_score_std",
        "aggregate",
        "interpretation",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
