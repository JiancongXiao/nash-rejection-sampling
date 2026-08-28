"""Algorithm-1 MPO/ReMax smoke runner on the paper's Safe-RLHF backend contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from examples.main.smoke_common import (
    CountingJudgeState,
    first_trainable_tensor,
    generate_completion,
    load_prompts,
    parameter_update,
    sequence_logprob,
    trajectory_forward_kl,
    write_records,
)
from nashrs.main_suite import validate_optimizer_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--preference-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--magnet-interval", type=int, default=1)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--prox-coefficient", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    args.steps = validate_optimizer_steps(args.steps)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    magnet = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    old_policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    magnet.eval()
    old_policy.eval()
    for reference in (magnet, old_policy):
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    name, tensor = first_trainable_tensor(policy)
    before = tensor.detach().float().cpu().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    components = json.loads(args.preference_config.read_text())["components"]
    judge = CountingJudgeState(components, tokenizer)
    outer_rounds = (args.steps + args.inner_steps - 1) // args.inner_steps
    prompts = load_prompts(args.prompts, outer_rounds)
    records = []
    optimizer_step = 0

    for outer_step, prompt in enumerate(prompts, start=1):
        old_policy.load_state_dict(policy.state_dict())
        policy.eval()
        sampled = generate_completion(
            policy, tokenizer, prompt, max_new_tokens=args.max_new_tokens, do_sample=True
        )
        greedy = generate_completion(
            policy, tokenizer, prompt, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        opponent = generate_completion(
            magnet, tokenizer, prompt, max_new_tokens=args.max_new_tokens, do_sample=True
        )
        preference, baseline = judge.compare(
            [prompt, prompt],
            [
                (sampled["text"], opponent["text"]),
                (greedy["text"], opponent["text"]),
            ],
        )
        advantage = float(preference - baseline)
        completion_tokens = max(int(len(sampled["completion_ids"])), 1)
        for inner_step in range(1, args.inner_steps + 1):
            if optimizer_step >= args.steps:
                break
            started = time.perf_counter()
            policy.train()
            policy_logp = sequence_logprob(
                policy, sampled["full_ids"], len(sampled["prompt_ids"])
            )
            normalized_logp = policy_logp / completion_tokens
            magnet_kl = trajectory_forward_kl(
                policy, magnet, sampled["full_ids"], len(sampled["prompt_ids"])
            )
            trust_region_kl = trajectory_forward_kl(
                policy, old_policy, sampled["full_ids"], len(sampled["prompt_ids"])
            )
            loss = (
                -advantage * normalized_logp
                + args.alpha * magnet_kl
                + args.prox_coefficient * trust_region_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer_step += 1
            records.append(
                {
                    "step": optimizer_step,
                    "outer_step": outer_step,
                    "inner_step": inner_step,
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": args.learning_rate,
                    "preference_model_calls": (
                        2 * judge.component_count if inner_step == 1 else 0
                    ),
                    "generated_tokens": (
                        int(
                            len(sampled["completion_ids"])
                            + len(greedy["completion_ids"])
                            + len(opponent["completion_ids"])
                        )
                        if inner_step == 1
                        else 0
                    ),
                    "gpu_hours": (time.perf_counter() - started) / 3600.0,
                    "preference": float(preference),
                    "remax_baseline": float(baseline),
                    "advantage": advantage,
                    "magnet_kl": float(magnet_kl.detach().cpu()),
                    "trust_region_kl": float(trust_region_kl.detach().cpu()),
                    "alpha": args.alpha,
                    "prox_coefficient": args.prox_coefficient,
                }
            )
        magnet_updated = outer_step % args.magnet_interval == 0
        if magnet_updated:
            magnet.load_state_dict(policy.state_dict())
        for record in records[-min(args.inner_steps, len(records)) :]:
            if record["outer_step"] == outer_step:
                record["magnet_updated_after_outer_step"] = magnet_updated

    write_records(args.output / "step_metrics.jsonl", records)
    policy.save_pretrained(args.output / "actor", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "actor")
    update = parameter_update(before, tensor, name=name)
    (args.output / "parameter_update.json").write_text(
        json.dumps(update, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(update, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
