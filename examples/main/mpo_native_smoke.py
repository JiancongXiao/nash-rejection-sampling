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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    args = parser.parse_args()
    args.steps = validate_optimizer_steps(args.steps)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint_root = args.output / "trainer_checkpoint"
    policy_source = checkpoint_root / "policy" if args.resume else args.model
    magnet_source = checkpoint_root / "magnet" if args.resume else args.model
    state_path = checkpoint_root / "state.pt"
    if args.resume:
        model_marker = "adapter_config.json" if args.lora_rank > 0 else "config.json"
        for required in (policy_source / model_marker, magnet_source / model_marker, state_path):
            if not required.exists():
                raise FileNotFoundError(f"missing MPO resume checkpoint: {required}")

    lora_config = None
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    def load_model(adapter_source=None, *, trainable=False):
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        if lora_config is None:
            if adapter_source is not None:
                base = AutoModelForCausalLM.from_pretrained(
                    adapter_source,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
            return base.to("cuda")
        if adapter_source is not None:
            return PeftModel.from_pretrained(
                base, adapter_source, is_trainable=trainable
            ).to("cuda")
        return get_peft_model(base, lora_config).to("cuda")

    policy = load_model(policy_source if args.resume else None, trainable=True)
    magnet = load_model(magnet_source if args.resume else None)
    old_policy = load_model()
    magnet.eval()
    old_policy.eval()
    for reference in (magnet, old_policy):
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    name, tensor = first_trainable_tensor(policy)
    before = tensor.detach().float().cpu().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    completed_steps = 0
    completed_outer_rounds = 0
    resume_state = None
    if args.resume:
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer"])
        completed_steps = int(resume_state["completed_steps"])
        completed_outer_rounds = int(resume_state["completed_outer_rounds"])
        if completed_steps % args.inner_steps:
            raise ValueError("MPO resume checkpoint is not on an outer-round boundary")
        if completed_steps >= args.steps:
            raise ValueError(
                f"resume checkpoint already has {completed_steps} steps, target is {args.steps}"
            )
    components = json.loads(args.preference_config.read_text())["components"]
    judge = CountingJudgeState(components, tokenizer)
    outer_rounds = (args.steps + args.inner_steps - 1) // args.inner_steps
    prompts = load_prompts(args.prompts, outer_rounds)
    metrics_path = args.output / "step_metrics.jsonl"
    records = []
    if args.resume and metrics_path.exists():
        records = [
            json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
        ]
    elif metrics_path.exists():
        metrics_path.unlink()
    written_records = len(records)
    optimizer_step = completed_steps
    if resume_state is not None:
        torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])

    def flush_records() -> None:
        nonlocal written_records
        if written_records == len(records):
            return
        with metrics_path.open("a") as handle:
            for record in records[written_records:]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        written_records = len(records)

    def save_checkpoint(completed_outer_step: int) -> None:
        flush_records()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(checkpoint_root / "policy", safe_serialization=True)
        magnet.save_pretrained(checkpoint_root / "magnet", safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_root / "policy")
        tokenizer.save_pretrained(checkpoint_root / "magnet")
        temporary_state = checkpoint_root / "state.pt.tmp"
        torch.save(
            {
                "completed_steps": optimizer_step,
                "completed_outer_rounds": completed_outer_step,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            },
            temporary_state,
        )
        temporary_state.replace(state_path)

    for outer_step, prompt in enumerate(prompts, start=1):
        if outer_step <= completed_outer_rounds:
            continue
        outer_started = time.perf_counter()
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
            started = outer_started if inner_step == 1 else time.perf_counter()
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
        flush_records()
        if (
            args.checkpoint_steps > 0
            and optimizer_step % args.checkpoint_steps == 0
        ):
            save_checkpoint(outer_step)

    save_checkpoint(outer_rounds)
    update = parameter_update(before, tensor, name=name)
    (args.output / "parameter_update.json").write_text(
        json.dumps(update, indent=2, sort_keys=True) + "\n"
    )
    if args.lora_rank > 0:
        policy.save_pretrained(args.output / "actor_adapter", safe_serialization=True)
        merged = policy.merge_and_unload()
        merged.save_pretrained(args.output / "actor", safe_serialization=True)
    else:
        policy.save_pretrained(args.output / "actor", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "actor")
    print(json.dumps(update, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
