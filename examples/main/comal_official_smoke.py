"""Exercise COMAL's published INPO loss for 2--4 real 0.5B updates.

The full Main entrypoint uses COMAL's sampling/scoring/logprob pipeline.  This
small job deliberately restricts the outer loop and validates the official
INPO update, checkpoint mutation, and the common metrics contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from examples.main.smoke_common import (
    CountingJudgeState,
    first_trainable_tensor,
    generate_completion,
    load_prompts,
    parameter_update,
    sequence_logprob,
    write_records,
)
from nashrs.main_suite import COMAL_SOURCE, validate_smoke_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--preference-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    args.steps = validate_smoke_steps(args.steps)
    sys.path.insert(0, str(args.source_root / COMAL_SOURCE.subdirectory))
    from losses import nash_loss

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.train()
    name, tensor = first_trainable_tensor(model)
    before = tensor.detach().float().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    components = json.loads(args.preference_config.read_text())["components"]
    judge = CountingJudgeState(components, tokenizer)
    prompts = load_prompts(args.prompts, args.steps)
    records = []

    for step, prompt in enumerate(prompts, start=1):
        started = time.perf_counter()
        model.eval()
        first = generate_completion(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, do_sample=True
        )
        second = generate_completion(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, do_sample=True
        )
        probability = judge.compare([prompt], [(first["text"], second["text"])])[0]
        chosen, rejected = (first, second) if probability >= 0.5 else (second, first)
        with torch.no_grad():
            old_chosen = sequence_logprob(
                model, chosen["full_ids"], len(chosen["prompt_ids"])
            ).detach()
            old_rejected = sequence_logprob(
                model, rejected["full_ids"], len(rejected["prompt_ids"])
            ).detach()
        model.train()
        current_chosen = sequence_logprob(
            model, chosen["full_ids"], len(chosen["prompt_ids"])
        )
        current_rejected = sequence_logprob(
            model, rejected["full_ids"], len(rejected["prompt_ids"])
        )
        loss = nash_loss(
            current_chosen.unsqueeze(0),
            current_rejected.unsqueeze(0),
            old_chosen.unsqueeze(0),
            old_rejected.unsqueeze(0),
            old_chosen.unsqueeze(0),
            old_rejected.unsqueeze(0),
            0.002,
            1.0 / 3.0,
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        records.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "learning_rate": args.learning_rate,
                "preference_model_calls": judge.component_count,
                "generated_tokens": int(
                    len(first["completion_ids"]) + len(second["completion_ids"])
                ),
                "gpu_hours": (time.perf_counter() - started) / 3600.0,
                "preference_probability": float(probability),
                "eta": 0.002,
                "tau_eta_ratio": 1.0 / 3.0,
            }
        )

    write_records(args.output / "step_metrics.jsonl", records)
    model.save_pretrained(args.output / "actor", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "actor")
    update = parameter_update(before, tensor, name=name)
    (args.output / "parameter_update.json").write_text(
        json.dumps(update, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(update, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
