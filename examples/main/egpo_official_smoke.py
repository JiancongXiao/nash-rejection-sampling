"""Configurable smoke entry around the authors' official EGPO trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from examples.main.smoke_common import (
    build_counting_judge,
    build_metrics_callback,
    first_trainable_tensor,
    load_prompts,
    parameter_update,
)
from nashrs.main_suite import EGPO_SOURCE, validate_optimizer_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--preference-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    args.steps = validate_optimizer_steps(args.steps)

    lms_root = args.source_root / EGPO_SOURCE.subdirectory / "lms"
    sys.path.insert(0, str(lms_root))
    from trainers.extragradient_config import ExtragradientConfig
    from trainers.extragradient_trainer import ExtragradientTrainer

    import torch
    from accelerate.optimizer import AcceleratedOptimizer
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    # The published single-GPU EGPO trainer gathers the backed-up optimizer
    # state into a one-element per-rank list before restoring it.  Current
    # Accelerate expects the local state dict directly.  Accept that official
    # representation at the wrapper boundary without changing EGPO's
    # prediction/correction update.
    original_load_state_dict = AcceleratedOptimizer.load_state_dict

    def load_egpo_state_dict(optimizer, state_dict):
        if isinstance(state_dict, list):
            local_states = [state for state in state_dict if state is not None]
            if len(local_states) != 1:
                raise ValueError(
                    "single-GPU EGPO expected exactly one optimizer state dict"
                )
            state_dict = local_states[0]
        return original_load_state_dict(optimizer, state_dict)

    AcceleratedOptimizer.load_state_dict = load_egpo_state_dict

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    name, tensor = first_trainable_tensor(model)
    before = tensor.detach().float().cpu().clone()
    preference = json.loads(args.preference_config.read_text())
    judge = build_counting_judge(preference["components"], tokenizer)
    metrics_path = args.output / "step_metrics.jsonl"
    callback = build_metrics_callback(metrics_path, judge)
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": prompt}]
                for prompt in load_prompts(args.prompts, args.steps)
            ]
        }
    )
    training_args = ExtragradientConfig(
        output_dir=str(args.output / "trainer"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_generate_batch_size=1,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        lr_scheduler_type="constant",
        max_new_tokens=args.max_new_tokens,
        beta=0.1,
        estimate_extra_grad=True,
        y_yp_mixture_coef=0.0,
        samples_per_prompt=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = ExtragradientTrainer(
        model=model,
        judge=judge,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.train()
    trainer.save_model(str(args.output / "actor"))
    update = parameter_update(before, tensor, name=name)
    (args.output / "parameter_update.json").write_text(
        json.dumps(update, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(update, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
