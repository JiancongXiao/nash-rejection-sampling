"""Configurable smoke entry around the authors' official EGPO trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from examples.main.smoke_common import (
    build_counting_judge,
    build_metrics_callback,
    load_prompts,
    snapshot_trainable_parameters,
    trainable_parameter_update,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
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
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model

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
    if args.lora_rank > 0:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    before = snapshot_trainable_parameters(model)
    preference = json.loads(args.preference_config.read_text())
    judge = build_counting_judge(preference["components"], tokenizer)
    metrics_path = args.output / "step_metrics.jsonl"
    callback = build_metrics_callback(metrics_path, judge, append_existing=args.resume)
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
        save_strategy="steps" if args.checkpoint_steps > 0 else "no",
        save_steps=max(args.checkpoint_steps, 1),
        save_total_limit=1,
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
    checkpoint = get_last_checkpoint(str(args.output / "trainer")) if args.resume else None
    if args.resume and checkpoint is None:
        raise FileNotFoundError(
            f"no Trainer checkpoint found under {args.output / 'trainer'}"
        )
    trainer.train(resume_from_checkpoint=checkpoint)
    update = trainable_parameter_update(before, model)
    (args.output / "parameter_update.json").write_text(
        json.dumps(update, indent=2, sort_keys=True) + "\n"
    )
    if args.lora_rank > 0:
        trainer.save_model(str(args.output / "actor_adapter"))
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        merged = unwrapped.merge_and_unload()
        merged.save_pretrained(args.output / "actor", safe_serialization=True)
        tokenizer.save_pretrained(args.output / "actor")
    else:
        trainer.save_model(str(args.output / "actor"))
    print(json.dumps(update, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
