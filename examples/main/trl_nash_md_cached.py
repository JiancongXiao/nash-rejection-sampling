"""Method-native Nash-MD with a faithful KV-cached geometric sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.main.smoke_common import (
    build_counting_judge,
    build_metrics_callback,
    first_trainable_tensor,
    load_prompts,
    parameter_update,
)
from nashrs.cached_geometric_generation import cached_geometric_generate
from nashrs.main_suite import validate_optimizer_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("nash_md",), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--preference-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    args.steps = validate_optimizer_steps(args.steps)
    if args.lora_rank <= 0:
        raise ValueError("cached shared-base Nash-MD requires LoRA rank > 0")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import NashMDConfig, NashMDTrainer
    from trl.models.utils import unwrap_model_for_generation

    class CachedNashMDTrainer(NashMDTrainer):
        def _generate_completions(self, model, prompts):
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                policy_output = unwrapped.generate(
                    input_ids=prompts["input_ids"],
                    attention_mask=prompts["attention_mask"],
                    generation_config=self.generation_config,
                )
                mixture_output = cached_geometric_generate(
                    unwrapped,
                    prompts["input_ids"],
                    prompts["attention_mask"],
                    generation_config=self.generation_config,
                    mixture_coef=self.mixture_coef,
                )
            return policy_output, mixture_output

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    config_data = json.loads(args.preference_config.read_text())
    judge = build_counting_judge(config_data["components"], tokenizer)
    metrics_path = args.output / "step_metrics.jsonl"
    callback = build_metrics_callback(
        metrics_path, judge, append_existing=args.resume
    )
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": prompt}]
                for prompt in load_prompts(args.prompts, args.steps)
            ]
        }
    )
    training_args = NashMDConfig(
        output_dir=str(args.output / "trainer"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        lr_scheduler_type="constant",
        max_new_tokens=args.max_new_tokens,
        temperature=0.9,
        beta=0.1,
        mixture_coef=0.125,
        logging_steps=1,
        save_strategy="steps" if args.checkpoint_steps > 0 else "no",
        save_steps=max(args.checkpoint_steps, 1),
        save_total_limit=1,
        report_to=[],
        bf16=True,
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = CachedNashMDTrainer(
        model=model,
        ref_model=None,
        judge=judge,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )
    if trainer.ref_model is not None:
        raise RuntimeError("expected the LoRA reference to share the frozen base model")

    name, tensor = first_trainable_tensor(trainer.model)
    before = tensor.detach().float().cpu().clone()
    checkpoint = (
        get_last_checkpoint(str(args.output / "trainer")) if args.resume else None
    )
    if args.resume and checkpoint is None:
        raise FileNotFoundError(
            f"no Trainer checkpoint found under {args.output / 'trainer'}"
        )
    trainer.train(resume_from_checkpoint=checkpoint)

    update = parameter_update(before, tensor, name=name)
    (args.output / "parameter_update.json").write_text(
        json.dumps(update, indent=2, sort_keys=True) + "\n"
    )
    trainer.save_model(str(args.output / "actor_adapter"))
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    merged = unwrapped.merge_and_unload()
    merged.save_pretrained(args.output / "actor", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "actor")
    print(json.dumps(update, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

