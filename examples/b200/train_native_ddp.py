"""Eight-GPU method-native Nash-MD, EGPO, and MPO training.

Nash-MD and EGPO retain their authors' Trainer implementations and run under
Accelerate DDP.  MPO retains the published two-inner-step update and uses a
thin synchronous DDP loop.  All three consume the same fixed prompt budget and
global batch and emit the same accounting/provenance files.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

from examples.main.smoke_common import (
    CountingJudgeState,
    build_counting_judge,
    build_metrics_callback,
    first_trainable_tensor,
    load_prompts,
    parameter_update,
    snapshot_trainable_parameters,
    trainable_parameter_update,
)
from nashrs.cached_geometric_generation import cached_geometric_generate
from nashrs.main_suite import EGPO_SOURCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("nash_md", "egpo", "mpo"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--preference-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-budget", type=int, required=True)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--train-micro-batch", type=int, default=1)
    parser.add_argument("--generate-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--checkpoint-steps", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 8:
        parser.error(f"this entry requires torchrun with exactly 8 ranks, got {world}")
    for name in (
        "prompt_budget", "global_batch", "train_micro_batch", "generate_batch",
        "lora_rank", "lora_alpha",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    per_accumulation = args.train_micro_batch * world
    if args.global_batch % per_accumulation:
        parser.error(
            "global batch must be divisible by train_micro_batch * world_size"
        )
    return args


def rank_paths(output: Path) -> tuple[int, int, Path]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, output / f"step_metrics.rank{rank}.jsonl"


def write_manifest(args: argparse.Namespace, *, optimizer_steps: int) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    manifest = {
        "status": "complete",
        "method": args.method,
        "model": args.model,
        "prompts": str(args.prompts),
        "preference_config": str(args.preference_config),
        "prompt_budget": args.prompt_budget,
        "world_size": 8,
        "global_batch": args.global_batch,
        "train_micro_batch_per_rank": args.train_micro_batch,
        "generate_batch_per_rank": args.generate_batch,
        "optimizer_steps": optimizer_steps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_new_tokens": args.max_new_tokens,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def train_trainer_method(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    rank, local_rank, metrics_path = rank_paths(args.output)
    torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    world = int(os.environ["WORLD_SIZE"])
    gradient_accumulation = args.global_batch // (args.train_micro_batch * world)
    optimizer_steps = math.ceil(args.prompt_budget / args.global_batch)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    preference = json.loads(args.preference_config.read_text())
    judge = build_counting_judge(preference["components"], tokenizer)
    callback = build_metrics_callback(
        metrics_path, judge, append_existing=args.resume
    )
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": prompt}]
                for prompt in load_prompts(args.prompts, args.prompt_budget)
            ]
        }
    )

    if args.method == "nash_md":
        from trl import NashMDConfig, NashMDTrainer
        from trl.models.utils import unwrap_model_for_generation

        class CachedNashMDTrainer(NashMDTrainer):
            def _generate_completions(self, active_model, prompts):
                with unwrap_model_for_generation(active_model, self.accelerator) as unwrapped:
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

        training_args = NashMDConfig(
            output_dir=str(args.output / "trainer"),
            max_steps=optimizer_steps,
            per_device_train_batch_size=args.train_micro_batch,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_steps=0,
            lr_scheduler_type="constant",
            max_new_tokens=args.max_new_tokens,
            temperature=0.9,
            beta=0.1,
            mixture_coef=0.125,
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.checkpoint_steps,
            save_total_limit=1,
            report_to=[],
            bf16=True,
            seed=args.seed,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
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
            raise RuntimeError("LoRA Nash-MD must share the frozen base reference")
    else:
        lms_root = args.source_root / EGPO_SOURCE.subdirectory / "lms"
        if not lms_root.is_dir():
            raise FileNotFoundError(f"missing pinned EGPO source: {lms_root}")
        sys.path.insert(0, str(lms_root))
        from accelerate.optimizer import AcceleratedOptimizer
        from accelerate.state import PartialState
        from trainers.extragradient_config import ExtragradientConfig
        from trainers.extragradient_trainer import ExtragradientTrainer

        original_load = AcceleratedOptimizer.load_state_dict

        def load_egpo_state(optimizer, state_dict):
            if isinstance(state_dict, list):
                states = [value for value in state_dict if value is not None]
                process_index = PartialState().process_index
                if len(states) == 1:
                    state_dict = states[0]
                elif len(states) == world:
                    state_dict = states[process_index]
                else:
                    raise ValueError(
                        f"unexpected EGPO optimizer-state gather: {len(states)}"
                    )
            return original_load(optimizer, state_dict)

        AcceleratedOptimizer.load_state_dict = load_egpo_state
        model = get_peft_model(model, peft_config)
        training_args = ExtragradientConfig(
            output_dir=str(args.output / "trainer"),
            max_steps=optimizer_steps,
            per_device_train_batch_size=args.train_micro_batch,
            gradient_accumulation_steps=gradient_accumulation,
            per_device_generate_batch_size=args.generate_batch,
            learning_rate=args.learning_rate,
            warmup_steps=0,
            lr_scheduler_type="constant",
            max_new_tokens=args.max_new_tokens,
            beta=0.1,
            estimate_extra_grad=True,
            y_yp_mixture_coef=0.0,
            samples_per_prompt=1,
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.checkpoint_steps,
            save_total_limit=1,
            report_to=[],
            bf16=True,
            seed=args.seed,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
        )
        trainer = ExtragradientTrainer(
            model=model,
            judge=judge,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback],
        )

    before = snapshot_trainable_parameters(trainer.model)
    checkpoint = get_last_checkpoint(str(args.output / "trainer")) if args.resume else None
    if args.resume and checkpoint is None:
        raise FileNotFoundError("resume requested but no Trainer checkpoint exists")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        update = trainable_parameter_update(before, trainer.model)
        (args.output / "parameter_update.json").write_text(
            json.dumps(update, indent=2, sort_keys=True) + "\n"
        )
    trainer.save_model(str(args.output / "actor_adapter"))
    trainer.accelerator.wait_for_everyone()
    write_manifest(args, optimizer_steps=optimizer_steps)
    if rank == 0:
        print(f"{args.method} completed {optimizer_steps} optimizer steps")


def train_mpo(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    dist.init_process_group("nccl")
    rank, local_rank, metrics_path = rank_paths(args.output)
    world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts, args.prompt_budget)
    outer_rounds = math.ceil(args.prompt_budget / args.global_batch)
    optimizer_steps = 2 * outer_rounds

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    checkpoint_root = args.output / "trainer_checkpoint"
    state_path = checkpoint_root / "state.pt"
    if args.resume and not state_path.is_file():
        raise FileNotFoundError(f"missing MPO resume state: {state_path}")

    def load_model(*, trainable: bool, adapter: Path | None = None):
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        base.config.use_cache = not trainable
        if adapter is None:
            value = get_peft_model(base, lora_config)
        else:
            from peft import PeftModel

            value = PeftModel.from_pretrained(
                base, adapter, is_trainable=trainable
            )
        value = value.to(device)
        if not trainable:
            value.eval()
            for parameter in value.parameters():
                parameter.requires_grad_(False)
        return value

    policy_adapter = checkpoint_root / "policy" if args.resume else None
    magnet_adapter = checkpoint_root / "magnet" if args.resume else None
    policy_model = load_model(trainable=True, adapter=policy_adapter)
    magnet = load_model(trainable=False, adapter=magnet_adapter)
    old_policy = load_model(trainable=False)
    policy = DistributedDataParallel(
        policy_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    name, tensor = first_trainable_tensor(policy_model)
    before = tensor.detach().float().cpu().clone()
    optimizer = torch.optim.AdamW(
        [value for value in policy_model.parameters() if value.requires_grad],
        lr=args.learning_rate,
    )
    completed_outer_rounds = 0
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        completed_outer_rounds = int(state["outer_step"])
        optimizer.load_state_dict(state["optimizer"])
        if completed_outer_rounds >= outer_rounds:
            raise ValueError("MPO checkpoint already reached the requested prompt budget")
    components = json.loads(args.preference_config.read_text())["components"]
    judge = CountingJudgeState(components, tokenizer)

    def generate_batch(model, batch, *, sample):
        rows = []
        model.eval()
        for start in range(0, len(batch), args.generate_batch):
            chunk = batch[start : start + args.generate_batch]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in chunk
            ]
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(device)
            kwargs = {"temperature": 1.0, "top_p": 0.95} if sample else {}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **kwargs,
                )
            width = encoded["input_ids"].shape[1]
            for row_index in range(len(chunk)):
                prompt_ids = encoded["input_ids"][row_index][
                    encoded["attention_mask"][row_index].bool()
                ].detach().cpu()
                completion = output[row_index, width:].detach().cpu()
                if tokenizer.eos_token_id in completion:
                    end = completion.tolist().index(tokenizer.eos_token_id) + 1
                    completion = completion[:end]
                rows.append(
                    {
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion,
                        "full_ids": torch.cat((prompt_ids, completion)),
                        "text": tokenizer.decode(completion, skip_special_tokens=True),
                    }
                )
        return rows

    def objective(rows):
        lengths = [int(row["full_ids"].numel()) for row in rows]
        width = max(lengths)
        values = torch.full(
            (len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        attention = torch.zeros_like(values)
        completion_mask = torch.zeros(
            (len(rows), width - 1), dtype=torch.float32, device=device
        )
        for index, (row, length) in enumerate(zip(rows, lengths)):
            values[index, :length] = row["full_ids"].to(device)
            attention[index, :length] = 1
            start = max(int(row["prompt_ids"].numel()) - 1, 0)
            completion_mask[index, start : length - 1] = 1
        policy_logits = policy(input_ids=values, attention_mask=attention).logits[:, :-1].float()
        labels = values[:, 1:]
        policy_log_probs = policy_logits.log_softmax(-1)
        sequence_logps = (
            policy_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            * completion_mask
        ).sum(-1)
        policy_probs = policy_log_probs.exp()

        def forward_kl(reference):
            with torch.no_grad():
                ref_log_probs = reference(
                    input_ids=values, attention_mask=attention
                ).logits[:, :-1].float().log_softmax(-1)
            token_kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(-1)
            return (token_kl * completion_mask).sum(-1)

        return sequence_logps, forward_kl(magnet), forward_kl(old_policy)

    records = []
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for outer_index in range(completed_outer_rounds, outer_rounds):
        start = outer_index * args.global_batch
        global_prompts = prompts[start : start + args.global_batch]
        # Pad the final global batch so all ranks execute the same number of
        # DDP backward calls. Padding rows carry zero loss and zero accounting.
        per_rank = math.ceil(len(global_prompts) / world)
        padded = list(global_prompts)
        while len(padded) < per_rank * world:
            padded.append(global_prompts[-1])
        local = padded[rank * per_rank : (rank + 1) * per_rank]
        real_count = max(0, min(per_rank, len(global_prompts) - rank * per_rank))

        old_policy.load_state_dict(policy_model.state_dict())
        sampled = generate_batch(policy_model, local, sample=True)
        greedy = generate_batch(policy_model, local, sample=False)
        opponents = generate_batch(magnet, local, sample=True)
        pairs = []
        pair_prompts = []
        for index in range(real_count):
            pair_prompts.extend((local[index], local[index]))
            pairs.extend(
                (
                    (sampled[index]["text"], opponents[index]["text"]),
                    (greedy[index]["text"], opponents[index]["text"]),
                )
            )
        scores = judge.compare(pair_prompts, pairs) if pairs else []
        advantages = [scores[index] - scores[index + 1] for index in range(0, len(scores), 2)]
        advantages.extend([0.0] * (per_rank - real_count))
        token_counts = [max(int(row["completion_ids"].numel()), 1) for row in sampled]

        for inner in range(2):
            policy_model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            micro_count = math.ceil(per_rank / args.train_micro_batch)
            for micro_index, offset in enumerate(
                range(0, per_rank, args.train_micro_batch)
            ):
                end = min(offset + args.train_micro_batch, per_rank)
                sync = micro_index == micro_count - 1
                context = nullcontext() if sync else policy.no_sync()
                with context:
                    logps, magnet_kls, trust_kls = objective(sampled[offset:end])
                    advantage = torch.tensor(
                        advantages[offset:end], dtype=torch.float32, device=device
                    )
                    lengths = torch.tensor(
                        token_counts[offset:end], dtype=torch.float32, device=device
                    )
                    real_mask = torch.tensor(
                        [1.0 if i < real_count else 0.0 for i in range(offset, end)],
                        dtype=torch.float32,
                        device=device,
                    )
                    losses = (
                        -advantage * (logps / lengths)
                        + 0.3 * magnet_kls
                        + trust_kls
                    ) * real_mask
                    # DDP averages across ranks; this scaling recovers the
                    # mean over the unpadded global prompt batch.
                    scaled = losses.sum() * world / len(global_prompts)
                    scaled.backward()
                    totals[0] += losses.detach().double().sum()
                    totals[1] += magnet_kls.detach().double().mul(real_mask).sum()
                    totals[2] += trust_kls.detach().double().mul(real_mask).sum()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            totals[3] = 2 * real_count
            totals[4] = sum(
                row["completion_ids"].numel()
                for group in (sampled[:real_count], greedy[:real_count], opponents[:real_count])
                for row in group
            )
            dist.all_reduce(totals)
            if rank == 0:
                records.append(
                    {
                        "step": outer_index * 2 + inner + 1,
                        "outer_step": outer_index + 1,
                        "inner_step": inner + 1,
                        "loss": float(totals[0] / len(global_prompts)),
                        "magnet_kl": float(totals[1] / len(global_prompts)),
                        "trust_region_kl": float(totals[2] / len(global_prompts)),
                        "preference_model_calls": int(totals[3]) if inner == 0 else 0,
                        "generated_tokens": int(totals[4]) if inner == 0 else 0,
                        "global_batch_size": len(global_prompts),
                        "learning_rate": args.learning_rate,
                    }
                )
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(records[-1], sort_keys=True) + "\n")
        magnet.load_state_dict(policy_model.state_dict())
        if (outer_index + 1) % args.checkpoint_steps == 0 and rank == 0:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(checkpoint_root / "policy", safe_serialization=True)
            magnet.save_pretrained(checkpoint_root / "magnet", safe_serialization=True)
            torch.save(
                {"outer_step": outer_index + 1, "optimizer": optimizer.state_dict()},
                state_path,
            )
        dist.barrier()

    if rank == 0:
        policy_model.save_pretrained(args.output / "actor_adapter", safe_serialization=True)
        tokenizer.save_pretrained(args.output / "actor_adapter")
        update = parameter_update(before, tensor, name=name)
        (args.output / "parameter_update.json").write_text(
            json.dumps(update, indent=2, sort_keys=True) + "\n"
        )
    dist.barrier()
    write_manifest(args, optimizer_steps=optimizer_steps)
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.method == "mpo":
        train_mpo(args)
    else:
        train_trainer_method(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"elapsed_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
