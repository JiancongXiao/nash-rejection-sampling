"""Shared, framework-neutral utilities for Main 0.5B smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
import time


def load_prompts(path: Path, count: int) -> list[str]:
    prompts = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        prompt = value.get("prompt") or value.get("input") or value.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt JSONL row has no text prompt: {value}")
        prompts.append(prompt)
        if len(prompts) == count:
            break
    if len(prompts) < count:
        raise ValueError(f"needed {count} prompts, found {len(prompts)} in {path}")
    return prompts


def first_trainable_tensor(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim >= 2:
            return name, parameter
    raise ValueError("model has no trainable matrix parameter")


def parameter_update(before, after, *, name: str) -> dict:
    import torch

    difference = after.detach().float().cpu() - before.detach().float().cpu()
    return {
        "tensor": name,
        "l2_update_norm": float(torch.linalg.vector_norm(difference).item()),
        "max_abs_update": float(difference.abs().max().item()),
        "changed_elements": int(torch.count_nonzero(difference).item()),
        "num_elements": int(difference.numel()),
    }


def render_prompt(tokenizer, prompt: str):
    import torch

    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids.to("cuda", dtype=torch.long)


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
):
    import torch

    prompt_ids = render_prompt(tokenizer, prompt)
    attention_mask = torch.ones_like(prompt_ids)
    generation_kwargs = {}
    if do_sample:
        generation_kwargs.update(temperature=1.0, top_p=0.95)
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    continuation = output[0, prompt_ids.shape[1] :]
    return {
        "prompt_ids": prompt_ids[0].detach(),
        "full_ids": output[0].detach(),
        "completion_ids": continuation.detach(),
        "text": tokenizer.decode(continuation, skip_special_tokens=True),
    }


def sequence_logprob(model, full_ids, prompt_length: int):
    """Differentiable summed completion log-probability."""

    import torch

    values = full_ids.unsqueeze(0).to(model.device)
    logits = model(input_ids=values).logits[:, :-1].float()
    labels = values[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    start = max(prompt_length - 1, 0)
    return token_logps[0, start:].sum()


def trajectory_forward_kl(policy, reference, full_ids, prompt_length: int):
    """Differentiable sequential KL(policy || reference) on one trajectory."""

    import torch

    values = full_ids.unsqueeze(0).to(policy.device)
    policy_logits = policy(input_ids=values).logits[:, :-1].float()
    with torch.no_grad():
        reference_logits = reference(input_ids=values).logits[:, :-1].float()
    start = max(prompt_length - 1, 0)
    policy_log_probs = torch.log_softmax(policy_logits[:, start:], dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits[:, start:], dim=-1)
    policy_probs = policy_log_probs.exp()
    return (policy_probs * (policy_log_probs - reference_log_probs)).sum()


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class CountingJudgeState:
    def __init__(self, component_configs: list[dict], tokenizer) -> None:
        from nashrs import build_preference_oracle

        self.oracle = build_preference_oracle(component_configs, device="cuda")
        self.component_count = len(self.oracle.components)
        self.tokenizer = tokenizer
        self.preference_model_calls = 0
        self.generated_tokens = 0

    def compare(self, prompts, completions):
        left = [pair[0] for pair in completions]
        right = [pair[1] for pair in completions]
        self.preference_model_calls += len(prompts) * self.component_count
        self.generated_tokens += sum(
            len(self.tokenizer(value, add_special_tokens=False)["input_ids"])
            for pair in completions
            for value in pair
        )
        return self.oracle.compare(prompts, left, right)


def build_counting_judge(component_configs: list[dict], tokenizer):
    """Create a TRL-compatible pairwise judge without pinning TRL at import time."""

    from trl import BasePairwiseJudge

    state = CountingJudgeState(component_configs, tokenizer)

    class CountingMixtureJudge(BasePairwiseJudge):
        def judge(self, prompts, completions, shuffle_order=True, **kwargs):
            del shuffle_order, kwargs
            return state.compare(prompts, completions)

    judge = CountingMixtureJudge()
    judge.accounting = state
    return judge


def build_metrics_callback(output: Path, judge):
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.started = time.perf_counter()
            self.last_time = self.started
            self.last_calls = 0
            self.last_tokens = 0
            self.last_step = 0
            self.records = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, control, kwargs
            if not logs or "loss" not in logs or state.global_step <= self.last_step:
                return
            now = time.perf_counter()
            calls = judge.accounting.preference_model_calls
            tokens = judge.accounting.generated_tokens
            record = {
                "step": int(state.global_step),
                "loss": float(logs["loss"]),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
                "preference_model_calls": int(calls - self.last_calls),
                "generated_tokens": int(tokens - self.last_tokens),
                "gpu_hours": float((now - self.last_time) / 3600.0),
            }
            for name, value in logs.items():
                if name not in record and isinstance(value, (int, float)):
                    record[name] = float(value)
            self.records.append(record)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w") as handle:
                for item in self.records:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
            self.last_step = int(state.global_step)
            self.last_calls = calls
            self.last_tokens = tokens
            self.last_time = now

    return MetricsCallback()
