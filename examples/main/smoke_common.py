"""Shared, framework-neutral utilities for Main 0.5B smoke runs."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
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
    # ``generate`` ran under inference mode, so its result is an inference
    # tensor.  Native MPO/COMAL subsequently reuse these token ids in a
    # differentiable forward pass; clone outside the context to make ordinary
    # tensors that autograd is allowed to save for backward.
    full_ids = output[0].detach().clone()
    continuation = full_ids[prompt_ids.shape[1] :].clone()
    return {
        "prompt_ids": prompt_ids[0].detach().clone(),
        "full_ids": full_ids,
        "completion_ids": continuation,
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


class PreferenceSidecar:
    """Run the shared preference oracle in a separately pinned Python stack."""

    _READY_PREFIX = "NASHRS_READY\t"
    _RESULT_PREFIX = "NASHRS_RESULT\t"

    def __init__(self, component_configs: list[dict], python: str) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        log_path = Path(
            os.environ.get(
                "NASHRS_PREFERENCE_SIDECAR_LOG",
                str(repo_root / "preference_sidecar.log"),
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w")
        self._process = subprocess.Popen(
            [
                python,
                "-m",
                "examples.main.preference_sidecar",
                "--components-json",
                json.dumps(component_configs, separators=(",", ":")),
            ],
            cwd=repo_root,
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )
        ready = self._read(self._READY_PREFIX)
        self.component_count = int(ready["component_count"])
        atexit.register(self.close)

    def _read(self, prefix: str):
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :])
            print(f"preference sidecar: {line.rstrip()}", file=sys.stderr)
        raise RuntimeError(
            "preference sidecar exited before returning a response; "
            f"see {self._log.name}"
        )

    def compare(self, prompts, left, right) -> list[float]:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"preference sidecar exited with {self._process.returncode}; "
                f"see {self._log.name}"
            )
        assert self._process.stdin is not None
        self._process.stdin.write(
            json.dumps(
                {"prompts": list(prompts), "left": list(left), "right": list(right)},
                separators=(",", ":"),
            )
            + "\n"
        )
        self._process.stdin.flush()
        return [float(value) for value in self._read(self._RESULT_PREFIX)]

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        if not self._log.closed:
            self._log.close()
        self._process = None


class CountingJudgeState:
    def __init__(self, component_configs: list[dict], tokenizer) -> None:
        sidecar_python = os.environ.get("NASHRS_PREFERENCE_PYTHON")
        if sidecar_python:
            self.oracle = PreferenceSidecar(component_configs, sidecar_python)
            self.component_count = self.oracle.component_count
        else:
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


def build_metrics_callback(output: Path, judge, *, append_existing: bool = False):
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.started = time.perf_counter()
            self.last_time = self.started
            self.last_calls = 0
            self.last_tokens = 0
            self.records = []
            if append_existing and output.exists():
                self.records = [
                    json.loads(line)
                    for line in output.read_text().splitlines()
                    if line.strip()
                ]
            self.last_step = max(
                (int(record["step"]) for record in self.records), default=0
            )

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
