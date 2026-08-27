"""Unified OpenRLHF AgentExecutor for PPO comparison adaptations."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import math
import os
import random
import time

from openrlhf.utils.agent import AgentExecutorBase

from nashrs import build_preference_oracle
from nashrs.geometric_resampling import (
    effective_sample_size,
    geometric_importance_weights,
    resample_indices,
)


SUPPORTED_METHODS = {
    "reward_ppo",
    "self_play_ppo",
    "nash_md_ppo",
    "mpo_ppo",
    "egpo_ppo",
    "comal_ppo",
}


class AgentExecutor(AgentExecutorBase):
    """Construct method-specific rewards while sharing one PPO backend.

    MPO is one fixed-magnet interval and COMAL is one fixed-anchor outer
    round. EGPO is launched twice by PBS: prediction against the iteration
    start policy, then correction restored to that same policy and evaluated
    against the frozen predictor.
    """

    def __init__(self) -> None:
        self.method = os.environ.get("NASHRS_METHOD", "self_play_ppo")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported PPO comparison method: {self.method}")
        self.opponents_per_prompt = int(os.environ.get("NASHRS_OPPONENTS", "2"))
        self.geometric_pool_size = int(os.environ.get("NASHRS_GEOMETRIC_POOL", "4"))
        self.reference_weight = float(os.environ.get("NASHRS_REFERENCE_WEIGHT", "0.25"))
        self.seed = int(os.environ.get("NASHRS_SEED", "47"))
        self.reference_model_name = os.environ.get(
            "NASHRS_REFERENCE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.reference_revision = os.environ.get("NASHRS_REFERENCE_REVISION")
        self.opponent_model_name = os.environ.get(
            "NASHRS_OPPONENT_MODEL", self.reference_model_name
        )
        components_json = os.environ.get("NASHRS_PREFERENCE_COMPONENTS_JSON")
        if not components_json:
            raise ValueError("NASHRS_PREFERENCE_COMPONENTS_JSON is required")
        self.preference_components = json.loads(components_json)
        self._opponent_model = None
        self._preference = None
        self._execution_index = 0
        if self.opponents_per_prompt <= 0 or self.geometric_pool_size < 2:
            raise ValueError("opponent and geometric pool sizes must be positive")
        if not 0.0 <= self.reference_weight <= 1.0:
            raise ValueError("reference weight must lie in [0, 1]")

    def _ensure_models(self) -> None:
        if self._preference is None:
            self._preference = build_preference_oracle(
                self.preference_components, device="cuda"
            )
        if self.method not in {"nash_md_ppo", "mpo_ppo", "egpo_ppo"}:
            return
        if self._opponent_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM

        self._opponent_model = AutoModelForCausalLM.from_pretrained(
            self.opponent_model_name,
            revision=self.reference_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        self._opponent_model.eval()

    @staticmethod
    def _truncate_prompt(prompt, sampling_params, max_length, tokenizer):
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        max_prompt_length = max_length - sampling_params.max_tokens
        return prompt_ids[-max_prompt_length:]

    async def _current_samples(self, llm_engine, prompt_ids, sampling_params, count):
        params = deepcopy(sampling_params)
        params.logprobs = None
        outputs = await asyncio.gather(
            *[llm_engine.generate(prompt_ids, deepcopy(params)) for _ in range(count)]
        )
        return [output.outputs[0].text for output in outputs], outputs

    def _fixed_samples(self, prompt_ids, count, sampling_params, tokenizer):
        import torch

        input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = self._opponent_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=max(float(sampling_params.temperature), 1e-5),
                top_p=float(sampling_params.top_p),
                max_new_tokens=int(sampling_params.max_tokens),
                num_return_sequences=count,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        continuations = generated[:, input_ids.shape[1] :]
        return tokenizer.batch_decode(continuations, skip_special_tokens=True), int(
            continuations.numel()
        )

    async def _current_log_probability(
        self, llm_engine, full_ids, candidate_start, sampling_params
    ) -> float:
        params = deepcopy(sampling_params)
        params.max_tokens = 1
        params.temperature = 0.0
        params.top_p = 1.0
        params.logprobs = None
        params.prompt_logprobs = 1
        output = await llm_engine.generate(full_ids, params)
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return requested prompt log-probabilities")
        total = 0.0
        for position in range(candidate_start, len(full_ids)):
            values = prompt_logprobs[position]
            token_id = full_ids[position]
            if values is None or token_id not in values:
                raise RuntimeError("candidate token is missing from vLLM prompt logprobs")
            total += float(values[token_id].logprob)
        return total

    def _fixed_log_probability(self, full_ids, candidate_start) -> float:
        import torch

        input_ids = torch.tensor([full_ids], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            logits = self._opponent_model(input_ids=input_ids).logits.float()
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            labels = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        start = max(candidate_start - 1, 0)
        return float(token_log_probs[0, start:].sum().item())

    async def _geometric_opponents(
        self,
        llm_engine,
        prompt_ids,
        sampling_params,
        tokenizer,
        max_length,
        rng,
    ):
        current_count = self.geometric_pool_size // 2
        fixed_count = self.geometric_pool_size - current_count
        current, current_outputs = await self._current_samples(
            llm_engine, prompt_ids, sampling_params, current_count
        )
        fixed, fixed_tokens = self._fixed_samples(
            prompt_ids, fixed_count, sampling_params, tokenizer
        )
        candidates = current + fixed
        candidate_ids = [
            tokenizer(value, add_special_tokens=False)["input_ids"] for value in candidates
        ]
        maximum_candidate_length = max(max_length - len(prompt_ids) - 1, 1)
        candidate_ids = [values[:maximum_candidate_length] for values in candidate_ids]
        candidates = [
            tokenizer.decode(values, skip_special_tokens=True) for values in candidate_ids
        ]
        full_sequences = [prompt_ids + values for values in candidate_ids]
        current_log_probs = await asyncio.gather(
            *[
                self._current_log_probability(
                    llm_engine, values, len(prompt_ids), sampling_params
                )
                for values in full_sequences
            ]
        )
        fixed_log_probs = [
            self._fixed_log_probability(values, len(prompt_ids))
            for values in full_sequences
        ]
        weights = geometric_importance_weights(
            current_log_probs, fixed_log_probs, self.reference_weight
        )
        indices = resample_indices(weights, self.opponents_per_prompt, rng)
        current_tokens = sum(
            len(output.outputs[0].token_ids) for output in current_outputs
        )
        return (
            [candidates[index] for index in indices],
            current_count,
            fixed_count,
            current_tokens + fixed_tokens,
            effective_sample_size(weights),
        )

    def _scalar_reward(self, prompt: str, response: str):
        weighted = 0.0
        total_weight = 0.0
        component_values = {}
        for index, component in enumerate(self._preference.components):
            value = float(component.oracle.score([prompt], [response])[0])
            normalized = value / component.temperature
            weighted += component.weight * normalized
            total_weight += component.weight
            component_values[
                self.preference_components[index].get("name", f"component_{index}")
            ] = value
        utility = weighted / total_weight
        score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, utility))))
        return utility, score, component_values

    @staticmethod
    def _rollout_payload(prompt, prompt_ids, rollout, reward, score, extra_logs):
        observation_tokens = prompt_ids + list(rollout.token_ids)
        rollout_log_probs = None
        if rollout.logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_ids)
            for token_id, logprob_dict in zip(rollout.token_ids, rollout.logprobs):
                token_logprob = logprob_dict.get(token_id)
                rollout_log_probs.append(
                    token_logprob.logprob if token_logprob is not None else 0.0
                )
        return {
            "prompt": prompt,
            "label": "",
            "observation_tokens": observation_tokens,
            "action_ranges": [(len(prompt_ids), len(observation_tokens))],
            "rollout_log_probs": rollout_log_probs,
            "truncated": rollout.finish_reason == "length",
            "reward": float(reward),
            "scores": float(score),
            "extra_logs": extra_logs,
        }

    async def execute(
        self,
        prompt,
        label,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        llm_engine,
    ):
        del label
        started = time.perf_counter()
        self._ensure_models()
        execution_index = self._execution_index
        self._execution_index += 1
        rng = random.Random(self.seed + execution_index)
        prompt_ids = self._truncate_prompt(
            prompt, sampling_params, max_length, hf_tokenizer
        )
        rollout_output = await llm_engine.generate(
            prompt_ids, deepcopy(sampling_params)
        )
        rollout = rollout_output.outputs[0]
        response = rollout.text

        preference_calls = 0
        current_generations = 0
        reference_generations = 0
        generated_tokens = 0
        diagnostics = {}
        component_count = len(self._preference.components)

        if self.method == "reward_ppo":
            reward, score, component_values = self._scalar_reward(prompt, response)
            preference_calls = component_count
            diagnostics.update(
                {f"{self.method}/component_{name}": value for name, value in component_values.items()}
            )
        else:
            if self.method in {"self_play_ppo", "comal_ppo"}:
                opponents, outputs = await self._current_samples(
                    llm_engine,
                    prompt_ids,
                    sampling_params,
                    self.opponents_per_prompt,
                )
                current_generations = self.opponents_per_prompt
                generated_tokens = sum(
                    len(output.outputs[0].token_ids) for output in outputs
                )
            elif self.method in {"mpo_ppo", "egpo_ppo"}:
                opponents, generated_tokens = self._fixed_samples(
                    prompt_ids,
                    self.opponents_per_prompt,
                    sampling_params,
                    hf_tokenizer,
                )
                reference_generations = self.opponents_per_prompt
            else:
                (
                    opponents,
                    current_generations,
                    reference_generations,
                    generated_tokens,
                    ess,
                ) = await self._geometric_opponents(
                    llm_engine,
                    prompt_ids,
                    sampling_params,
                    hf_tokenizer,
                    max_length,
                    rng,
                )
                diagnostics[f"{self.method}/geometric_pool_ess"] = float(ess)

            probabilities = self._preference.compare(
                [prompt] * len(opponents),
                [response] * len(opponents),
                opponents,
            )
            preference_calls = len(opponents) * component_count
            reward = sum(float(value) for value in probabilities) / len(probabilities)
            score = reward

        elapsed = time.perf_counter() - started
        common_logs = {
            "preference_model_calls": float(preference_calls),
            "preference_components": float(component_count),
            "policy_generations": float(current_generations),
            "reference_generations": float(reference_generations),
            "opponent_generations": float(
                current_generations + reference_generations
            ),
            "generated_tokens": float(generated_tokens),
            "wall_time_seconds": float(elapsed),
            "gpu_hours": float(elapsed / 3600.0),
        }
        diagnostics.update(
            {f"{self.method}/{name}": value for name, value in common_logs.items()}
        )
        diagnostics.update(
            {f"comparison/{name}": value for name, value in common_logs.items()}
        )
        return self._rollout_payload(
            prompt, prompt_ids, rollout, reward, score, diagnostics
        )
