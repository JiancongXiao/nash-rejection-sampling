"""OpenRLHF AgentExecutor for a minimal, faithful Nash-RS PPO vertical slice."""

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
from nashrs.rejection import accept_gibbs_proposal, acceptance_probability


class AgentExecutor(AgentExecutorBase):
    """Generate current samples online and fixed-reference proposals locally.

    This executor runs inside OpenRLHF's vLLM Ray actor. The rollout and B1
    samples therefore use the weight-synchronized current policy. Reference
    proposals come from a separately loaded, fixed pretrained checkpoint.
    """

    def __init__(self) -> None:
        self.tau = float(os.environ.get("NASHRS_TAU", "1.0"))
        self.b1 = int(os.environ.get("NASHRS_B1", "2"))
        self.b2 = int(os.environ.get("NASHRS_B2", "2"))
        self.proposal_batch_size = int(
            os.environ.get("NASHRS_PROPOSAL_BATCH_SIZE", "4")
        )
        self.max_proposals = int(os.environ.get("NASHRS_MAX_PROPOSALS", "64"))
        self.seed = int(os.environ.get("NASHRS_SEED", "29"))
        self.reference_model_name = os.environ.get(
            "NASHRS_REFERENCE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self.reference_revision = os.environ.get("NASHRS_REFERENCE_REVISION")
        self.preference_model_name = os.environ.get(
            "NASHRS_PREFERENCE_MODEL",
            "OpenAssistant/reward-model-deberta-v3-large-v2",
        )
        self.preference_revision = os.environ.get("NASHRS_PREFERENCE_REVISION")
        components_json = os.environ.get("NASHRS_PREFERENCE_COMPONENTS_JSON")
        self.preference_components = json.loads(components_json) if components_json else None
        self._reference_model = None
        self._preference = None
        self._execution_index = 0
        if self.tau <= 0.0 or self.b1 <= 0 or self.b2 <= 0:
            raise ValueError("tau, B1, and B2 must be positive")

    def _ensure_models(self):
        if self._reference_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM

        self._reference_model = AutoModelForCausalLM.from_pretrained(
            self.reference_model_name,
            revision=self.reference_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        self._reference_model.eval()
        component_configs = self.preference_components or [
            {
                "kind": "pair",
                "model": self.preference_model_name,
                "revision": self.preference_revision,
                "batch_size": 8,
                "max_length": 512,
            }
        ]
        self._preference = build_preference_oracle(component_configs, device="cuda")

    @staticmethod
    def _truncate_prompt(prompt, sampling_params, max_length, tokenizer):
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        max_prompt_length = max_length - sampling_params.max_tokens
        return prompt_ids[-max_prompt_length:]

    async def _current_samples(self, llm_engine, prompt_ids, sampling_params):
        params = deepcopy(sampling_params)
        params.logprobs = None
        outputs = await asyncio.gather(
            *[
                llm_engine.generate(prompt_ids, deepcopy(params))
                for _ in range(self.b1)
            ]
        )
        return [output.outputs[0].text for output in outputs], outputs

    def _reference_samples(self, prompt_ids, count, sampling_params, tokenizer):
        import torch

        input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = self._reference_model.generate(
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
        current, current_outputs = await self._current_samples(
            llm_engine, prompt_ids, sampling_params
        )

        accepted = []
        proposals = 0
        acceptance_trials = 0
        reference_tokens = 0
        preference_calls = 0
        while len(accepted) < self.b2:
            if proposals >= self.max_proposals:
                raise RuntimeError("Nash-RS proposal budget exhausted")
            count = min(self.proposal_batch_size, self.max_proposals - proposals)
            candidates, token_count = self._reference_samples(
                prompt_ids, count, sampling_params, hf_tokenizer
            )
            reference_tokens += token_count
            proposals += count
            for candidate in candidates:
                acceptance_trials += 1
                probabilities = self._preference.compare(
                    [prompt] * self.b1,
                    current,
                    [candidate] * self.b1,
                )
                preference_calls += self.b1
                g_hat = sum(probabilities) / self.b1
                if accept_gibbs_proposal(g_hat, self.tau, rng):
                    accepted.append((candidate, g_hat))
                    if len(accepted) == self.b2:
                        break

        reward_probabilities = self._preference.compare(
            [prompt] * self.b2,
            [response] * self.b2,
            [candidate for candidate, _ in accepted],
        )
        preference_calls += self.b2
        reward = sum(reward_probabilities) / (self.tau * self.b2)

        observation_tokens = prompt_ids + list(rollout.token_ids)
        rollout_log_probs = None
        if sampling_params.logprobs is not None and rollout.logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_ids)
            for token_id, logprob_dict in zip(rollout.token_ids, rollout.logprobs):
                token_logprob = logprob_dict.get(token_id)
                rollout_log_probs.append(
                    token_logprob.logprob if token_logprob is not None else 0.0
                )

        current_tokens = sum(
            len(output.outputs[0].token_ids) for output in current_outputs
        )
        elapsed = time.perf_counter() - started
        return {
            "prompt": prompt,
            "label": "",
            "observation_tokens": observation_tokens,
            "action_ranges": [(len(prompt_ids), len(observation_tokens))],
            "rollout_log_probs": rollout_log_probs,
            "truncated": rollout.finish_reason == "length",
            "reward": float(reward),
            "scores": min(1.0, max(0.0, self.tau * reward)),
            "extra_logs": {
                "nashrs/preference_model_calls": float(preference_calls),
                "nashrs/policy_generations": float(self.b1),
                "nashrs/reference_generations": float(proposals),
                "nashrs/generated_tokens": float(current_tokens + reference_tokens),
                "nashrs/proposals": float(proposals),
                "nashrs/acceptance_trials": float(acceptance_trials),
                "nashrs/accepted": float(self.b2),
                "nashrs/acceptance_rate": float(self.b2 / acceptance_trials),
                "nashrs/proposal_efficiency": float(self.b2 / proposals),
                "nashrs/mean_accepted_g_hat": float(
                    sum(g_hat for _, g_hat in accepted) / self.b2
                ),
                "nashrs/rollout_finished_by_length": float(
                    rollout.finish_reason == "length"
                ),
                "nashrs/rollout_finished_by_stop": float(
                    rollout.finish_reason == "stop"
                ),
                "nashrs/wall_time_seconds": float(elapsed),
                "nashrs/gpu_hours": float(elapsed / 3600.0),
            },
        }
