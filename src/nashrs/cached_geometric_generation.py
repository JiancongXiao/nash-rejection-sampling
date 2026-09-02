"""KV-cached geometric-mixture generation for method-native Nash-MD.

The TRL geometric-mixture wrapper intentionally disables the KV cache and
therefore recomputes both the policy and reference prefixes for every token.
For a PEFT policy the reference distribution is the same base model with the
adapter disabled.  This module keeps separate policy/reference KV caches while
sharing that base model, without changing the geometric-mixture distribution.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def geometric_mixture_logits(policy_logits, reference_logits, mixture_coef: float):
    """Return logits for ``pi_ref**a * pi_policy**(1-a)``.

    Model logits may be used directly: normalizing each component before the
    weighted sum only changes the result by a token-independent constant.
    """

    if not 0.0 <= mixture_coef <= 1.0:
        raise ValueError(f"mixture_coef must be in [0, 1], got {mixture_coef}")
    return mixture_coef * reference_logits + (1.0 - mixture_coef) * policy_logits


@contextmanager
def _reference_mode(model) -> Iterator[None]:
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        raise TypeError(
            "cached shared-base generation requires a PEFT model exposing "
            "disable_adapter()"
        )
    with disable_adapter():
        yield


def _cached_next_logits(
    model,
    generated,
    attention_mask,
    *,
    past_key_values=None,
):
    import torch

    if past_key_values is None:
        cache_position = torch.arange(
            generated.shape[1], device=generated.device, dtype=torch.long
        )
    else:
        cache_position = torch.tensor(
            [generated.shape[1] - 1], device=generated.device, dtype=torch.long
        )
    prepared = model.prepare_inputs_for_generation(
        generated,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
    )
    outputs = model(**prepared, return_dict=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


def cached_geometric_generate(
    model,
    input_ids,
    attention_mask,
    *,
    generation_config: Any,
    mixture_coef: float,
    generator=None,
):
    """Sample a geometric policy/reference mixture with two independent caches.

    The function currently targets the method-native experiment contract:
    sampling with temperature, ``top_k=0``, ``top_p=1``, one sequence per
    prompt, and a PEFT policy whose disabled adapter is the frozen reference.
    These are the exact settings constructed by TRL's ``OnlineDPOTrainer`` in
    the pinned Main environment.
    """

    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("cached Nash-MD generation currently requires batch size 1")
    if not bool(getattr(generation_config, "do_sample", True)):
        raise ValueError("cached Nash-MD generation requires do_sample=True")
    if int(getattr(generation_config, "top_k", 0) or 0) != 0:
        raise ValueError("cached Nash-MD generation requires top_k=0")
    if float(getattr(generation_config, "top_p", 1.0) or 1.0) != 1.0:
        raise ValueError("cached Nash-MD generation requires top_p=1.0")

    max_new_tokens = int(generation_config.max_new_tokens)
    temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos_ids: set[int] = set()
    elif isinstance(eos, int):
        eos_ids = {eos}
    else:
        eos_ids = {int(token) for token in eos}

    generated = input_ids
    mask = attention_mask
    policy_past = None
    reference_past = None

    with torch.inference_mode():
        policy_logits, policy_past = _cached_next_logits(
            model, generated, mask, past_key_values=policy_past
        )
        with _reference_mode(model):
            reference_logits, reference_past = _cached_next_logits(
                model, generated, mask, past_key_values=reference_past
            )

        for token_index in range(max_new_tokens):
            logits = geometric_mixture_logits(
                policy_logits, reference_logits, mixture_coef
            )
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            generated = torch.cat((generated, next_token), dim=-1)
            mask = torch.cat((mask, torch.ones_like(next_token)), dim=-1)

            if int(next_token.item()) in eos_ids or token_index + 1 == max_new_tokens:
                break

            policy_logits, policy_past = _cached_next_logits(
                model, generated, mask, past_key_values=policy_past
            )
            with _reference_mode(model):
                reference_logits, reference_past = _cached_next_logits(
                    model, generated, mask, past_key_values=reference_past
                )

    return generated
