from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest


class CachedGeometricGenerationTest(unittest.TestCase):
    def test_mixture_logits_match_geometric_policy(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is only installed in the Main training environment")
        from nashrs.cached_geometric_generation import geometric_mixture_logits

        policy = torch.tensor([[0.2, -0.4, 1.1]])
        reference = torch.tensor([[-0.8, 0.7, 0.3]])
        coefficient = 0.125
        actual = torch.softmax(
            geometric_mixture_logits(policy, reference, coefficient), dim=-1
        )
        expected = (
            torch.softmax(reference, dim=-1).pow(coefficient)
            * torch.softmax(policy, dim=-1).pow(1.0 - coefficient)
        )
        expected = expected / expected.sum(dim=-1, keepdim=True)
        torch.testing.assert_close(actual, expected)

    def test_cached_sampling_matches_full_prefix_recomputation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is only installed in the Main training environment")
        from nashrs.cached_geometric_generation import (
            cached_geometric_generate,
            geometric_mixture_logits,
        )

        class ToyAdapterModel:
            def __init__(self):
                self.adapter_enabled = True
                self.tokens_processed = 0
                self.generation_config = SimpleNamespace(eos_token_id=None)

            @contextmanager
            def disable_adapter(self):
                previous = self.adapter_enabled
                self.adapter_enabled = False
                try:
                    yield
                finally:
                    self.adapter_enabled = previous

            def prepare_inputs_for_generation(
                self, input_ids, attention_mask, past_key_values=None, **kwargs
            ):
                del kwargs
                if past_key_values is not None:
                    input_ids = input_ids[:, -1:]
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }

            def __call__(
                self,
                input_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            ):
                del attention_mask, use_cache, return_dict
                self.tokens_processed += input_ids.numel()
                prefix = (
                    torch.zeros(input_ids.shape[0], dtype=torch.long)
                    if past_key_values is None
                    else past_key_values
                )
                cumulative = prefix[:, None] + input_ids.cumsum(dim=-1)
                vocab = torch.arange(7, dtype=torch.float32)[None, None, :]
                centers = (cumulative % 7).to(torch.float32)[:, :, None]
                logits = -(vocab - centers).abs()
                if self.adapter_enabled:
                    logits[..., 3] += 0.75
                return SimpleNamespace(
                    logits=logits,
                    past_key_values=cumulative[:, -1],
                )

        def full_prefix_generate(model, input_ids, config, coefficient, generator):
            generated = input_ids
            for _ in range(config.max_new_tokens):
                policy = model(
                    generated, past_key_values=None, return_dict=True
                ).logits[:, -1, :]
                with model.disable_adapter():
                    reference = model(
                        generated, past_key_values=None, return_dict=True
                    ).logits[:, -1, :]
                logits = geometric_mixture_logits(policy, reference, coefficient)
                token = torch.multinomial(
                    torch.softmax(logits / config.temperature, dim=-1),
                    1,
                    generator=generator,
                )
                generated = torch.cat((generated, token), dim=-1)
            return generated

        config = SimpleNamespace(
            max_new_tokens=5,
            temperature=0.9,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            eos_token_id=None,
        )
        input_ids = torch.tensor([[1, 2, 4]])
        attention_mask = torch.ones_like(input_ids)

        cached_model = ToyAdapterModel()
        cached = cached_geometric_generate(
            cached_model,
            input_ids,
            attention_mask,
            generation_config=config,
            mixture_coef=0.125,
            generator=torch.Generator().manual_seed(1234),
        )
        full_model = ToyAdapterModel()
        full = full_prefix_generate(
            full_model,
            input_ids,
            config,
            0.125,
            torch.Generator().manual_seed(1234),
        )

        torch.testing.assert_close(cached, full)
        self.assertLess(cached_model.tokens_processed, full_model.tokens_processed)


if __name__ == "__main__":
    unittest.main()

