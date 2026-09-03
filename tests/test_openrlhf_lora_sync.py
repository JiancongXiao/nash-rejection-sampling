from __future__ import annotations

import unittest

from nashrs.openrlhf_lora_sync import merged_vllm_parameters, vllm_weight_name


class FakeModel:
    def named_parameters(self):
        return iter(
            (
                ("base_model.model.model.embed_tokens.weight", object()),
                ("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight", object()),
                ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight", object()),
                ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight", object()),
                ("base_model.model.lm_head.weight", object()),
            )
        )


class OpenRLHFLoRASyncTest(unittest.TestCase):
    def test_peft_prefix_is_removed(self) -> None:
        self.assertEqual(
            vllm_weight_name("base_model.model.model.embed_tokens.weight"),
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            vllm_weight_name("base_model.model.lm_head.weight"), "lm_head.weight"
        )

    def test_adapter_tensors_are_not_sent_to_dense_vllm_model(self) -> None:
        self.assertIsNone(
            vllm_weight_name(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
            )
        )

    def test_dense_parameter_list_has_vllm_names_only(self) -> None:
        names = [name for name, _ in merged_vllm_parameters(FakeModel())]
        self.assertEqual(
            names,
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "lm_head.weight",
            ],
        )


if __name__ == "__main__":
    unittest.main()
