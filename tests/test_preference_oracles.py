import unittest
from unittest.mock import patch

from nashrs.preference_oracles import (
    BTLComponent,
    MixtureBTLPreferenceOracle,
    TransformersChatRewardOracle,
    build_preference_oracle,
)


class LengthReward:
    def __init__(self, sign=1.0):
        self.sign = sign

    def score(self, prompts, responses):
        del prompts
        return [self.sign * len(response) for response in responses]


class PreferenceOracleTest(unittest.TestCase):
    def test_btl_is_antisymmetric(self):
        oracle = MixtureBTLPreferenceOracle([BTLComponent(LengthReward())])
        forward = oracle.compare(["p"], ["long"], ["x"])[0]
        reverse = oracle.compare(["p"], ["x"], ["long"])[0]
        self.assertAlmostEqual(forward + reverse, 1.0)
        self.assertGreater(forward, 0.5)

    def test_mixture_combines_component_probabilities(self):
        oracle = MixtureBTLPreferenceOracle(
            [
                BTLComponent(LengthReward(1.0), weight=3.0),
                BTLComponent(LengthReward(-1.0), weight=1.0),
            ]
        )
        probability = oracle.compare(["p"], ["long"], ["x"])[0]
        self.assertGreater(probability, 0.5)

    def test_rejects_invalid_components(self):
        with self.assertRaises(ValueError):
            MixtureBTLPreferenceOracle([])
        with self.assertRaises(ValueError):
            BTLComponent(LengthReward(), temperature=0.0)

    def test_chat_reward_renders_user_and_assistant(self):
        calls = []

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return "rendered"

        oracle = TransformersChatRewardOracle.__new__(TransformersChatRewardOracle)
        oracle.tokenizer = Tokenizer()
        oracle.system_prompt = None
        self.assertEqual(oracle._render("question", "answer"), "rendered")
        self.assertEqual(calls[0][0], [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ])
        self.assertFalse(calls[0][1]["add_generation_prompt"])

    def test_factory_dispatches_pair_and_chat_adapters(self):
        with patch("nashrs.preference_oracles.TransformersScalarRewardOracle") as pair, patch(
            "nashrs.preference_oracles.TransformersChatRewardOracle"
        ) as chat:
            pair.return_value = LengthReward()
            chat.return_value = LengthReward(-1.0)
            mixture = build_preference_oracle([
                {"kind": "pair", "model": "pair-model"},
                {"kind": "chat", "model": "chat-model", "weight": 2.0},
            ])
        self.assertEqual(len(mixture.components), 2)
        pair.assert_called_once()
        chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
