from __future__ import annotations

import unittest

from nashrs import NashRSConfig, NashRSRewardConstructor


class FixedSampler:
    def __init__(self, value: str) -> None:
        self.value = value

    def sample(self, prompt: str, n: int) -> list[str]:
        del prompt
        return [self.value] * n


class ConstantOracle:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def compare(self, prompts, left, right):
        assert len(prompts) == len(left) == len(right)
        return [self.probability] * len(prompts)


class NashRSTest(unittest.TestCase):
    def test_implicit_reward_and_accounting(self) -> None:
        constructor = NashRSRewardConstructor(
            NashRSConfig(tau=2.0, b1=3, b2=2, seed=3, proposal_batch_size=1),
            FixedSampler("policy"),
            FixedSampler("reference"),
            ConstantOracle(0.0),
        )
        batch = constructor.construct(["p", "p"], ["r1", "r2"])
        self.assertEqual(batch.rewards, [0.0, 0.0])
        self.assertEqual(batch.accounting.policy_generations, 3)
        self.assertEqual(batch.accounting.reference_generations, 2)
        self.assertEqual(batch.accounting.accepted, 2)
        self.assertEqual(batch.accounting.preference_model_calls, 3 * 2 + 2 * 2)

    def test_reward_is_scaled_by_inverse_tau(self) -> None:
        constructor = NashRSRewardConstructor(
            NashRSConfig(tau=2.0, b1=1, b2=1, seed=1),
            FixedSampler("policy"),
            FixedSampler("reference"),
            ConstantOracle(0.5),
        )
        batch = constructor.construct(["p"], ["response"])
        self.assertAlmostEqual(batch.rewards[0], 0.25)


if __name__ == "__main__":
    unittest.main()

