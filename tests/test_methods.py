from __future__ import annotations

import unittest

from nashrs.methods import (
    COMALPPOConfig,
    COMALPPOController,
    EGPPOConfig,
    EGPPOController,
    MPOPPOConfig,
    MPOPPOController,
    NashMDPPOConfig,
    NashMDPPOController,
    RewardPPOController,
    SelfPlayPPOConfig,
    SelfPlayPPOController,
)
from nashrs.methods.registry import ALL_METHODS


class NamedSampler:
    def __init__(self, name: str) -> None:
        self.name = name

    def sample(self, prompt: str, n: int):
        del prompt
        return [self.name] * n


class ConstantPreference:
    def __init__(self, value: float = 0.75) -> None:
        self.value = value

    def compare(self, prompts, left, right):
        assert len(prompts) == len(left) == len(right)
        return [self.value] * len(prompts)


class ScalarOracle:
    def score(self, prompts, responses):
        assert len(prompts) == len(responses)
        return [1.5, -0.5][: len(prompts)]


class FakeMixture:
    def __init__(self) -> None:
        self.calls = []

    def sample_geometric(self, prompt, n, current, reference, reference_weight):
        self.calls.append((prompt, current.name, reference.name, reference_weight))
        return [f"mix:{reference_weight}"] * n


class Snapshotter:
    def __init__(self) -> None:
        self.labels = []

    def __call__(self, policy, label):
        self.labels.append((policy.name, label))
        return NamedSampler(f"snapshot:{policy.name}:{label}")


class MethodTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = NamedSampler("reference")
        self.current = NamedSampler("current")
        self.oracle = ConstantPreference()

    def test_registry_contains_confirmed_methods(self) -> None:
        self.assertEqual(
            ALL_METHODS,
            (
                "reference",
                "reward_ppo",
                "self_play_ppo",
                "nash_md_ppo",
                "nash_rs",
                "mpo_ppo",
                "egpo_ppo",
                "comal_ppo",
            ),
        )

    def test_reward_ppo(self) -> None:
        method = RewardPPOController(ScalarOracle(), self.reference)
        batch = method.construct("main", ["p1", "p2"], ["a", "b"], self.current)
        self.assertEqual(batch.rewards, [1.5, -0.5])
        self.assertEqual(batch.scores, [1.0, 0.0])
        self.assertIs(method.phase_specs()[0].kl_reference, self.reference)

    def test_self_play_uses_current_policy_as_opponent(self) -> None:
        method = SelfPlayPPOController(SelfPlayPPOConfig(2), self.oracle, self.reference)
        batch = method.construct("main", ["p"], ["answer"], self.current)
        self.assertEqual([item.response for item in batch.opponents[0]], ["current", "current"])
        self.assertAlmostEqual(batch.rewards[0], 0.25)

    def test_nash_md_uses_geometric_mixture(self) -> None:
        mixture = FakeMixture()
        method = NashMDPPOController(
            NashMDPPOConfig(reference_weight=0.3, opponents_per_prompt=2),
            self.oracle,
            self.reference,
            mixture,
        )
        batch = method.construct("main", ["p"], ["answer"], self.current)
        self.assertEqual(mixture.calls, [("p", "current", "reference", 0.3)])
        self.assertEqual(batch.opponents[0][0].response, "mix:0.3")

    def test_mpo_synchronizes_magnet_and_opponent_on_interval(self) -> None:
        snapshots = Snapshotter()
        method = MPOPPOController(MPOPPOConfig(update_interval=2), self.oracle, self.reference, snapshots)
        initial = method.magnet_policy
        method.after_iteration(0, NamedSampler("updated-1"))
        self.assertIs(method.magnet_policy, initial)
        method.after_iteration(1, NamedSampler("updated-2"))
        self.assertIn("updated-2", method.magnet_policy.name)
        self.assertIs(method.magnet_policy, method.opponent_policy)
        self.assertIs(method.phase_specs()[0].kl_reference, method.magnet_policy)

    def test_egpo_prediction_correction_contract(self) -> None:
        snapshots = Snapshotter()
        method = EGPPOController(EGPPOConfig(), self.oracle, self.reference, snapshots)
        specs = method.phase_specs()
        self.assertEqual([spec.name for spec in specs], ["prediction", "correction"])
        self.assertEqual(specs[1].start_from, "iteration_start")
        with self.assertRaises(RuntimeError):
            method.construct("correction", ["p"], ["a"], self.current)
        method.after_phase("prediction", 3, NamedSampler("predictor"))
        batch = method.construct("correction", ["p"], ["a"], self.current)
        self.assertIn("predictor", batch.opponents[0][0].response)
        method.after_iteration(3, self.current)
        self.assertIsNone(method.predicted_policy)

    def test_comal_updates_anchor_only_at_outer_boundary(self) -> None:
        snapshots = Snapshotter()
        method = COMALPPOController(
            COMALPPOConfig(inner_iterations=2), self.oracle, self.reference, snapshots
        )
        initial = method.anchor_policy
        batch = method.construct("main", ["p"], ["a"], self.current)
        self.assertEqual(batch.opponents[0][0].response, "current")
        method.after_iteration(0, NamedSampler("updated-1"))
        self.assertIs(method.anchor_policy, initial)
        method.after_iteration(1, NamedSampler("updated-2"))
        self.assertIn("updated-2", method.anchor_policy.name)
        self.assertEqual(method.outer_round, 1)


if __name__ == "__main__":
    unittest.main()

