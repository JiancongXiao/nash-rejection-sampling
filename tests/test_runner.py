from __future__ import annotations

import unittest

from nashrs.methods import EGPPOConfig, EGPPOController
from nashrs.runner import run_iteration


class Policy:
    def __init__(self, name):
        self.name = name

    def sample(self, prompt, n):
        del prompt
        return [self.name] * n


class Oracle:
    def compare(self, prompts, left, right):
        del left, right
        return [0.75] * len(prompts)


class Backend:
    def __init__(self):
        self.policy = Policy("pi-0")
        self.events = []

    def snapshot_current(self, label):
        self.events.append(("snapshot", label, self.policy.name))
        return Policy(self.policy.name)

    def restore(self, policy):
        self.events.append(("restore", policy.name))
        self.policy = Policy(policy.name)

    def current_policy(self):
        return self.policy

    def rollout(self, prompts):
        self.events.append(("rollout", self.policy.name))
        return [f"answer:{self.policy.name}"] * len(prompts)

    def ppo_update(self, prompts, responses, rewards, phase):
        del prompts, responses, rewards
        self.events.append(("update", phase.name, self.policy.name))
        self.policy = Policy(f"{self.policy.name}+{phase.name}")


class RunnerTest(unittest.TestCase):
    def test_egpo_correction_restores_iteration_start(self):
        backend = Backend()

        def snapshot(policy, label):
            return Policy(f"frozen:{policy.name}:{label}")

        method = EGPPOController(EGPPOConfig(), Oracle(), Policy("ref"), snapshot)
        result = run_iteration(method, backend, ["prompt"], iteration=0)
        self.assertEqual([phase.phase for phase in result.phases], ["prediction", "correction"])
        self.assertIn(("restore", "pi-0"), backend.events)
        correction_update = [event for event in backend.events if event[:2] == ("update", "correction")]
        self.assertEqual(correction_update[0][2], "pi-0")
        self.assertEqual(backend.policy.name, "pi-0+correction")


if __name__ == "__main__":
    unittest.main()

