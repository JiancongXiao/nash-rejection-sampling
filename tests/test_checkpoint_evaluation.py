from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "examples" / "evaluate_ppo_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ppo_checkpoints", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CheckpointEvaluationTest(unittest.TestCase):
    def test_discovers_checkpoints_in_step_order(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            base = root / "base"
            checkpoints = root / "checkpoints"
            final = root / "final"
            for path in (
                base,
                checkpoints / "global_step16_hf",
                checkpoints / "global_step8_hf",
                final,
            ):
                path.mkdir(parents=True)
                (path / "config.json").write_text("{}")
            models = MODULE.discover_models(base, checkpoints, final)
            self.assertEqual([name for name, _ in models], ["base", "step_8", "step_16"])
            selected = MODULE.discover_models(base, checkpoints, final, {16})
            self.assertEqual([name for name, _ in selected], ["base", "step_16"])

    def test_loads_jsonl_prompts(self):
        with tempfile.TemporaryDirectory() as root_text:
            path = Path(root_text) / "prompts.jsonl"
            path.write_text(
                json.dumps({"prompt": "first"})
                + "\n"
                + json.dumps({"prompt": "second"})
                + "\n"
            )
            self.assertEqual(MODULE.load_prompts(path), ["first", "second"])

    def test_bootstrap_intervals_are_deterministic(self):
        methods = ["a", "b"]
        scores = {"a": [1.0, 2.0], "b": [0.0, 1.0]}
        per_prompt = [
            [[0.5, 0.5], [0.75, 0.75]],
            [[0.25, 0.25], [0.5, 0.5]],
        ]
        first = MODULE.bootstrap_intervals(methods, scores, per_prompt, 50, 7)
        second = MODULE.bootstrap_intervals(methods, scores, per_prompt, 50, 7)
        self.assertEqual(first, second)
        self.assertEqual(first["average_win_rate"]["a"], [0.75, 0.75])

    def test_pairwise_bootstrap_without_scalar_proxy(self):
        methods = ["base", "step_24"]
        per_prompt = [
            [[0.5, 0.5], [0.4, 0.6]],
            [[0.6, 0.4], [0.5, 0.5]],
        ]
        result = MODULE.bootstrap_intervals(methods, None, per_prompt, 20, 3)
        self.assertNotIn("mean_scalar_reward", result)
        self.assertIn("average_win_rate", result)

    def test_component_reward_intervals(self):
        scores = {
            "helpfulness": {
                "base": [1.0, 2.0],
                "step_24": [2.0, 3.0],
            }
        }
        result = MODULE.bootstrap_component_intervals(scores, 20, 11)
        self.assertIn("helpfulness", result)
        self.assertIn("step_24", result["helpfulness"])


if __name__ == "__main__":
    unittest.main()
